import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from typing import List, Optional
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)
warnings.filterwarnings("ignore", message=".*video decoding.*")


# ==========================================
# 1. Iterative Reasoning Adapter
#    核心改动：同一个 adapter，跑 1 次 = fast path，跑 K 次 = slow path
#    这样 fast / slow 才有真实的计算深度差异，gate 才有意义
# ==========================================
class IterativeReasoningAdapter(nn.Module):
    def __init__(self, dim, num_tokens=16, heads=8):
        super().__init__()
        self.reason_tokens = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        # 读写分开两个 attn，避免参数互相干扰（之前共享一个是历史包袱）
        self.write_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.read_attn  = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm_r = nn.LayerNorm(dim)
        self.norm_h = nn.LayerNorm(dim)

        # 零初始化最后一层 → 初始 num_iters 次迭代都≈identity，安全启动
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h, num_iters: int = 1):
        """
        h: [B, K, D]  action token hidden states
        num_iters: 迭代次数。1 = fast path, K(>1) = slow path
        """
        B = h.shape[0]
        for _ in range(num_iters):
            # 写：从 h 中读信息进 reason tokens
            r = self.reason_tokens.expand(B, -1, -1)
            r_upd, _ = self.write_attn(query=r, key=h, value=h)
            r = self.norm_r(r + r_upd)
            # 读：reason tokens 把推理结果写回 h
            h_upd, _ = self.read_attn(query=h, key=r, value=r)
            h = self.norm_h(h + h_upd + self.mlp(h_upd))
        return h


# ==========================================
# 2. Gate: 决定 fast/slow 路径的混合权重
# ==========================================
class DifficultyGate(nn.Module):
    def __init__(self, scene_dim: int, state_dim: int = 8):
        super().__init__()
        self.state_dim = state_dim
        self.use_state = state_dim > 0

        in_dim = scene_dim
        if self.use_state:
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Linear(32, 16),
            )
            in_dim += 16

        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, scene_feat: torch.Tensor, state: Optional[torch.Tensor] = None):
        if self.use_state:
            if state is None:
                state = torch.zeros(
                    (scene_feat.shape[0], self.state_dim),
                    device=scene_feat.device, dtype=scene_feat.dtype,
                )
            cur = state[:, -1, :] if state.dim() == 3 else state
            s_feat = self.state_mlp(cur.to(scene_feat.dtype))
            x = torch.cat([scene_feat, s_feat], dim=-1)
        else:
            x = scene_feat
        return torch.sigmoid(self.net(x))


@FRAMEWORK_REGISTRY.register("CIGVLA")
class CIGVLA(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.vlm_instance = get_vlm_model(config=self.config)

        print("[CIG-VLA v2] Iterative reasoning mode: fast=1 iter / slow=K iters, gate-routed.")

        # ====== Action tokens ======
        qwenvl_cfg = getattr(self.config.framework, "qwenvl", {})
        if isinstance(qwenvl_cfg, dict):
            self.num_action_tokens = qwenvl_cfg.get("num_latent_action_query", 64)
        else:
            self.num_action_tokens = getattr(qwenvl_cfg, "num_latent_action_query", 64)

        self.action_tokens_list = [f"<|action_{i}|>" for i in range(self.num_action_tokens)]
        self.action_tokens_str = "".join(self.action_tokens_list)

        tokenizer = self.vlm_instance.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.convert_tokens_to_ids(self.action_tokens_list[0]) is None:
            num_added = tokenizer.add_tokens(self.action_tokens_list, special_tokens=True)
            if num_added > 0:
                self.vlm_instance.model.resize_token_embeddings(len(tokenizer))
                emb = self.vlm_instance.model.get_input_embeddings().weight.data
                with torch.no_grad():
                    emb[-num_added:] = emb[:-num_added].mean(dim=0, keepdim=True)

        # ====== Dim wiring ======
        vlm_config = self.vlm_instance.model.config
        vlm_hidden_size = getattr(vlm_config, "hidden_size", 2048)
        if hasattr(vlm_config, "text_config"):
            vlm_hidden_size = getattr(vlm_config.text_config, "hidden_size", vlm_hidden_size)

        action_cfg = getattr(self.config.framework, "action_model", {})
        if isinstance(action_cfg, dict):
            self.state_dim = action_cfg.get("state_dim", 8)
            diff_cfg = action_cfg.get("diffusion_model_cfg", {})
            target_action_dim = diff_cfg.get("cross_attention_dim", 2048) if isinstance(diff_cfg, dict) \
                else getattr(diff_cfg, "cross_attention_dim", 2048)
            self.future_window = action_cfg.get("future_action_window_size", 7)
        else:
            self.state_dim = getattr(action_cfg, "state_dim", 8)
            diff_cfg = getattr(action_cfg, "diffusion_model_cfg", {})
            target_action_dim = diff_cfg.get("cross_attention_dim", 2048) if isinstance(diff_cfg, dict) \
                else getattr(diff_cfg, "cross_attention_dim", 2048)
            self.future_window = getattr(action_cfg, "future_action_window_size", 7)

        # ====== Modules ======
        self.adapter = IterativeReasoningAdapter(target_action_dim, num_tokens=16).to(torch.float32)
        self.gate    = DifficultyGate(scene_dim=vlm_hidden_size, state_dim=self.state_dim).to(torch.float32)
        self.action_header = get_action_model(config=self.config)

        if vlm_hidden_size != target_action_dim:
            self.vlm_to_action_proj = nn.Linear(vlm_hidden_size, target_action_dim)
        else:
            self.vlm_to_action_proj = nn.Identity()

        # ====== Hyper-params ======
        self.slow_iters       = int(getattr(self.config.framework, "slow_iters", 3))   # K
        self.fast_loss_weight = float(getattr(self.config.framework, "fast_loss_weight", 0.3))
        self.gate_loss_weight = float(getattr(self.config.framework, "gate_loss_weight", 0.1))

        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _extract_action_tokens(self, hidden, input_ids):
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.as_tensor(input_ids, device=hidden.device, dtype=torch.long)
        B, L, D = hidden.shape
        tokenizer = self.vlm_instance.processor.tokenizer
        start_id = tokenizer.convert_tokens_to_ids("<|action_0|>")
        mask = (input_ids == start_id)
        matched = mask.nonzero(as_tuple=True)

        if len(matched[0]) == 0:
            return torch.zeros((B, self.num_action_tokens, D), device=hidden.device, dtype=hidden.dtype)

        start_indices = torch.full((B,), L - self.num_action_tokens, device=hidden.device, dtype=torch.long)
        for b in range(B):
            sample = matched[1][matched[0] == b]
            if len(sample) > 0:
                start_indices[b] = sample[-1]

        offsets = torch.arange(self.num_action_tokens, device=hidden.device).unsqueeze(0)
        idx = torch.clamp(start_indices.unsqueeze(1) + offsets, max=L - 1)
        idx_ext = idx.unsqueeze(-1).expand(-1, -1, D)
        action_tokens = torch.gather(hidden, 1, idx_ext)

        proj_dtype = self.vlm_to_action_proj.weight.dtype \
            if isinstance(self.vlm_to_action_proj, nn.Linear) else action_tokens.dtype
        return self.vlm_to_action_proj(action_tokens.to(proj_dtype)).to(action_tokens.dtype)

    @staticmethod
    def _prep_images(examples):
        imgs = []
        for ex in examples:
            raw = ex["image"]
            if not isinstance(raw, (list, tuple)):
                raw = [raw]
            sample = []
            for x in raw:
                if isinstance(x, np.ndarray):
                    if x.ndim == 3 and x.shape[0] in [1, 3, 4]:
                        x = np.transpose(x, (1, 2, 0))
                    sample.append(Image.fromarray(x.astype("uint8")))
                else:
                    sample.append(x)
            imgs.append(sample)
        return imgs

    @staticmethod
    def _prep_state(examples, device, dtype=torch.float32):
        state_list = [ex.get("state", None) for ex in examples]
        if state_list[0] is None:
            return None
        st = torch.tensor(np.array(state_list), device=device).to(dtype)
        if st.dim() == 2:
            st = st.unsqueeze(1)
        elif st.dim() == 3:
            st = st[:, -1:, :]
        return st

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        if self.training:
            self.global_step += 1

        imgs = self._prep_images(examples)
        raw_langs = [ex["lang"] for ex in examples]
        # 注：去掉了花哨的 ADVANCED_TASK_TEMPLATE，VLM 没做 CoT 监督，prompt 戏剧化无意义
        prompts = [l + self.action_tokens_str for l in raw_langs]
        device = next(self.parameters()).device
        state = self._prep_state(examples, device)

        # ====== 1. VLM 单次前向（保留你原设计） ======
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs = self.vlm_instance.build_qwenvl_inputs(imgs, prompts)
            out = self.vlm_instance(**inputs, output_hidden_states=True)
            hidden_last = out.hidden_states[-1]
            scene_feat = hidden_last[:, :-self.num_action_tokens].mean(dim=1).float()
            h_base = self._extract_action_tokens(hidden_last, inputs["input_ids"])

        with torch.autocast("cuda", dtype=torch.float32):
            h_base = h_base.float()

            # ====== 2. Fast path (1 iter) 与 Slow path (K iters) ======
            #   核心改动：两条路径用同一个 adapter，差别只在迭代次数 → 真实计算深度差异
            h_fast = self.adapter(h_base, num_iters=1)
            h_slow = self.adapter(h_base, num_iters=self.slow_iters)

            # ====== 3. Gate 决定混合权重 ======
            g = self.gate(scene_feat.detach(), state.squeeze(1) if state is not None else None)
            h_final = (1.0 - g.unsqueeze(-1)) * h_fast + g.unsqueeze(-1) * h_slow

            # ====== 4. Action target ======
            gt_actions = torch.tensor(np.array([ex["action"] for ex in examples]),
                                      device=device).float()
            if gt_actions.dim() == 3 and gt_actions.shape[1] > 1:
                targets = gt_actions
            else:
                # dataloader 给单步动作时退化处理
                targets = gt_actions.reshape(gt_actions.shape[0], -1, gt_actions.shape[-1])

            # ====== 5. Action losses ======
            trainer_cfg = getattr(self.config, "trainer", None)
            if trainer_cfg is None:
                rep_steps = 4
            elif isinstance(trainer_cfg, dict):
                rep_steps = trainer_cfg.get("repeated_diffusion_steps", 4)
            else:
                rep_steps = getattr(trainer_cfg, "repeated_diffusion_steps", 4)

            targets_rep = targets.repeat(rep_steps, 1, 1)
            state_rep   = state.repeat(rep_steps, 1, 1) if state is not None else None
            h_final_rep = h_final.repeat(rep_steps, 1, 1)
            h_fast_rep  = h_fast.repeat(rep_steps, 1, 1)

            # 主 loss：fused 通路
            loss_main = self.action_header(h_final_rep, targets_rep, state_rep)

            # 辅助 loss：fast 通路独立监督
            #   作用 1: 让 fast path 自己也是个能用的 policy（base 不被混合训练污染）
            #   作用 2: 给 gate 提供干净的难度信号
            loss_fast = self.action_header(h_fast_rep, targets_rep, state_rep)

            # ====== 6. Gate 监督：来自独立训练的 fast path 的 per-sample loss ======
            #   注意：用 reduction='none' 一次前向算完，不再串行 for 循环
            with torch.no_grad():
                per_sample_diff = self._per_sample_loss(h_fast, targets, state)  # [B]
                # 用 median 做中心，比 mean 更鲁棒；除以 std 做归一化
                med = per_sample_diff.median()
                std = per_sample_diff.std() + 1e-8
                g_target = torch.sigmoid((per_sample_diff - med) / std)

            loss_gate = F.mse_loss(g.squeeze(-1), g_target)

            # ====== 7. Total ======
            total_loss = (
                loss_main
                + self.fast_loss_weight * loss_fast
                + self.gate_loss_weight * loss_gate
            )

            if self.training and self.global_step % 50 == 0:
                with torch.no_grad():
                    print(
                        f"[step {self.global_step.item()}] "
                        f"main={loss_main.item():.4f} fast={loss_fast.item():.4f} "
                        f"gate={loss_gate.item():.4f} | "
                        f"g_pred=[{g.min().item():.2f},{g.mean().item():.2f},{g.max().item():.2f}] "
                        f"g_tgt=[{g_target.min().item():.2f},{g_target.mean().item():.2f},{g_target.max().item():.2f}]"
                    )

        return {
            "action_loss": total_loss,
            "loss_main": loss_main.detach(),
            "loss_fast": loss_fast.detach(),
            "loss_gate": loss_gate.detach(),
            "gate_mean": g.mean().detach(),
        }

    # ------------------------------------------------------------
    # Per-sample diffusion loss for difficulty signal
    # ------------------------------------------------------------
    def _per_sample_loss(self, h, targets, state):
        """
        一次前向算 per-sample loss，取代之前 B 次串行的 for 循环。
        注意：依赖 action_header 支持 reduction 参数。如果你的 FlowmatchingActionHead
        默认 reduction='mean'，需要在它的 forward 里加一个分支：
            if reduction == 'none': return per_sample_loss
        如果不想改 header，把 batch loss 当 difficulty 也行（损失一些精度）。
        """
        B = h.shape[0]
        try:
            losses = self.action_header(h, targets, state, reduction="none")  # [B]
            return losses.detach()
        except TypeError:
            # 兜底：header 不支持 reduction='none'。退化为整 batch 复用同一个难度。
            # 这种情况建议改下 header，否则 gate 学不到 per-sample 信号。
            mean_loss = self.action_header(h, targets, state).detach()
            return mean_loss.expand(B)

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------
    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        imgs = self._prep_images(examples)
        device = next(self.parameters()).device
        target_dtype = next(self.action_header.parameters()).dtype
        state = self._prep_state(examples, device, dtype=target_dtype)

        raw_langs = [ex["lang"] for ex in examples]
        prompts = [l + self.action_tokens_str for l in raw_langs]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs = self.vlm_instance.build_qwenvl_inputs(imgs, prompts)
            out = self.vlm_instance(**inputs, output_hidden_states=True)
            hidden_last = out.hidden_states[-1]
            scene_feat = hidden_last[:, :-self.num_action_tokens].mean(dim=1).float()
            h_base = self._extract_action_tokens(hidden_last, inputs["input_ids"])

        with torch.autocast("cuda", dtype=torch.float32):
            h_base = h_base.float()
            g = self.gate(scene_feat, state.squeeze(1) if state is not None else None)

            # 推理也用软路由（和训练一致），避免 train/test mismatch
            h_fast = self.adapter(h_base, num_iters=1)
            h_slow = self.adapter(h_base, num_iters=self.slow_iters)
            h_final = (1.0 - g.unsqueeze(-1)) * h_fast + g.unsqueeze(-1) * h_slow
            h_final = h_final.to(target_dtype)

        with torch.autocast("cuda", dtype=target_dtype):
            pred = self.action_header.predict_action(h_final, state)

        if kwargs.get("debug", False):  # 默认关掉，不再刷屏
            print(f"[Inference] gate={g.mean().item():.3f} act_mag={pred.abs().mean().item():.4f}")

        return {"normalized_actions": pred.detach().cpu().float().numpy()}
