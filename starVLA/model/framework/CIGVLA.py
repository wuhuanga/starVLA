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

ADVANCED_TASK_TEMPLATE = "Task: {}. Think step by step about geometry, object pose, and manipulation constraints before producing the action."

# ==========================================
# 1. 潜在推理适配器 (Latent Reasoning Adapter)
# ==========================================
class LatentReasoningAdapter(nn.Module):
    def __init__(self, dim, num_tokens=16, heads=8):
        super().__init__()
        self.reason_tokens = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # ====== 🌟 核心救命代码：零初始化 ======
        # 强制让 Adapter 初始阶段的残差增量为 0，防止破坏原生特征
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h_base):
        B = h_base.shape[0]
        r = self.reason_tokens.expand(B, -1, -1)
        r2, _ = self.cross_attn(query=r, key=h_base, value=h_base)
        r = self.norm1(r + r2)
        update, _ = self.cross_attn(query=h_base, key=r, value=r)
        h_mid = self.norm2(h_base + update)
        delta = self.mlp(h_mid)
        return h_base + delta

# ==========================================
# 2. 极简版状态门控 (Simple State-Aware Router)
# 直接输出 0~1 的权重，用于融合快慢特征
# ==========================================
class InformationGainTrigger(nn.Module):
    def __init__(self, scene_dim: int, state_dim: int = 8):
        super().__init__()
        self.state_dim = state_dim
        self.use_state = state_dim > 0
        
        trigger_in_dim = scene_dim
        if self.use_state:
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Linear(32, 16)
            )
            trigger_in_dim += 16

        self.net = nn.Sequential(
            nn.Linear(trigger_in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, scene_feat: torch.Tensor, state: Optional[torch.Tensor] = None):
        if self.use_state:
            if state is None:
                state = torch.zeros((scene_feat.shape[0], self.state_dim), device=scene_feat.device, dtype=scene_feat.dtype)
            current_state = state[:, -1, :] if state.dim() == 3 else state
            s_feat = self.state_mlp(current_state.to(scene_feat.dtype))
            trigger_input = torch.cat([scene_feat, s_feat], dim=-1)
        else:
            trigger_input = scene_feat

        logits = self.net(trigger_input)
        if self.training:
            noise = torch.randn_like(logits) * 0.5
            logits = logits + noise
        return torch.sigmoid(logits)


@FRAMEWORK_REGISTRY.register("CIGVLA")
class CIGVLA(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.vlm_instance = get_vlm_model(config=self.config)
        
        print("[CIG-VLA] 🚀 Minimalist End-to-End Mode Enabled (Single VLM Forward)!")
        
        # ====== 动态注册 Action Tokens ======
        self.num_action_tokens = getattr(self.config.framework, "qwenvl", {}).get("num_latent_action_query", 64) if isinstance(getattr(self.config.framework, "qwenvl", {}), dict) else getattr(self.config.framework.qwenvl, "num_latent_action_query", 64)
        self.action_tokens_list = [f"<|action_{i}|>" for i in range(self.num_action_tokens)]
        self.action_tokens_str = "".join(self.action_tokens_list)
        
        tokenizer = self.vlm_instance.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.convert_tokens_to_ids(self.action_tokens_list[0]) is None:
            num_added = tokenizer.add_tokens(self.action_tokens_list, special_tokens=True)
            if num_added > 0:
                self.vlm_instance.model.resize_token_embeddings(len(tokenizer))
                input_embeds = self.vlm_instance.model.get_input_embeddings().weight.data
                with torch.no_grad():
                    input_embeds[-num_added:] = input_embeds[:-num_added].mean(dim=0, keepdim=True)

        vlm_config = self.vlm_instance.model.config
        vlm_hidden_size = getattr(vlm_config, "hidden_size", 2048)
        if hasattr(vlm_config, "text_config"):
            vlm_hidden_size = getattr(vlm_config.text_config, "hidden_size", vlm_hidden_size)

        self.state_dim = 8
        target_action_dim = 2048
        
        action_cfg = getattr(self.config.framework, "action_model", {})
        if isinstance(action_cfg, dict):
            self.state_dim = action_cfg.get("state_dim", 8)
            diff_cfg = action_cfg.get("diffusion_model_cfg", {})
            target_action_dim = diff_cfg.get("cross_attention_dim", 2048) if isinstance(diff_cfg, dict) else getattr(diff_cfg, "cross_attention_dim", 2048)
        else:
            self.state_dim = getattr(action_cfg, "state_dim", 8)
            diff_cfg = getattr(action_cfg, "diffusion_model_cfg", {})
            target_action_dim = diff_cfg.get("cross_attention_dim", 2048) if isinstance(diff_cfg, dict) else getattr(diff_cfg, "cross_attention_dim", 2048)

        # ====== 核心模块 ======
        self.cognition_adapter = LatentReasoningAdapter(target_action_dim, num_tokens=16).to(torch.float32)
        self.trigger = InformationGainTrigger(scene_dim=vlm_hidden_size, state_dim=self.state_dim).to(torch.float32)
        self.action_header = get_action_model(config=self.config)
        
        if vlm_hidden_size != target_action_dim:
            self.vlm_to_action_proj = nn.Linear(vlm_hidden_size, target_action_dim)
        else:
            self.vlm_to_action_proj = nn.Identity()

        self.future_window = action_cfg.get("future_action_window_size", 7) if isinstance(action_cfg, dict) else getattr(action_cfg, "future_action_window_size", 7)
        self.entropy_weight = 0.01  # 熵正则：鼓励 g 远离 0/1，保持探索
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

    def _extract_action_tokens(self, hidden, input_ids):
        # ... (保持原样，这部分提取逻辑没问题)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.as_tensor(input_ids, device=hidden.device, dtype=torch.long)
        B, L, D = hidden.shape
        tokenizer = self.vlm_instance.processor.tokenizer
        start_id = tokenizer.convert_tokens_to_ids("<|action_0|>")
        mask = (input_ids == start_id)
        matched_indices = mask.nonzero(as_tuple=True)
        
        if len(matched_indices[0]) == 0:
            return torch.zeros((B, self.num_action_tokens, D), device=hidden.device, dtype=hidden.dtype)
            
        start_indices = torch.zeros(B, device=hidden.device, dtype=torch.long)
        for b in range(B):
            sample_indices = matched_indices[1][matched_indices[0] == b]
            start_indices[b] = sample_indices[-1] if len(sample_indices) > 0 else L - self.num_action_tokens
            
        offsets = torch.arange(self.num_action_tokens, device=hidden.device).unsqueeze(0)
        indices = torch.clamp(start_indices.unsqueeze(1) + offsets, max=L-1)
        indices_ext = indices.unsqueeze(-1).expand(-1, -1, D)
        action_tokens = torch.gather(hidden, 1, indices_ext)
        
        proj_dtype = self.vlm_to_action_proj.weight.dtype if isinstance(self.vlm_to_action_proj, nn.Linear) else action_tokens.dtype
        return self.vlm_to_action_proj(action_tokens.to(proj_dtype)).to(action_tokens.dtype)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        if self.training:
            self.global_step += 1
            
        imgs = []
        for ex in examples:
            raw_img = ex["image"]
            sample_imgs = []
            if not isinstance(raw_img, (list, tuple)):
                raw_img = [raw_img]
            for x in raw_img:
                if isinstance(x, np.ndarray):
                    if x.ndim == 3 and x.shape[0] in [1, 3, 4]: 
                        x = np.transpose(x, (1, 2, 0))
                    sample_imgs.append(Image.fromarray(x.astype('uint8')))
                else:
                    sample_imgs.append(x)
            imgs.append(sample_imgs)
                
        raw_langs = [ex["lang"] for ex in examples]
        advanced_langs = [ADVANCED_TASK_TEMPLATE.format(l) for l in raw_langs]
        target_device = next(self.parameters()).device
        
        # ====== 1. 严格防御的 State 维度对齐 [B, 1, Dim] ======
        state_list = [ex.get("state", None) for ex in examples]
        if state_list[0] is not None:
            state_tensor = torch.tensor(np.array(state_list), device=target_device).float()
            if state_tensor.dim() == 2:
                safe_state = state_tensor.unsqueeze(1)
            elif state_tensor.dim() == 3:
                safe_state = state_tensor[:, -1:, :] # 防御：只取序列最后一步
            else:
                safe_state = state_tensor
        else:
            safe_state = None

        # ====== 2. 单次前向 VLM ======
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs_unified = self.vlm_instance.build_qwenvl_inputs(imgs, [l + self.action_tokens_str for l in advanced_langs])
            out_unified = self.vlm_instance(**inputs_unified, output_hidden_states=True)
            
            global_scene_feat = out_unified.hidden_states[-1][:, :-self.num_action_tokens].mean(dim=1).float()
            h_base_proj = self._extract_action_tokens(out_unified.hidden_states[-1], inputs_unified["input_ids"])

        with torch.autocast("cuda", dtype=torch.float32):
            h_base_f32 = h_base_proj.float() 
            
            # ====== 3. 认知流形与端到端特征融合 ======
            h_cog_f32 = self.cognition_adapter(h_base_f32)
            g_pred = self.trigger(global_scene_feat.detach(), safe_state.squeeze(1) if safe_state is not None else None)
            
            # 软路由：直接按照权重混合特征，让网络端到端决定用谁！
            h_fused = (1.0 - g_pred.unsqueeze(-1)) * h_base_f32 + g_pred.unsqueeze(-1) * h_cog_f32

            # ====== 4. 严格防御的 Action Target 切片 ======
            gt_actions = torch.tensor(np.array([ex["action"] for ex in examples]), device=h_base_f32.device).float()
            
            if gt_actions.dim() == 3 and gt_actions.shape[1] > 1:
                # Dataloader 已经准备好了 Chunk，直接用，绝不乱切！
                targets = gt_actions
            else:
                # 只有单步动作时，才截取未来窗口
                targets = gt_actions[:, -(self.future_window + 1):, :] 
                
            repeated_steps = getattr(self.config.trainer, "repeated_diffusion_steps", 4) if hasattr(self.config, "trainer") and not isinstance(self.config.trainer, dict) else (self.config.trainer.get("repeated_diffusion_steps", 4) if hasattr(self.config, "trainer") else 4)
            
            targets_repeated = targets.repeat(repeated_steps, 1, 1)
            h_fused_rep = h_fused.repeat(repeated_steps, 1, 1)
            state_rep = safe_state.repeat(repeated_steps, 1, 1) if safe_state is not None else None
            
            # ====== 5. 极简的主 Loss 计算 ======
            # 直接计算融合特征的 Diffusion Loss，告别方差爆炸
            B = h_fused.shape[0]
            losses = []
            for i in range(B):
                idx = [i + j * B for j in range(repeated_steps)]
                loss_i = self.action_header(h_fused_rep[idx], targets_repeated[idx], state_rep[idx] if state_rep is not None else None)
                losses.append(loss_i)
            action_loss = torch.stack(losses).mean()

            # 熵正则：-[g*log(g) + (1-g)*log(1-g)]，鼓励 g 停留在 0.5 附近而非坍塌到 0 或 1
            g_clamped = g_pred.clamp(1e-6, 1 - 1e-6)
            gate_entropy = -(g_clamped * g_clamped.log() + (1 - g_clamped) * (1 - g_clamped).log()).mean()
            total_loss = action_loss - self.entropy_weight * gate_entropy

            if self.training and self.global_step % 50 == 0:
                print(f"Step {self.global_step.item()} | Gate Mean: {g_pred.mean().item():.3f} | Action Loss: {action_loss.item():.4f}")

        return {
            "action_loss": total_loss,
            "gate_mean": g_pred.mean().detach()
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        # ====== 推理阶段彻底砍掉 Temporal Ensembling，纯净输出 ======
        imgs = []
        for ex in examples:
            raw_img = ex["image"]
            sample_imgs = []
            if not isinstance(raw_img, (list, tuple)):
                raw_img = [raw_img]
            for x in raw_img:
                if isinstance(x, np.ndarray):
                    if x.ndim == 3 and x.shape[0] in [1, 3, 4]: 
                        x = np.transpose(x, (1, 2, 0))
                    sample_imgs.append(Image.fromarray(x.astype('uint8')))
                else:
                    sample_imgs.append(x)
            imgs.append(sample_imgs)
                
        target_device = next(self.parameters()).device
        target_dtype = next(self.action_header.parameters()).dtype
        
        # 严格维度对齐
        state_list = [ex.get("state", None) for ex in examples]
        if state_list[0] is not None:
            state_tensor = torch.tensor(np.array(state_list), device=target_device, dtype=target_dtype)
            if state_tensor.dim() == 2:
                safe_state = state_tensor.unsqueeze(1)
            elif state_tensor.dim() == 3:
                safe_state = state_tensor[:, -1:, :]
            else:
                safe_state = state_tensor
        else:
            safe_state = None

        raw_langs = [ex["lang"] for ex in examples]
        advanced_langs = [ADVANCED_TASK_TEMPLATE.format(l) for l in raw_langs]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs_unified = self.vlm_instance.build_qwenvl_inputs(imgs, [l + self.action_tokens_str for l in advanced_langs])
            out_unified = self.vlm_instance(**inputs_unified, output_hidden_states=True)
            
            global_scene_feat = out_unified.hidden_states[-1][:, :-self.num_action_tokens].mean(dim=1).float()
            h_base_proj = self._extract_action_tokens(out_unified.hidden_states[-1], inputs_unified["input_ids"])

        with torch.autocast("cuda", dtype=torch.float32):
            g_pred = self.trigger(global_scene_feat, safe_state.squeeze(1) if safe_state is not None else None)
            h_cog_proj = self.cognition_adapter(h_base_proj.float())
            h_fused = (1.0 - g_pred.unsqueeze(-1)) * h_base_proj.float() + g_pred.unsqueeze(-1) * h_cog_proj
            h_final_aligned = h_fused.to(target_dtype)

        with torch.autocast("cuda", dtype=target_dtype):
            pred = self.action_header.predict_action(h_final_aligned, safe_state)

        if kwargs.get('debug', True):
            act_mag = pred.abs().mean().item()
            print(f"[Inference] Gate Score: {g_pred.mean().item():.3f} | Action Mag: {act_mag:.4f}")

        # 干干净净，拿到就走！
        return {"normalized_actions": pred.detach().cpu().float().numpy()}