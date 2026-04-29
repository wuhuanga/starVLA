# Copyright 2026 starVLA community.
# Architecture: Latent Convergent Dynamics via Action-Space Consistency
# Features: Temporal Action Decoder (ACT-style), Direct Head Supervision, Dynamic CD Schedule.

import math
import copy
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = initialize_overwatch(__name__)


# ==============================================================================
# 1. 认知阶段编码 (离散 Step)
# ==============================================================================
class SinusoidalStepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=step.device, dtype=torch.float32) * -emb)
        emb = step.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ==============================================================================
# 2. 空间特征无损压缩 (提取 256 个高清 3D 几何特征)
# ==============================================================================
class PerceiverBottleneck(nn.Module):
    def __init__(self, dim: int, num_latents: int = 256):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, raw_context: torch.Tensor) -> torch.Tensor:
        B = raw_context.shape[0]
        q = self.latent_queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=q, key=raw_context, value=raw_context, need_weights=False)
        h = self.norm1(q + attn_out)
        return self.norm2(h + self.ffn(h))


# ==============================================================================
# 3. 🚀 ACT 风格时序动作解码器 (Temporal Action Decoder)
# ==============================================================================
class TemporalActionDecoder(nn.Module):
    def __init__(self, hidden_dim: int, chunk_len: int, action_dim: int):
        super().__init__()
        self.chunk_len = chunk_len
        # 纯粹的时间查询通证
        self.time_embeddings = nn.Parameter(torch.randn(1, chunk_len, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.action_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, h_intent: torch.Tensor, spatial_context: torch.Tensor) -> torch.Tensor:
        B = spatial_context.shape[0]
        q = self.time_embeddings.expand(B, -1, -1)

        # h_intent: [B, K, dim] (K=num_latent_query, 可能>1)
        # spatial_context: [B, 256, dim]
        # 池化为 [B, 1, dim] 以广播注入到所有 spatial tokens
        h_pooled = h_intent.mean(dim=1, keepdim=True)  # [B, 1, dim]
        kv = spatial_context + h_pooled

        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        out = self.norm(q + attn_out)
        return self.action_proj(out)

# ==============================================================================
# 4. 隐空间演化引擎
# ==============================================================================
class ContinuousRefiner(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.step_mlp = nn.Sequential(
            SinusoidalStepEmbedding(dim),
            nn.Linear(dim, dim)
        )
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, context: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        time_emb = self.step_mlp(step).unsqueeze(1)
        h = h + time_emb
        update, _ = self.attn(query=h, key=context, value=context, need_weights=False)
        h = self.norm1(h + update)
        return self.norm2(h + self.mlp(h))


# ==============================================================================
# 5. 主框架
# ==============================================================================
@FRAMEWORK_REGISTRY.register("ConsistencyVLA")
class ConsistencyVLA(baseframework):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.hidden_dim = self.qwen_vl_interface.model.config.hidden_size

        try:
            am = config.framework.action_model
            self.action_dim = getattr(am, "action_dim", 7)
            self.future_window = getattr(am, "future_action_window_size", 10)
            self.past_window = getattr(am, "past_action_window_size", 0)
            self.chunk_len = self.past_window + 1 + self.future_window
        except AttributeError:
            self.action_dim = 7
            self.future_window = 10
            self.past_window = 0
            self.chunk_len = 11

        self.num_latent_query = getattr(self.config.framework.qwenvl, "num_latent_action_query", 1)
        self.latent_action_query = "".join([f"<|action_{i}|>" for i in range(self.num_latent_query)])
        self.action_token_ids = None

        self.max_steps = 4
        # 🚀 调慢 EMA 衰减，让 Teacher 成为极度稳定的定海神针
        self.ema_decay_init = 0.95
        self.ema_decay_target = 0.999
        self.ema_decay_rampup_steps = 5000

        self.cd_loss_space = getattr(config.framework, "cd_loss_space", "action")

        self.context_bottleneck = PerceiverBottleneck(self.hidden_dim, num_latents=256)

        self.student_refiner = ContinuousRefiner(self.hidden_dim)
        self.teacher_refiner = copy.deepcopy(self.student_refiner)
        self.teacher_refiner.requires_grad_(False)
        self.teacher_refiner.eval()

        # 🚀 替换为主力动作头 (处理 Refiner 输出)
        self.action_head = TemporalActionDecoder(self.hidden_dim, self.chunk_len, self.action_dim)
        
        

        self._ema_step = 0

    def _ensure_action_token_ids(self, tokenizer):
        if self.action_token_ids is None:
            self.action_token_ids = {"first": tokenizer.convert_tokens_to_ids("<|action_0|>")}

    def _extract_h0_from_vlm(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        self._ensure_action_token_ids(tokenizer)
        first_id = self.action_token_ids["first"]

        B = hidden_states.shape[0]
        out = []
        for b in range(B):
            pos = (input_ids[b] == first_id).nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                out.append(hidden_states[b, -self.num_latent_query:, :])
            else:
                start = pos[0].item()
                out.append(hidden_states[b, start:start + self.num_latent_query, :])
        return torch.stack(out, dim=0)

    def _rollout(self, refiner: nn.Module, h0: torch.Tensor, context: torch.Tensor,
                 start_step: int, steps: int) -> torch.Tensor:
        h = h0
        for i in range(steps):
            current_step = min(start_step + i, self.max_steps)
            step_tensor = torch.full((h.size(0),), current_step, dtype=torch.long, device=h.device)
            h = refiner(h, context, step_tensor)
        return h

    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        device = self.qwen_vl_interface.model.device

        batch_images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] + self.latent_action_query for ex in examples]
        actions = np.array([ex["action"] for ex in examples])

        # ==========================================
        # A. VLM 前向与 Perceiver 压缩
        # ==========================================
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(batch_images, instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs, output_hidden_states=True, return_dict=True, use_cache=False
            )
            raw_context = outputs.hidden_states[-1].to(torch.float32)

        h0_raw = self._extract_h0_from_vlm(raw_context, qwen_inputs["input_ids"])
        h0_detached = h0_raw.detach()  # [B, 1, dim]
        context_reduced = self.context_bottleneck(raw_context)  # [B, 256, dim]

        actions_t = torch.tensor(actions, device=device, dtype=torch.float32)
        # Target shape: [B, chunk_len, action_dim]
        actions_target = actions_t[:, -(self.future_window + 1):, :] 

        # ==========================================
        # C. 主力 Action loss (满步 rollout)
        # ==========================================
        h_student_full = self._rollout(self.student_refiner, h0_detached, context_reduced,
                                       start_step=1, steps=self.max_steps)
        pred_student_full = self.action_head(h_student_full, context_reduced)
        action_loss = F.smooth_l1_loss(pred_student_full, actions_target)

        # ==========================================
        # D. 相邻步一致性 (Adjacent-step Consistency)
        # ==========================================
        t_steps = torch.randint(1, self.max_steps, (1,)).item()
        
        h_student_t = self._rollout(self.student_refiner, h0_detached, context_reduced,
                                    start_step=1, steps=t_steps)

        with torch.no_grad():
            h_teacher_t1 = self._rollout(self.teacher_refiner, h0_detached, context_reduced,
                                         start_step=1, steps=t_steps + 1)

        # 在动作空间拉齐
        pred_student_t = self.action_head(h_student_t, context_reduced)
        with torch.no_grad():
            pred_teacher_t1 = self.action_head(h_teacher_t1, context_reduced).detach()
            
        cd_loss = F.mse_loss(pred_student_t, pred_teacher_t1)

        # ==========================================
        # E. 合并损失 (引入动态 CD 权重调度)
        # ==========================================
        # 🚀 前期专注拟合动作，后期慢慢加上一致性约束 (目标权重 0.1)
        rampup_ratio = min(self._ema_step / max(self.ema_decay_rampup_steps, 1), 1.0)
        cd_weight = 0.1 * rampup_ratio

        # 辅助直连 loss 权重给 0.5
        total_loss = action_loss  + (cd_weight * cd_loss)

        return {
            "loss": total_loss,
            "action_loss": action_loss,
            "cd_loss": cd_loss,
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        batch_images = [
            [to_pil_preserve(im) for im in (ex["image"] if isinstance(ex["image"], list) else [ex["image"]])]
            for ex in examples
        ]
        instructions = [ex["lang"] + self.latent_action_query for ex in examples]

        target_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if target_size:
            batch_images = resize_images(batch_images, target_size=target_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(batch_images, instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs, output_hidden_states=True, return_dict=True, use_cache=False
            )
            raw_context = outputs.hidden_states[-1].to(torch.float32)

            h0 = self._extract_h0_from_vlm(raw_context, qwen_inputs["input_ids"])
            context_reduced = self.context_bottleneck(raw_context)

            # 动态支持 Early-Exit
            inference_steps = kwargs.get("inference_steps", self.max_steps)
            h_final = self._rollout(self.teacher_refiner, h0, context_reduced,
                                    start_step=1, steps=inference_steps)

            # 直接解码出 [B, chunk_len, action_dim]
            pred_actions = self.action_head(h_final, context_reduced)

        return {"normalized_actions": pred_actions.float().cpu().numpy()}

    def _get_ema_decay(self) -> float:
        ratio = min(self._ema_step / max(self.ema_decay_rampup_steps, 1), 1.0)
        return self.ema_decay_init + (self.ema_decay_target - self.ema_decay_init) * ratio

    @torch.no_grad()
    def update_ema(self):
        decay = self._get_ema_decay()
        for param_s, param_t in zip(self.student_refiner.parameters(), self.teacher_refiner.parameters()):
            param_t.data.mul_(decay).add_((1.0 - decay) * param_s.data)
        self._ema_step += 1


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str,
                        default="./examples/MultiRobot/train_files/starvla_cotrain_multiRobot.yaml")
    args, _ = parser.parse_known_args()

    try:
        cfg = OmegaConf.load(args.config_yaml)
    except Exception:
        from types import SimpleNamespace
        cfg = PretrainedConfig()
        cfg.framework = SimpleNamespace(
            action_model=SimpleNamespace(action_dim=7, future_action_window_size=10, past_action_window_size=0),
            qwenvl=SimpleNamespace(num_latent_action_query=1),
            cd_loss_space="action",
        )
        cfg.model_type = "qwen2_vl"

    model = ConsistencyVLA(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(20, 7)).astype(np.float32),
        "image": [image],
        "lang": "Carefully pick up the block and place it on the table.",
    }

    out = model([sample, sample])
    print(f"Total Loss: {out['loss'].item():.4f} | Action: {out['action_loss'].item():.4f} | Direct: {out['direct_loss'].item():.4f} | CD: {out['cd_loss'].item():.4f}")

    model.update_ema()

    pred_4step = model.predict_action([sample], inference_steps=4)
    pred_1step = model.predict_action([sample], inference_steps=1)
    print(f"4-step shape: {pred_4step['normalized_actions'].shape}")
    print(f"1-step shape: {pred_1step['normalized_actions'].shape}")
    print("Pareto-Optimal Release is ready. Let's make history.")