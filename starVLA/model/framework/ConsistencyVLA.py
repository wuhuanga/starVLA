# Copyright 2026 starVLA community.
# Architecture: Latent Convergent Dynamics via Global Consistency
# Self-distillation VLA with early-exit capability.

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
# 1. 认知阶段编码
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
# 2. 空间特征无损压缩
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
# 3. 隐空间演化引擎
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
# 4. 主框架
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
        self.ema_decay_init = 0.9
        self.ema_decay_target = 0.999
        self.ema_decay_rampup_steps = 5000

        # Consistency loss mode: "action" (推荐) 或 "hidden"
        self.cd_loss_space = getattr(config.framework, "cd_loss_space", "action")

        self.context_bottleneck = PerceiverBottleneck(self.hidden_dim, num_latents=256)

        self.student_refiner = ContinuousRefiner(self.hidden_dim)
        self.teacher_refiner = copy.deepcopy(self.student_refiner)
        self.teacher_refiner.requires_grad_(False)
        self.teacher_refiner.eval()

        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.chunk_len * self.action_dim)
        )

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
        h0_detached = h0_raw.detach()
        context_reduced = self.context_bottleneck(raw_context)

        actions_t = torch.tensor(actions, device=device, dtype=torch.float32)
        actions_target = actions_t[:, -(self.future_window + 1):, :].reshape(actions_t.shape[0], -1)

        # ==========================================
        # B. 单次 student rollout，同时用于 action loss 和 cd loss
        # ==========================================
        t_steps = torch.randint(1, self.max_steps, (1,)).item()  # [1, max_steps-1]
        h_student = self._rollout(self.student_refiner, h0_detached, context_reduced,
                                  start_step=1, steps=t_steps)

        pred_student = self.action_head(h_student.mean(dim=1))
        action_loss = F.smooth_l1_loss(pred_student, actions_target)

        # ==========================================
        # C. Teacher 满步 rollout → consistency loss
        # ==========================================
        with torch.no_grad():
            h_teacher = self._rollout(self.teacher_refiner, h0_detached, context_reduced,
                                      start_step=1, steps=self.max_steps)

        if self.cd_loss_space == "action":
            with torch.no_grad():
                pred_teacher = self.action_head(h_teacher.mean(dim=1)).detach()
            cd_loss = F.mse_loss(pred_student, pred_teacher)
        else:
            cd_loss = F.mse_loss(
                F.normalize(h_student, p=2, dim=-1),
                F.normalize(h_teacher, p=2, dim=-1).detach()
            )

        # ==========================================
        # D. 合并损失
        # ==========================================
        cd_weight = 0.1
        total_loss = action_loss + cd_weight * cd_loss

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
            raw_context = outputs.hidden_states[-1]

            h0 = self._extract_h0_from_vlm(raw_context, qwen_inputs["input_ids"])
            context_reduced = self.context_bottleneck(raw_context)

            inference_steps = kwargs.get("inference_steps", self.max_steps)
            h_final = self._rollout(self.teacher_refiner, h0, context_reduced,
                                    start_step=1, steps=inference_steps)

            pred_actions = self.action_head(h_final.mean(dim=1))

        pred_actions = pred_actions.float().reshape(len(examples), self.chunk_len, self.action_dim)
        return {"normalized_actions": pred_actions.cpu().numpy()}

    def _get_ema_decay(self) -> float:
        """EMA decay with linear rampup: 初期快速跟上 student，后期稳定。"""
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
    print(f"Total Loss: {out['loss'].item():.4f} | Action: {out['action_loss'].item():.4f} | CD: {out['cd_loss'].item():.4f}")

    model.update_ema()

    pred_4step = model.predict_action([sample], inference_steps=4)
    pred_1step = model.predict_action([sample], inference_steps=1)
    print(f"4-step shape: {pred_4step['normalized_actions'].shape}")
    print(f"1-step shape: {pred_1step['normalized_actions'].shape}")
    print("ConsistencyVLA ready.")
