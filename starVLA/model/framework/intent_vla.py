# Copyright 2026 starVLA community.
# IntentVLA: Intent-level Self-Distillation for Vision-Language-Action Models
#
# Core idea:
#   Teacher sees clean (image, text). Student sees augmented (image, text).
#   Both produce a "latent intent" h at the action-query positions.
#   Student is forced to align its intent with the EMA teacher's via cosine distillation,
#   while both paths are supervised by the action loss through a shared flow-matching head.
#
# Design decisions (see accompanying design doc):
#   (1) Teacher = EMA(Student), not a separate network. EMA stabilizes the alignment target.
#   (2) Distillation loss = 1 - cosine_similarity (BYOL-style), InfoNCE available for ablation.
#   (3) Augmentation strength: medium visual, offline paraphrase bank for language.
#   (4) Intent extracted at action-query positions of the last VLM hidden layer.
#   (5) Student action loss (main) + Teacher action loss (auxiliary, grad through shared head only).
#   (6) VLM backbone is trainable (expected via LoRA externally). No detach on h0.

import copy
import random
from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = initialize_overwatch(__name__)


# ==============================================================================
# Visual augmentation: medium strength, robotics-safe
# ==============================================================================
class RoboSafeAugment:
    """
    Medium-strength visual augmentation tuned for manipulation data.
    - No horizontal flip (left/right has semantic meaning for arms).
    - No rotation (gravity direction matters).
    - Conservative crop scale to avoid cutting key objects.
    """
    def __init__(self, p_apply: float = 1.0):
        self.p_apply = p_apply
        self.jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        # scale (0.8, 1.0) is deliberately conservative vs CV's (0.5, 1.0)
        self.resized_crop = T.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1))
        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()

    def __call__(self, pil_image: Image.Image) -> Image.Image:
        if random.random() > self.p_apply:
            return pil_image
        img = self.jitter(pil_image)
        img = self.resized_crop(img)
        # Gaussian noise in tensor space, then back to PIL
        t = self.to_tensor(img)
        t = t + torch.randn_like(t) * 0.02
        t = t.clamp(0.0, 1.0)
        return self.to_pil(t)


# ==============================================================================
# Language paraphrase bank (offline, populated externally via GPT-4 and curated)
# ==============================================================================
class ParaphraseBank:
    """
    Maps original instruction -> list of human-verified paraphrases.
    Populated offline. If an instruction is not in the bank, return the original.
    """
    def __init__(self, bank: Optional[Dict[str, List[str]]] = None):
        self.bank = bank or {}

    def sample(self, instruction: str) -> str:
        candidates = self.bank.get(instruction.strip(), None)
        if not candidates:
            return instruction
        return random.choice(candidates)


# ==============================================================================
# Main framework
# ==============================================================================
@FRAMEWORK_REGISTRY.register("IntentVLA")
class IntentVLA(baseframework):
    """
    Intent-level self-distillation VLA.

    Training forward does two VLM passes:
      - Teacher pass (EMA VLM, no grad): clean image + original text
      - Student pass (trainable VLM): augmented image + paraphrased text

    Losses:
      - L_action_student: primary action supervision from student intent
      - L_action_teacher: auxiliary action supervision from teacher intent
                         (grad flows only through shared action head)
      - L_distill: 1 - cos(h_student, sg(h_teacher)) at action-query positions
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config

        # ----- VLM (Student). This path stays trainable. -----
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.hidden_dim = self.qwen_vl_interface.model.config.hidden_size

        # align DiT cross-attn dim to VLM hidden dim
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.hidden_dim

        # ----- Latent action queries -----
        self.num_latent_action_query = int(
            self.config.framework.qwenvl.get("num_latent_action_query", 32)
        )
        self.latent_action_query = "".join(
            [f"<|action_{i}|>" for i in range(self.num_latent_action_query)]
        )
        self.action_token_ids = None  # filled lazily

        # ----- Flow-matching action head (shared by teacher and student paths) -----
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        am = config.framework.action_model
        self.future_action_window_size = am.future_action_window_size
        self.past_action_window_size = am.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # ----- Loss weights -----
        fw = self.config.framework
        self.w_action_student = float(fw.get("w_action_student", 1.0))
        self.w_action_teacher = float(fw.get("w_action_teacher", 0.3))
        self.w_distill = float(fw.get("w_distill", 0.5))
        self.distill_type = str(fw.get("distill_type", "cosine"))  # 'cosine' | 'mse' | 'infonce'
        self.infonce_temp = float(fw.get("infonce_temp", 0.1))

        # ----- EMA teacher (deep copy of student VLM; params only, no grad) -----
        # NOTE: this doubles VLM parameter memory. For very large backbones,
        # consider EMA over the LoRA deltas only; left as future work.
        teacher_vlm = copy.deepcopy(self.qwen_vl_interface)
        for p in teacher_vlm.parameters():
            p.requires_grad_(False)
        teacher_vlm.eval()
        # Keep EMA teacher out of nn.Module registration so optimizer/DeepSpeed
        # only sees student parameters.
        object.__setattr__(self, "_teacher_vlm", teacher_vlm)

        self.ema_decay_init = float(fw.get("ema_decay_init", 0.99))
        self.ema_decay_final = float(fw.get("ema_decay_final", 0.9995))
        self.ema_rampup_steps = int(fw.get("ema_rampup_steps", 5000))
        self._ema_step = 0

        # ----- Augmentation and paraphrase -----
        self.visual_aug = RoboSafeAugment(p_apply=float(fw.get("p_visual_aug", 1.0)))
        paraphrase_bank = fw.get("paraphrase_bank", None)
        self.paraphrase = ParaphraseBank(paraphrase_bank)
        self.p_paraphrase = float(fw.get("p_paraphrase", 0.5))

    @property
    def teacher_vlm(self):
        return self._teacher_vlm

    def _sync_teacher_device_dtype(self):
        """Move the unregistered EMA teacher alongside the student on demand."""
        student_model = self.qwen_vl_interface.model
        teacher_model = self.teacher_vlm.model
        student_param = next(student_model.parameters(), None)
        teacher_param = next(teacher_model.parameters(), None)
        if student_param is None or teacher_param is None:
            return
        if teacher_param.device != student_param.device or teacher_param.dtype != student_param.dtype:
            self.teacher_vlm.to(device=student_param.device, dtype=student_param.dtype)
            self.teacher_vlm.eval()

    # ------------------------------------------------------------------ helpers
    def _ensure_action_token_ids(self, tokenizer):
        if self.action_token_ids is None:
            self.action_token_ids = {
                "first": tokenizer.convert_tokens_to_ids("<|action_0|>"),
                "last": tokenizer.convert_tokens_to_ids(
                    f"<|action_{self.num_latent_action_query - 1}|>"
                ),
            }

    def _extract_intent(
        self,
        hidden_states: torch.Tensor,  # [B, S, H]
        input_ids: torch.Tensor,      # [B, S]
        tokenizer,
    ) -> torch.Tensor:
        """Extract the K-token latent action block from each sequence. Returns [B, K, H]."""
        self._ensure_action_token_ids(tokenizer)
        first_id = self.action_token_ids["first"]
        last_id = self.action_token_ids["last"]
        K = self.num_latent_action_query

        B = hidden_states.shape[0]
        out = []
        for b in range(B):
            pos = (input_ids[b] == int(first_id)).nonzero(as_tuple=True)[0]
            assert pos.numel() > 0, f"Sample {b}: no <|action_0|> found in input_ids."
            start = int(pos[0].item())
            end = start + K
            assert end <= input_ids.shape[1], f"Sample {b}: action block truncated."
            assert int(input_ids[b, end - 1].item()) == int(last_id), (
                f"Sample {b}: action block is not contiguous (expected <|action_{K-1}|> at end)."
            )
            out.append(hidden_states[b, start:end, :])
        return torch.stack(out, dim=0)  # [B, K, H]

    # ------------------------------------------------------ distillation losses
    def _distill_loss(self, h_s: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        h_s, h_t: [B, K, H]. h_t must be already detached.
        Computed per-token then averaged.
        """
        if self.distill_type == "cosine":
            # 1 - cos, averaged over K tokens and batch
            cos = F.cosine_similarity(h_s, h_t, dim=-1)  # [B, K]
            return (1.0 - cos).mean()
        elif self.distill_type == "mse":
            return F.mse_loss(h_s, h_t)
        elif self.distill_type == "infonce":
            # Treat each sample's mean-pooled intent as the query; other samples in batch are negatives.
            q = F.normalize(h_s.mean(dim=1), dim=-1)          # [B, H]
            k = F.normalize(h_t.mean(dim=1), dim=-1)          # [B, H]
            logits = q @ k.t() / self.infonce_temp             # [B, B]
            labels = torch.arange(q.size(0), device=q.device)
            return F.cross_entropy(logits, labels)
        else:
            raise ValueError(f"Unknown distill_type: {self.distill_type}")

    # -------------------------------------------------------------- augmenters
    def _augment_batch_images(self, batch_images: List[List[Image.Image]]) -> List[List[Image.Image]]:
        out = []
        for imgs in batch_images:
            out.append([self.visual_aug(im) for im in imgs])
        return out

    def _paraphrase_batch(self, instructions: List[str]) -> List[str]:
        out = []
        for ins in instructions:
            if random.random() < self.p_paraphrase:
                out.append(self.paraphrase.sample(ins))
            else:
                out.append(ins)
        return out

    # ------------------------------------------------------------------ forward
    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        tokenizer = self.qwen_vl_interface.processor.tokenizer

        # --- raw batch
        batch_images_raw = [
            [to_pil_preserve(im) for im in (ex["image"] if isinstance(ex["image"], list) else [ex["image"]])]
            for ex in examples
        ]
        instructions_raw = [ex["lang"] for ex in examples]
        actions = np.array([ex["action"] for ex in examples])
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        # --- teacher view: clean image + original text (+ action query appended)
        instructions_teacher = [ins + self.latent_action_query for ins in instructions_raw]

        # --- student view: augmented image + paraphrased text (+ action query appended)
        batch_images_student = self._augment_batch_images(batch_images_raw)
        instructions_student = [
            ins + self.latent_action_query
            for ins in self._paraphrase_batch(instructions_raw)
        ]

        # ============== Teacher forward (EMA VLM, no grad) ==============
        self._sync_teacher_device_dtype()
        with torch.no_grad():
            teacher_inputs = self.teacher_vlm.build_qwenvl_inputs(
                batch_images_raw, instructions_teacher
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_out = self.teacher_vlm(
                    **teacher_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                teacher_last = teacher_out.hidden_states[-1].to(torch.float32)
            h_teacher = self._extract_intent(
                teacher_last, teacher_inputs["input_ids"], tokenizer
            )  # [B, K, H]
        h_teacher_sg = h_teacher.detach()

        # ============== Student forward (trainable VLM) ==============
        student_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            batch_images_student, instructions_student
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_out = self.qwen_vl_interface(
                **student_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            student_last = student_out.hidden_states[-1].to(torch.float32)
        h_student = self._extract_intent(
            student_last, student_inputs["input_ids"], tokenizer
        )  # [B, K, H]

        # ============== Distillation loss ==============
        loss_distill = self._distill_loss(h_student, h_teacher_sg)

        # ============== Action losses (shared flow-matching head) ==============
        with torch.autocast("cuda", dtype=torch.float32):
            device = h_student.device
            actions_t = torch.tensor(actions, device=device, dtype=h_student.dtype)
            actions_target = actions_t[:, -(self.future_action_window_size + 1):, :]  # [B, chunk_len, action_dim]

            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(np.array(state), device=device, dtype=h_student.dtype)

            repeated_diff_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4)
                if self.config and self.config.trainer else 4
            )
            actions_target_rep = actions_target.repeat(repeated_diff_steps, 1, 1)
            state_rep = state_tensor.repeat(repeated_diff_steps, 1, 1) if state_tensor is not None else None

            h_student_rep = h_student.repeat(repeated_diff_steps, 1, 1).float()
            h_teacher_rep = h_teacher_sg.repeat(repeated_diff_steps, 1, 1).float()

            loss_act_student = self.action_model(h_student_rep, actions_target_rep, state_rep)
            # Teacher path: h is detached, so gradient flows only through shared action head params.
            loss_act_teacher = self.action_model(h_teacher_rep, actions_target_rep, state_rep)

        # ============== Total ==============
        total = (
            self.w_action_student * loss_act_student
            + self.w_action_teacher * loss_act_teacher
            + self.w_distill * loss_distill
        )

        return {
            "loss": total,
            "action_loss": loss_act_student.detach(),
            "action_loss_teacher": loss_act_teacher.detach(),
            "distill_loss": loss_distill.detach(),
        }

    # ------------------------------------------------------------------ EMA
    def _current_ema_decay(self) -> float:
        r = min(self._ema_step / max(self.ema_rampup_steps, 1), 1.0)
        return self.ema_decay_init + (self.ema_decay_final - self.ema_decay_init) * r

    @torch.no_grad()
    def update_ema(self):
        """Call once per optimizer step, AFTER student weights are updated."""
        self._sync_teacher_device_dtype()
        d = self._current_ema_decay()
        s_params = dict(self.qwen_vl_interface.named_parameters())
        for name, t_param in self.teacher_vlm.named_parameters():
            if name in s_params:
                t_param.data.mul_(d).add_(s_params[name].data, alpha=1.0 - d)
        # Also EMA buffers (e.g., layernorm running stats if any)
        s_buffers = dict(self.qwen_vl_interface.named_buffers())
        for name, t_buf in self.teacher_vlm.named_buffers():
            if name in s_buffers and t_buf.dtype == s_buffers[name].dtype and t_buf.is_floating_point():
                t_buf.data.mul_(d).add_(s_buffers[name].data, alpha=1.0 - d)
            elif name in s_buffers:
                t_buf.data.copy_(s_buffers[name].data)
        self._ema_step += 1

    # --------------------------------------------------------------- inference
    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        """
        Inference uses the STUDENT path with no augmentation and no paraphrase.
        The student has been trained to be robust; at test time we feed clean inputs.
        """
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [
            [to_pil_preserve(im) for im in (ex["image"] if isinstance(ex["image"], list) else [ex["image"]])]
            for ex in examples
        ]
        instructions = [ex["lang"] + self.latent_action_query for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        target_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if target_size:
            batch_images = resize_images(batch_images, target_size=target_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(batch_images, instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs, output_hidden_states=True, return_dict=True, use_cache=False
            )
            last = outputs.hidden_states[-1].to(torch.float32)
        h = self._extract_intent(
            last, qwen_inputs["input_ids"], self.qwen_vl_interface.processor.tokenizer
        )

        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(h.device, dtype=h.dtype)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(h, state_tensor)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}


# ==============================================================================
# Smoke test
# ==============================================================================
if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./examples/MultiRobot/train_files/starvla_cotrain_multiRobot.yaml",
    )
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)

    model = IntentVLA(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float32),
        "image": [image],
        "lang": "Carefully pick up the block and place it on the table.",
    }

    out = model([sample, sample])
    print(
        f"[forward] total={out['loss'].item():.4f} | "
        f"act_s={out['action_loss'].item():.4f} | "
        f"act_t={out['action_loss_teacher'].item():.4f} | "
        f"distill={out['distill_loss'].item():.4f}"
    )

    model.update_ema()
    pred = model.predict_action([sample])
    print(f"[predict] shape={pred['normalized_actions'].shape}")
    print("IntentVLA ready.")
