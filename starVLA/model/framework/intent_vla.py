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
#
# train_variant controls which training objective is used:
#   "base"               – clean input, action loss only (no aug, no distill)
#   "aug_only"           – perturbed input, action loss only (no distill)
#   "output_consistency" – clean + perturbed, velocity consistency on action head output
#   "ridevla"            – full method: EMA teacher + representation distillation

import contextlib
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.vlm.QWen2_5 import IMAGE_TOKEN_INDEX
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

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
        self.bank = self._load_bank(bank)

    @staticmethod
    def _load_bank(bank) -> Dict[str, List[str]]:
        if bank is None or bank == "":
            return {}
        if isinstance(bank, dict):
            return bank
        if isinstance(bank, str):
            bank_path = Path(bank).expanduser()
            if not bank_path.exists():
                raise FileNotFoundError(f"Paraphrase bank path does not exist: {bank_path}")
            try:
                with bank_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
            except Exception as exc:
                raise ValueError(f"Failed to load paraphrase bank from {bank_path}: {exc}") from exc
            if isinstance(loaded, dict):
                return loaded
            raise ValueError(f"Paraphrase bank at {bank_path} must be a JSON object, got {type(loaded).__name__}.")
        raise TypeError(
            f"Unsupported paraphrase_bank type {type(bank).__name__}. "
            "Expected None, empty string, dict, or a JSON file path."
        )

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
    Intent-level self-distillation VLA with four training variants.

    train_variant = "ridevla" (default):
        Teacher pass (EMA VLM, no grad): clean image + original text
        Student pass (trainable VLM):    augmented image + paraphrased text
        Losses: L_action_student + L_action_teacher + L_distill

    train_variant = "base":
        Single pass: clean image + original text
        Loss: L_action only

    train_variant = "aug_only":
        Single pass: augmented image + paraphrased text
        Loss: L_action only

    train_variant = "output_consistency":
        Two student passes: clean and perturbed
        Losses: L_action_clean + L_action_pert + L_velocity_consistency
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
        self.num_latent_action_query = int(self.config.framework.qwenvl.get("num_latent_action_query", 32))
        self.latent_action_query = "".join([f"<|action_{i}|>" for i in range(self.num_latent_action_query)])
        self.action_token_ids = None  # filled lazily

        # ----- Flow-matching action head (shared by all paths) -----
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        am = config.framework.action_model
        self.future_action_window_size = am.future_action_window_size
        self.past_action_window_size = am.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # ----- Train variant -----
        fw = self.config.framework
        self.train_variant = str(fw.get("train_variant", "ridevla")).lower()
        assert self.train_variant in {"base", "aug_only", "output_consistency", "ridevla"}, (
            f"Unknown train_variant: {self.train_variant}. "
            "Expected one of: base | aug_only | output_consistency | ridevla"
        )
        logger.info(f"[IntentVLA] train_variant = {self.train_variant}")

        # ----- Loss weights -----
        self.w_action_student = float(fw.get("w_action_student", 1.0))
        self.w_action_teacher = float(fw.get("w_action_teacher", 0.3))
        self.w_distill = float(fw.get("w_distill", 0.5))
        self.w_output_consistency = float(fw.get("w_output_consistency", 0.5))
        self.distill_regularizer = str(fw.get("distill_regularizer", "none")).lower()
        assert self.distill_regularizer in {"none", "projected_rank", "raw_rank"}, (
            f"Unknown distill_regularizer: {self.distill_regularizer}. Expected one of: none | projected_rank | raw_rank"
        )
        self.w_rank_var = float(fw.get("w_rank_var", 0.0))
        self.w_rank_cov = float(fw.get("w_rank_cov", 0.0))
        self.rank_var_gamma = float(fw.get("rank_var_gamma", 1.0))
        self.rank_eps = float(fw.get("rank_eps", 1e-4))
        self.distill_proj_dim = int(fw.get("distill_proj_dim", 0))
        self.distill_proj_hidden_dim = int(fw.get("distill_proj_hidden_dim", min(self.hidden_dim, 1024)))
        self.rank_preserve_on = str(fw.get("rank_preserve_on", "raw"))

        # Presets keep the paper-facing method compact. Individual values above
        # remain overrideable for diagnostics, but normal experiments should use
        # the preset rather than tune every coefficient.
        if self.distill_regularizer == "projected_rank":
            self.distill_proj_dim = int(fw.get("distill_proj_dim", 256))
            self.distill_proj_hidden_dim = int(fw.get("distill_proj_hidden_dim", min(self.hidden_dim, 1024)))
            self.rank_preserve_on = str(fw.get("rank_preserve_on", "projected"))
            self.w_rank_var = float(fw.get("w_rank_var", 1.0))
            self.w_rank_cov = float(fw.get("w_rank_cov", 0.005))
        elif self.distill_regularizer == "raw_rank":
            self.rank_preserve_on = str(fw.get("rank_preserve_on", "raw"))
            self.w_rank_var = float(fw.get("w_rank_var", 0.1))
            self.w_rank_cov = float(fw.get("w_rank_cov", 0.005))

        assert self.rank_preserve_on in {"raw", "projected"}, (
            f"Unknown rank_preserve_on: {self.rank_preserve_on}. Expected one of: raw | projected"
        )
        self.distill_type = str(fw.get("distill_type", "cosine"))  # 'cosine' | 'mse' | 'infonce'
        self.infonce_temp = float(fw.get("infonce_temp", 0.1))
        self.distill_locus = str(fw.get("distill_locus", "action_query"))
        assert self.distill_locus in {"action_query", "visual", "all_hidden", "output"}, (
            f"Unknown distill_locus: {self.distill_locus}. Expected one of: action_query | visual | all_hidden | output"
        )
        logger.info(f"[IntentVLA] distill_locus = {self.distill_locus}")
        logger.info(f"[IntentVLA] distill_regularizer = {self.distill_regularizer}")
        if self.w_rank_var > 0.0 or self.w_rank_cov > 0.0:
            logger.info(
                "[IntentVLA] rank-preserving regularization enabled: "
                f"w_rank_var={self.w_rank_var}, w_rank_cov={self.w_rank_cov}, "
                f"rank_var_gamma={self.rank_var_gamma}, rank_preserve_on={self.rank_preserve_on}"
            )

        # Optional training-only projection head for action-query distillation.
        # The flow-matching action head still consumes raw action-query states;
        # this head only controls the auxiliary clean-to-perturbed alignment.
        if self.distill_proj_dim > 0:
            self.distill_projector = nn.Sequential(
                nn.Linear(self.hidden_dim, self.distill_proj_hidden_dim),
                nn.GELU(),
                nn.Linear(self.distill_proj_hidden_dim, self.distill_proj_dim),
                nn.LayerNorm(self.distill_proj_dim),
            )
            logger.info(
                "[IntentVLA] projected action-query distillation enabled: "
                f"{self.hidden_dim}->{self.distill_proj_hidden_dim}->{self.distill_proj_dim}"
            )
        else:
            self.distill_projector = None

        # ----- EMA teacher (only created for ridevla to save memory) -----
        # NOTE: creating the teacher doubles VLM parameter memory.
        if self.train_variant == "ridevla":
            teacher_vlm = copy.deepcopy(self.qwen_vl_interface)
            for p in teacher_vlm.parameters():
                p.requires_grad_(False)
            teacher_vlm.eval()
            # Keep EMA teacher out of nn.Module registration so optimizer/DeepSpeed
            # only sees student parameters.
            object.__setattr__(self, "_teacher_vlm", teacher_vlm)
            logger.info("[IntentVLA] EMA teacher created.")
        else:
            object.__setattr__(self, "_teacher_vlm", None)

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
        if self.teacher_vlm is None:
            return
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
                "last": tokenizer.convert_tokens_to_ids(f"<|action_{self.num_latent_action_query - 1}|>"),
            }

    def _extract_intent(
        self,
        hidden_states: torch.Tensor,  # [B, S, H]
        input_ids: torch.Tensor,  # [B, S]
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
                f"Sample {b}: action block is not contiguous (expected <|action_{K - 1}|> at end)."
            )
            out.append(hidden_states[b, start:end, :])
        return torch.stack(out, dim=0)  # [B, K, H]

    def _encode_action_queries(
        self,
        vlm,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        tokenizer,
        no_grad: bool = False,
    ) -> torch.Tensor:
        """Run a VLM forward and return action-query hidden states [B, K, H]."""
        ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with ctx:
            inputs = vlm.build_qwenvl_inputs(batch_images, instructions)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = vlm(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                last = out.hidden_states[-1].to(torch.float32)
            return self._extract_intent(last, inputs["input_ids"], tokenizer)

    def _encode_full_hidden(
        self,
        vlm,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        tokenizer,
        no_grad: bool = False,
    ):
        """Run VLM forward; return (h_action [B,K,H], h_full [B,L,H], input_ids [B,L]).

        Used by visual / all_hidden loci which need the full last-layer hidden states,
        not just the action-query slice.
        """
        ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with ctx:
            inputs = vlm.build_qwenvl_inputs(batch_images, instructions)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = vlm(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                last = out.hidden_states[-1].to(torch.float32)
            h_action = self._extract_intent(last, inputs["input_ids"], tokenizer)
            return h_action, last, inputs["input_ids"]

    def _prep_action_tensors(self, actions, state, device, dtype):
        """Convert numpy actions/state to tensors and slice to the future window."""
        actions_t = torch.tensor(actions, device=device, dtype=dtype)
        actions_target = actions_t[:, -(self.future_action_window_size + 1) :, :]
        state_tensor = None
        if state is not None:
            state_tensor = torch.tensor(np.array(state), device=device, dtype=dtype)
        return actions_target, state_tensor

    def _compute_action_loss(
        self,
        h: torch.Tensor,
        actions_target: torch.Tensor,
        state_tensor,
    ) -> torch.Tensor:
        """Flow-matching action loss with repeated diffusion steps."""
        repeated_steps = (
            self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
        )
        actions_rep = actions_target.repeat(repeated_steps, 1, 1)
        state_rep = state_tensor.repeat(repeated_steps, 1, 1) if state_tensor is not None else None
        h_rep = h.repeat(repeated_steps, 1, 1).float()
        return self.action_model(h_rep, actions_rep, state_rep)

    def _velocity_consistency_loss(
        self,
        h_clean: torch.Tensor,
        h_pert: torch.Tensor,
        actions_target: torch.Tensor,
        state_tensor,
    ) -> torch.Tensor:
        """
        Velocity consistency: sample a random noise level, compute predicted velocity
        from both branches at the same noisy action, and penalise their divergence.
        Gradient flows through h_pert; h_clean is used as the stop-gradient target.
        """
        B = actions_target.shape[0]
        device = h_clean.device
        dtype = h_clean.dtype

        eps = torch.randn_like(actions_target)
        s = torch.rand(B, device=device, dtype=dtype)  # [B]
        s_bcast = s[:, None, None]  # [B, 1, 1]
        x_s = (1.0 - s_bcast) * eps + s_bcast * actions_target

        v_clean = self.action_model.predict_velocity(h_clean.float(), x_s, s, state_tensor)
        v_pert = self.action_model.predict_velocity(h_pert.float(), x_s, s, state_tensor)
        return F.mse_loss(v_pert, v_clean.detach())

    # ------------------------------------------------------ distillation losses
    def _distill_loss(self, h_s: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        h_s, h_t: [B, K, H]. h_t must be already detached.
        Computed per-token then averaged.
        """
        if self.distill_type == "cosine":
            cos = F.cosine_similarity(h_s, h_t, dim=-1)  # [B, K]
            return (1.0 - cos).mean()
        elif self.distill_type == "mse":
            return F.mse_loss(h_s, h_t)
        elif self.distill_type == "infonce":
            q = F.normalize(h_s.mean(dim=1), dim=-1)  # [B, H]
            k = F.normalize(h_t.mean(dim=1), dim=-1)  # [B, H]
            logits = q @ k.t() / self.infonce_temp  # [B, B]
            labels = torch.arange(q.size(0), device=q.device)
            return F.cross_entropy(logits, labels)
        else:
            raise ValueError(f"Unknown distill_type: {self.distill_type}")

    def _distill_repr(self, h: torch.Tensor) -> torch.Tensor:
        """Return the representation used by the auxiliary distillation loss."""
        if self.distill_projector is None:
            return h
        shape = h.shape[:-1]
        z = h.reshape(-1, h.shape[-1]).float()
        z = self.distill_projector(z)
        return z.reshape(*shape, z.shape[-1])

    def _select_visual_mean_pool(self, h_full: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool the visual-token hidden states for each sample. Returns [B, H].

        Uses IMAGE_TOKEN_INDEX to locate vision tokens in input_ids.
        Mean-pooling is required here because random crop can change the spatial
        layout of visual tokens, so token-wise alignment is not well-defined.
        """
        mask = input_ids == IMAGE_TOKEN_INDEX  # [B, L]
        out = []
        for b in range(h_full.shape[0]):
            vis_h = h_full[b, mask[b]]  # [n_vis, H]
            assert vis_h.numel() > 0, f"Sample {b}: no IMAGE tokens found in input_ids."
            out.append(vis_h.mean(0))
        return torch.stack(out)  # [B, H]

    def _distill_loss_pooled(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Cosine distillation on pooled [B, H] vectors. z_t must be already detached."""
        cos = F.cosine_similarity(z_s, z_t, dim=-1)  # [B]
        return (1.0 - cos).mean()

    def _rank_preserving_loss(self, h: torch.Tensor):
        """
        VICReg-style non-collapse regularization on action-query states.

        The invariance loss can be minimized by compressing all action-query
        embeddings into a low-rank subspace. This auxiliary objective keeps
        dimensions active (variance term) and discourages redundant dimensions
        (covariance term). It is disabled by default unless w_rank_var or
        w_rank_cov is set in the config.
        """
        if self.w_rank_var <= 0.0 and self.w_rank_cov <= 0.0:
            zero = h.new_zeros(())
            return zero, zero, zero

        z = h.reshape(-1, h.shape[-1]).float()  # [B*K, H]
        if z.shape[0] <= 1:
            zero = z.new_zeros(())
            return zero, zero, zero

        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + self.rank_eps)
        loss_var = F.relu(self.rank_var_gamma - std).mean()

        cov = (z.T @ z) / (z.shape[0] - 1)
        cov_offdiag = cov - torch.diag(torch.diag(cov))
        loss_cov = cov_offdiag.pow(2).sum() / z.shape[-1]

        loss_rank = self.w_rank_var * loss_var + self.w_rank_cov * loss_cov
        return loss_rank, loss_var, loss_cov

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

    # ------------------------------------------------------- variant forwards
    def _forward_base(self, batch_images_raw, instructions_raw, actions, state, tokenizer):
        """Clean input, single action loss, no augmentation."""
        instructions = [ins + self.latent_action_query for ins in instructions_raw]
        h = self._encode_action_queries(self.qwen_vl_interface, batch_images_raw, instructions, tokenizer)
        actions_target, state_tensor = self._prep_action_tensors(actions, state, h.device, h.dtype)
        with torch.autocast("cuda", dtype=torch.float32):
            loss = self._compute_action_loss(h, actions_target, state_tensor)
        zero = loss.new_zeros(())
        return {
            "loss": loss,
            "action_loss": loss.detach(),
            "action_loss_teacher": zero,
            "distill_loss": zero,
            "output_consistency_loss": zero,
        }

    def _forward_aug_only(self, batch_images_raw, instructions_raw, actions, state, tokenizer):
        """Perturbed input, single action loss. Same augmentation as ridevla for fair comparison."""
        batch_images_aug = self._augment_batch_images(batch_images_raw)
        instructions = [ins + self.latent_action_query for ins in self._paraphrase_batch(instructions_raw)]
        h = self._encode_action_queries(self.qwen_vl_interface, batch_images_aug, instructions, tokenizer)
        actions_target, state_tensor = self._prep_action_tensors(actions, state, h.device, h.dtype)
        with torch.autocast("cuda", dtype=torch.float32):
            loss = self._compute_action_loss(h, actions_target, state_tensor)
        zero = loss.new_zeros(())
        return {
            "loss": loss,
            "action_loss": loss.detach(),
            "action_loss_teacher": zero,
            "distill_loss": zero,
            "output_consistency_loss": zero,
        }

    def _forward_output_consistency(self, batch_images_raw, instructions_raw, actions, state, tokenizer):
        """
        Two student passes (clean + perturbed). No EMA teacher.
        Aligns action head *velocity output* rather than hidden representations.
        """
        instructions_clean = [ins + self.latent_action_query for ins in instructions_raw]
        batch_images_pert = self._augment_batch_images(batch_images_raw)
        instructions_pert = [ins + self.latent_action_query for ins in self._paraphrase_batch(instructions_raw)]

        h_clean = self._encode_action_queries(self.qwen_vl_interface, batch_images_raw, instructions_clean, tokenizer)
        h_pert = self._encode_action_queries(self.qwen_vl_interface, batch_images_pert, instructions_pert, tokenizer)

        actions_target, state_tensor = self._prep_action_tensors(actions, state, h_clean.device, h_clean.dtype)

        with torch.autocast("cuda", dtype=torch.float32):
            loss_act_clean = self._compute_action_loss(h_clean, actions_target, state_tensor)
            loss_act_pert = self._compute_action_loss(h_pert, actions_target, state_tensor)
            # h_clean is stop-grad target for velocity consistency — detach to avoid
            # building a second computation graph through the clean VLM activations.
            loss_out = self._velocity_consistency_loss(h_clean.detach(), h_pert, actions_target, state_tensor)

        total = (
            self.w_action_student * loss_act_pert
            + self.w_action_teacher * loss_act_clean
            + self.w_output_consistency * loss_out
        )
        zero = total.new_zeros(())
        return {
            "loss": total,
            "action_loss": loss_act_pert.detach(),
            "action_loss_teacher": loss_act_clean.detach(),
            "distill_loss": zero,
            "output_consistency_loss": loss_out.detach(),
        }

    def _forward_ridevla(self, batch_images_raw, instructions_raw, actions, state, tokenizer):
        """Full RIDE-VLA: EMA teacher on clean view + student on perturbed view + distillation.

        The distillation target is controlled by self.distill_locus:
          "action_query" (default) – token-wise cosine on the K action-query hidden states [B,K,H]
          "visual"                 – mean-pool cosine over visual-token hidden states [B,H]
          "all_hidden"             – mean-pool cosine over the full last-layer sequence [B,H]
          "output"                 – MSE on action-head velocity with shared (s, eps) noise
        All four variants share the same backbone, EMA teacher, augmentation pipeline,
        lambda weights, and training steps for a fair locus ablation.
        """
        assert self.teacher_vlm is not None, "ridevla requires EMA teacher."

        instructions_teacher = [ins + self.latent_action_query for ins in instructions_raw]
        batch_images_student = self._augment_batch_images(batch_images_raw)
        instructions_student = [ins + self.latent_action_query for ins in self._paraphrase_batch(instructions_raw)]

        self._sync_teacher_device_dtype()

        # ------------------------------------------------------------------
        # Encode teacher and student; compute locus-specific distill loss.
        # ------------------------------------------------------------------
        z_student_for_rank = None
        if self.distill_locus in {"visual", "all_hidden"}:
            # Need full last-layer hidden states to select the right token slice.
            h_teacher, h_full_teacher, ids_teacher = self._encode_full_hidden(
                self.teacher_vlm, batch_images_raw, instructions_teacher, tokenizer, no_grad=True
            )
            h_teacher_sg = h_teacher.detach()

            h_student, h_full_student, ids_student = self._encode_full_hidden(
                self.qwen_vl_interface, batch_images_student, instructions_student, tokenizer
            )

            if self.distill_locus == "visual":
                z_t = self._select_visual_mean_pool(h_full_teacher.detach(), ids_teacher)
                z_s = self._select_visual_mean_pool(h_full_student, ids_student)
            else:  # all_hidden: mean-pool the entire sequence
                z_t = h_full_teacher.detach().mean(dim=1)  # [B, H]
                z_s = h_full_student.mean(dim=1)  # [B, H]

            loss_distill = self._distill_loss_pooled(z_s, z_t)

        else:
            # action_query and output loci only need action-query hidden states.
            h_teacher_sg = self._encode_action_queries(
                self.teacher_vlm, batch_images_raw, instructions_teacher, tokenizer, no_grad=True
            ).detach()
            h_student = self._encode_action_queries(
                self.qwen_vl_interface, batch_images_student, instructions_student, tokenizer
            )
            # loss_distill for "output" is computed below after _prep_action_tensors.
            if self.distill_locus == "action_query":
                z_teacher = self._distill_repr(h_teacher_sg).detach()
                z_student = self._distill_repr(h_student)
                z_student_for_rank = z_student
                loss_distill = self._distill_loss(z_student, z_teacher)

        actions_target, state_tensor = self._prep_action_tensors(actions, state, h_student.device, h_student.dtype)

        if self.distill_locus == "output":
            # Align action-head velocity with shared (s, eps): teacher is stop-grad target.
            with torch.autocast("cuda", dtype=torch.float32):
                loss_distill = self._velocity_consistency_loss(h_teacher_sg, h_student, actions_target, state_tensor)

        with torch.autocast("cuda", dtype=torch.float32):
            loss_act_student = self._compute_action_loss(h_student, actions_target, state_tensor)
            # h_teacher_sg is detached: grad flows only through shared action head params.
            loss_act_teacher = self._compute_action_loss(h_teacher_sg, actions_target, state_tensor)
            rank_source = (
                z_student_for_rank
                if self.rank_preserve_on == "projected" and z_student_for_rank is not None
                else h_student
            )
            loss_rank, loss_rank_var, loss_rank_cov = self._rank_preserving_loss(rank_source)

        total = (
            self.w_action_student * loss_act_student
            + self.w_action_teacher * loss_act_teacher
            + self.w_distill * loss_distill
            + loss_rank
        )
        zero = total.new_zeros(())
        return {
            "loss": total,
            "action_loss": loss_act_student.detach(),
            "action_loss_teacher": loss_act_teacher.detach(),
            "distill_loss": loss_distill.detach(),
            "rank_preserve_loss": loss_rank.detach(),
            "rank_var_loss": loss_rank_var.detach(),
            "rank_cov_loss": loss_rank_cov.detach(),
            "output_consistency_loss": zero,
        }

    # ------------------------------------------------------------------ forward
    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        tokenizer = self.qwen_vl_interface.processor.tokenizer

        batch_images_raw = [
            [to_pil_preserve(im) for im in (ex["image"] if isinstance(ex["image"], list) else [ex["image"]])]
            for ex in examples
        ]
        instructions_raw = [ex["lang"] for ex in examples]
        actions = np.array([ex["action"] for ex in examples])
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        if self.train_variant == "base":
            return self._forward_base(batch_images_raw, instructions_raw, actions, state, tokenizer)
        if self.train_variant == "aug_only":
            return self._forward_aug_only(batch_images_raw, instructions_raw, actions, state, tokenizer)
        if self.train_variant == "output_consistency":
            return self._forward_output_consistency(batch_images_raw, instructions_raw, actions, state, tokenizer)
        return self._forward_ridevla(batch_images_raw, instructions_raw, actions, state, tokenizer)

    # ------------------------------------------------------------------ EMA
    def _current_ema_decay(self) -> float:
        r = min(self._ema_step / max(self.ema_rampup_steps, 1), 1.0)
        return self.ema_decay_init + (self.ema_decay_final - self.ema_decay_init) * r

    @torch.no_grad()
    def update_ema(self):
        """Call once per optimizer step, AFTER student weights are updated. No-op for non-ridevla."""
        if self.teacher_vlm is None:
            return
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
            outputs = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            last = outputs.hidden_states[-1].to(torch.float32)
        h = self._extract_intent(last, qwen_inputs["input_ids"], self.qwen_vl_interface.processor.tokenizer)

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
    import argparse
    import warnings

    from omegaconf import OmegaConf

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./examples/MultiRobot/train_files/starvla_cotrain_multiRobot.yaml",
    )
    parser.add_argument(
        "--variant", type=str, default="ridevla", choices=["base", "aug_only", "output_consistency", "ridevla"]
    )
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.train_variant = args.variant

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
        f"[{args.variant}] total={out['loss'].item():.4f} | "
        f"act_s={out['action_loss'].item():.4f} | "
        f"act_t={out['action_loss_teacher'].item():.4f} | "
        f"distill={out['distill_loss'].item():.4f} | "
        f"out_cons={out['output_consistency_loss'].item():.4f}"
    )

    if args.variant == "ridevla":
        model.update_ema()

    pred = model.predict_action([sample])
    print(f"[predict] shape={pred['normalized_actions'].shape}")
    print("IntentVLA ready.")
