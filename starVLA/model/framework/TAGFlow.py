# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
TAG-Flow v2: Trajectory-Aligned Guided Flow Matching
=====================================================
Single-branch VLA framework with two core innovations:

Training ("大道至简"):
  - Single VLM forward with condition dropout (p=cond_drop_rate).
    Dropped samples: instruction = action_tokens only => p(a|v)  (prior)
    Kept samples:    instruction = lang + action_tokens => pi(a|v,l) (posterior)
  - Pure flow matching loss — no contrastive, no KL, no dual-branch.
  - No Train-Test Gap: guidance operates on velocity / conditioning space,
    which the DiT always sees during training.

Inference ("时空双重 Guidance"):
  Two VLM passes (cond + uncond), then a custom Euler integration loop with:

  1. **Spatial** (Token-wise): Per-action-query omega computed from last-layer
     attention using sigmoid z-score normalization (NOT min-max), so tokens
     that genuinely don't attend to language get low omega.

  2. **Temporal** (Time-scheduled): omega decays from high (early, noisy steps)
     to low (late, clean steps) via cosine/linear schedule, so coarse direction
     follows the instruction while fine motor control stays smooth.

  3. **Combined**: current_omega = 1 + (spatial_omega - 1) * temporal_decay
     Applied at the conditioning level: h_guided = h_uncond + omega * (h_cond - h_uncond)
     then a single DiT forward per step (more efficient than velocity-level CFG
     which needs 2 DiT passes per step).

Guidance modes:
  * "spatiotemporal" – token-wise spatial omega + temporal decay (recommended)
  * "scheduled"      – uniform omega with time schedule (temporal only)
  * "velocity"       – uniform omega on each flow-matching step (BayesianCAG compat)
  * "action"         – uniform omega on final predicted actions
"""
import sys
import math
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# ===== Qwen special tokens =====
VISION_START_TOKEN_INDEX = 151652  # <|vision_start|>
VISION_END_TOKEN_INDEX   = 151654  # <|vision_end|>
IMAGE_TOKEN_INDEX        = 151655  # <|image_pad|>
VIDEO_TOKEN_INDEX        = 151656  # <|video_pad|>


@FRAMEWORK_REGISTRY.register("TAGFlow")
class TAGFlow(baseframework):
    """
    TAG-Flow v2: Single-branch VLA with condition dropout + spatio-temporal guidance.

    Config keys (under ``cfg.framework``):
        cond_drop_rate       (float): Language dropout probability.     Default 0.15.
        omega_max            (float): Max guidance scale (early steps). Default 3.0.
        omega_min            (float): Min guidance scale (late steps).  Default 1.0.
        omega_base           (float): Min per-token omega floor.        Default 1.0.
        omega_schedule       (str):   "cosine" or "linear".            Default "cosine".
        guidance_mode        (str):   Default inference guidance mode.  Default "spatiotemporal".
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # align dims
        hidden_size = self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = hidden_size

        # Action tokens (same format as LangForce: <|action_X|>)
        self.num_latent_action_query = self.config.framework.qwenvl.get("num_latent_action_query", 32)
        self.latent_action_query = "".join([f"<|action_{i}|>" for i in range(self.num_latent_action_query)])
        self.action_token_ids = None  # cached {'first','last'}

        # Action model
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # ===== Condition Dropout =====
        self.cond_drop_rate = float(self.config.framework.get("cond_drop_rate", 0.15))

        # ===== Spatio-Temporal Guidance =====
        self.omega_max = float(self.config.framework.get("omega_max", 3.0))
        self.omega_min = float(self.config.framework.get("omega_min", 1.0))
        self.omega_base = float(self.config.framework.get("omega_base", 1.0))
        self.omega_schedule = str(self.config.framework.get("omega_schedule", "cosine"))
        self.guidance_mode = str(self.config.framework.get("guidance_mode", "spatiotemporal"))

        # ===== Vision cross-attention injection in DiT =====
        self.enable_vision_cross_attn = bool(
            self.config.framework.action_model.get("enable_vision_cross_attn", False)
        )
        self.vis_proj = None
        if self.enable_vision_cross_attn:
            self._vis_cross_dim = int(self.config.framework.action_model.get("vision_cross_attn_dim", 2048))
            vit_output_dim = int(self.config.framework.action_model.get("vit_output_dim", 0))
            if vit_output_dim > 0 and vit_output_dim != self._vis_cross_dim:
                self.vis_proj = nn.Linear(vit_output_dim, self._vis_cross_dim)
                logger.info(f"[TAGFlow] Vision projection: {vit_output_dim} -> {self._vis_cross_dim}")

        logger.info(
            f"[TAGFlow] Single-branch with cond_dropout={self.cond_drop_rate}, "
            f"omega_max={self.omega_max}, omega_min={self.omega_min}, "
            f"omega_base={self.omega_base}, schedule={self.omega_schedule}, "
            f"guidance_mode={self.guidance_mode}, "
            f"vision_cross_attn={self.enable_vision_cross_attn}"
        )

    # ------------------------------------------------------------------
    # Token id helpers
    # ------------------------------------------------------------------
    def _ensure_action_token_ids(self, tokenizer):
        if self.action_token_ids is None:
            first = tokenizer.convert_tokens_to_ids("<|action_0|>")
            last = tokenizer.convert_tokens_to_ids(f"<|action_{self.num_latent_action_query-1}|>")
            if first is None or last is None:
                raise RuntimeError(
                    "Action tokens (<|action_0|>, ..., <|action_N|>) not found in tokenizer. "
                    "Please run add_special_tokens_to_qwen.py with langforce_tokens.txt "
                    "to add action tokens before training."
                )
            self.action_token_ids = {"first": first, "last": last}

    def _get_action_block_start(self, input_ids_1d: torch.Tensor, tokenizer) -> int:
        self._ensure_action_token_ids(tokenizer)
        first_id = self.action_token_ids["first"]
        last_id = self.action_token_ids["last"]

        pos = (input_ids_1d == int(first_id)).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            return -1

        start = int(pos[0].item())
        end = start + self.num_latent_action_query
        if end > input_ids_1d.shape[0]:
            return -1
        if int(input_ids_1d[end - 1].item()) != int(last_id):
            return -1
        return start

    def _find_last_pos(self, seq_1d: torch.Tensor, token_id: int) -> int:
        idx = (seq_1d == int(token_id)).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return -1
        return int(idx[-1].item())

    def _extract_action_query_hidden_states(
        self,
        hidden_states: torch.Tensor,   # [B, S, H]
        input_ids: torch.Tensor,       # [B, S]
        tokenizer,
        return_starts: bool = False,
    ):
        self._ensure_action_token_ids(tokenizer)

        B = hidden_states.shape[0]
        out = []
        starts = []
        for b in range(B):
            start = self._get_action_block_start(input_ids[b], tokenizer)
            assert start != -1, "No valid contiguous action token block found in the sequence."
            end = start + self.num_latent_action_query
            out.append(hidden_states[b, start:end, :])
            starts.append(start)

        out = torch.stack(out, dim=0)  # [B, K, H]
        if return_starts:
            return out, torch.tensor(starts, device=input_ids.device, dtype=torch.long)
        return out

    # ------------------------------------------------------------------
    # Extract raw ViT features from Qwen2.5-VL visual encoder
    # ------------------------------------------------------------------
    def _extract_vision_features(
        self,
        qwen_inputs: dict,
    ) -> torch.Tensor:
        pixel_values = qwen_inputs.get("pixel_values", None)
        image_grid_thw = qwen_inputs.get("image_grid_thw", None)
        if pixel_values is None or image_grid_thw is None:
            return None

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vit_output = self.qwen_vl_interface.model.visual(
                pixel_values.to(dtype=torch.bfloat16), grid_thw=image_grid_thw,
            )

        if isinstance(vit_output, tuple):
            vit_output = vit_output[0]

        D_vis = vit_output.shape[-1]

        merge_size = getattr(
            self.qwen_vl_interface.model.visual, "spatial_merge_size",
            getattr(self.qwen_vl_interface.model.visual, "merge_size", 2),
        )
        patches_per_image = []
        for row in image_grid_thw:
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            n = t * (h // merge_size) * (w // merge_size)
            patches_per_image.append(n)

        input_ids = qwen_inputs.get("input_ids", None)
        if input_ids is not None:
            imgs_per_sample = []
            for b in range(input_ids.shape[0]):
                n_imgs = int((input_ids[b] == int(VISION_START_TOKEN_INDEX)).sum().item())
                imgs_per_sample.append(n_imgs)
        else:
            B_guess = max(1, len(patches_per_image))
            imgs_per_sample = [B_guess]

        per_image_feats = torch.split(vit_output, patches_per_image, dim=0)
        vis_list = []
        img_idx = 0
        for n_imgs in imgs_per_sample:
            if n_imgs > 0 and img_idx < len(per_image_feats):
                sample_feats = torch.cat(
                    [per_image_feats[img_idx + j] for j in range(n_imgs)
                     if img_idx + j < len(per_image_feats)],
                    dim=0,
                )
                vis_list.append(sample_feats)
                img_idx += n_imgs
            else:
                vis_list.append(torch.zeros(1, D_vis, device=vit_output.device, dtype=vit_output.dtype))

        max_len = max(v.shape[0] for v in vis_list)
        padded = []
        for v in vis_list:
            if v.shape[0] < max_len:
                pad = torch.zeros(max_len - v.shape[0], D_vis, device=v.device, dtype=v.dtype)
                padded.append(torch.cat([v, pad], dim=0))
            else:
                padded.append(v)
        out = torch.stack(padded, dim=0)  # [B, N_vis_max, D_vis]

        if self.vis_proj is not None:
            out = self.vis_proj(out)
        elif hasattr(self, "_vis_cross_dim") and D_vis != self._vis_cross_dim:
            self.vis_proj = nn.Linear(D_vis, self._vis_cross_dim).to(
                device=out.device, dtype=out.dtype
            )
            logger.warning(
                f"[TAGFlow] Lazily created vis_proj ({D_vis} -> {self._vis_cross_dim}). "
                f"Set vit_output_dim: {D_vis} in config to avoid this."
            )
            out = self.vis_proj(out)

        return out

    # ------------------------------------------------------------------
    # Attention capture (for spatio-temporal guidance)
    # ------------------------------------------------------------------
    def _get_last_layer_attn(self):
        """Return the last transformer layer's self_attn module."""
        vlm = self.qwen_vl_interface.model
        inner = getattr(vlm, "model", vlm)
        if hasattr(inner, "layers"):
            return inner.layers[-1].self_attn
        elif hasattr(inner, "language_model"):
            return inner.language_model.layers[-1].self_attn
        else:
            raise AttributeError(
                f"Cannot locate transformer layers on {type(inner).__name__}. "
                f"Available attributes: {[n for n, _ in inner.named_children()]}"
            )

    def _capture_last_layer_attn_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple:
        """
        Register a one-shot forward hook on the last layer's self_attn to
        manually compute attention weights from Q and K projections.
        Works with any attention backend (including FlashAttention).

        Returns: (hook_handle, captured_dict)
        """
        attn_module = self._get_last_layer_attn()
        captured = {}

        def _hook(module, args, kwargs, output):
            if args:
                hidden_states = args[0]
            else:
                hidden_states = kwargs["hidden_states"]
            B, S, _ = hidden_states.shape

            q = module.q_proj(hidden_states)
            k = module.k_proj(hidden_states)

            num_heads = getattr(module, "num_heads", None) or module.config.num_attention_heads
            head_dim = module.head_dim
            num_kv_heads = getattr(module, "num_key_value_heads", None) or getattr(module.config, "num_key_value_heads", num_heads)
            num_kv_groups = num_heads // num_kv_heads

            q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, S, num_kv_heads, head_dim).transpose(1, 2)

            if num_kv_groups > 1:
                k = k.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1)
                k = k.reshape(B, num_heads, S, head_dim)

            scale = head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

            causal = torch.triu(
                torch.ones(S, S, device=attn_weights.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
            captured["attn"] = attn_weights.detach()

        handle = attn_module.register_forward_hook(_hook, with_kwargs=True)
        return handle, captured

    # ------------------------------------------------------------------
    # Token-wise spatial omega (sigmoid z-score, NOT min-max)
    # ------------------------------------------------------------------
    def _compute_spatial_omega(
        self,
        attn_avg: torch.Tensor,     # [B, S, S]  head-averaged attention
        input_ids: torch.Tensor,     # [B, S]
        action_starts: torch.Tensor, # [B]
        omega_max: float,
    ) -> torch.Tensor:
        """
        Per-action-query omega using sigmoid z-score normalization.

        Unlike BayesianCAG's min-max normalization which always assigns omega_max
        to at least one token (even when all tokens have low attention), this
        approach uses absolute thresholds: tokens with genuinely low attention
        to language get omega close to omega_base.

        Returns: [B, K, 1]
        """
        B = input_ids.shape[0]
        K = self.num_latent_action_query
        omega_vecs = []

        for b in range(B):
            ids = input_ids[b]
            a_start = int(action_starts[b].item())

            action_range = slice(a_start, a_start + K)

            v_end = self._find_last_pos(ids, VISION_END_TOKEN_INDEX)
            if v_end == -1 or v_end + 1 >= a_start:
                omega_vecs.append(torch.full((K,), omega_max, device=ids.device))
                continue

            lang_range = slice(v_end + 1, a_start)

            attn_block = attn_avg[b, action_range, lang_range]  # [K, N_lang]
            s_k = attn_block.sum(dim=-1)                        # [K]

            # Sigmoid z-score: absolute threshold instead of relative min-max
            # Use unbiased=False to avoid NaN when K=1 (single element std)
            s_mean = s_k.mean()
            s_std = s_k.std(unbiased=False) + 1e-6
            s_norm = torch.sigmoid((s_k - s_mean) / s_std)

            omega_k = self.omega_base + (omega_max - self.omega_base) * s_norm
            omega_vecs.append(omega_k)

        return torch.stack(omega_vecs, dim=0).unsqueeze(-1)  # [B, K, 1]

    # ------------------------------------------------------------------
    # Training forward: 大道至简 (pure flow matching + condition dropout)
    # ------------------------------------------------------------------
    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        batch_images = [example["image"] for example in examples]
        batch_langs = [example["lang"] for example in examples]

        # Resize images to prevent Qwen2.5-VL dynamic-resolution OOM
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        B = len(examples)

        # ===== Condition Dropout: single-branch design =====
        instructions = []
        for i in range(B):
            if random.random() < self.cond_drop_rate:
                # Unconditional: action tokens only (no language, pure visual prior)
                instructions.append(self.latent_action_query)
            else:
                # Conditional: lang + action tokens (posterior: V + L + A)
                instructions.append(batch_langs[i] + self.latent_action_query)

        # ===== Single VLM Forward Pass (显存省一半，速度翻倍) =====
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            action_hidden = self._extract_action_query_hidden_states(
                last_hidden,
                qwen_inputs["input_ids"],
                self.qwen_vl_interface.processor.tokenizer,
            )  # [B, K, H]

        # ===== Action head: pure flow matching loss =====
        with torch.autocast("cuda", dtype=torch.float32):
            actions_t = torch.tensor(
                np.array(actions), device=action_hidden.device, dtype=action_hidden.dtype
            )
            actions_target = actions_t[:, -(self.future_action_window_size + 1):, :]

            repeated_diffusion_steps = int(
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and self.config.framework
                else 4
            )

            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state), device=action_hidden.device, dtype=action_hidden.dtype
                )

            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            action_cond_repeated = action_hidden.repeat(repeated_diffusion_steps, 1, 1).float()
            state_repeated = state_tensor.repeat(repeated_diffusion_steps, 1, 1) if state_tensor is not None else None

            vis_feats_repeated = None
            if self.enable_vision_cross_attn:
                vis_feats = self._extract_vision_features(qwen_inputs)
                if vis_feats is not None:
                    vis_feats_repeated = vis_feats.repeat(repeated_diffusion_steps, 1, 1).float()

            flow_loss = self.action_model(
                action_cond_repeated, actions_target_repeated, state_repeated,
                vision_features=vis_feats_repeated,
            )

        return {
            "action_loss": flow_loss,
            "flow_loss": flow_loss.detach(),
        }

    # ------------------------------------------------------------------
    # Custom Euler loop with spatio-temporal conditioning-level guidance
    # ------------------------------------------------------------------
    def _predict_action_spatiotemporal(
        self,
        action_hidden_cond: torch.Tensor,    # [B, K, H]
        action_hidden_uncond: torch.Tensor,  # [B, K, H]
        spatial_omega: torch.Tensor,         # [B, K, 1]
        state_tensor: Optional[torch.Tensor],
        vis_feats: Optional[torch.Tensor],
        omega_max: float,
        omega_min: float,
        schedule: str,
    ) -> torch.Tensor:
        """
        Custom Euler integration with spatio-temporal guidance at the
        **velocity level** using batched cond/uncond DiT forward.

        CFG must operate on velocity: v = v_uncond + omega * (v_cond - v_uncond).
        Applying guidance on conditioning latents (h_guided = h_uncond + omega * delta)
        feeds out-of-distribution "Frankenstein" embeddings into the DiT, because the
        model only ever saw pure h_cond or h_uncond during training, never their
        linear extrapolation.  The DiT's internal nonlinearities (attention, MLP)
        make f(h_uncond + omega * delta) != f(h_uncond) + omega * (f(h_cond) - f(h_uncond)).

        To keep efficiency high we concat cond and uncond into a single [2B] batch
        for the DiT forward, then split the output and apply spatio-temporal omega
        on the resulting velocity field.

        Spatial omega [B, K, 1] is derived from per-action-query attention to
        language tokens (sigmoid z-score).  We reduce it to a per-sample scalar
        [B, 1, 1] for velocity-level guidance since the velocity tensor has shape
        [B, action_horizon, action_dim] which differs from the K-dimensional
        conditioning space.

        At each denoising step:
          1. Compute temporal decay based on denoising progress
          2. Combine spatial_omega (reduced to per-sample) with temporal decay
          3. Run DiT ONCE on [2B] batch (uncond || cond)
          4. Split velocities, apply CFG: v = v_uncond + omega * (v_cond - v_uncond)
          5. Euler update
        """
        am = self.action_model
        batch_size = action_hidden_cond.shape[0]
        device = action_hidden_cond.device

        actions = torch.randn(
            size=(batch_size, am.config.action_horizon, am.config.action_dim),
            dtype=action_hidden_cond.dtype,
            device=device,
        )

        num_steps = am.num_inference_timesteps
        dt = 1.0 / num_steps

        # Concat cond/uncond conditioning for batched DiT forward
        double_h = torch.cat([action_hidden_uncond, action_hidden_cond], dim=0)  # [2B, K, H]

        state_features = am.state_encoder(state_tensor) if state_tensor is not None else None
        double_state_features = (
            torch.cat([state_features, state_features], dim=0)
            if state_features is not None else None
        )
        double_vis_feats = (
            torch.cat([vis_feats, vis_feats], dim=0)
            if vis_feats is not None else None
        )

        # Reduce spatial omega from [B, K, 1] to per-sample [B, 1, 1]
        # (velocity shape [B, action_horizon, action_dim] != K)
        spatial_omega_scalar = spatial_omega.mean(dim=1, keepdim=True)  # [B, 1, 1]

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * am.num_timestep_buckets)

            # --- Temporal decay: high omega early (noisy), low omega late (clean) ---
            progress = t / float(max(num_steps - 1, 1))  # 0 -> 1
            if schedule == "cosine":
                temporal_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            else:  # linear
                temporal_decay = 1.0 - progress

            # --- Spatio-temporal combined omega ---
            current_omega = 1.0 + (spatial_omega_scalar - 1.0) * temporal_decay  # [B, 1, 1]

            # --- Batched DiT forward (uncond + cond in one pass) ---
            double_actions = torch.cat([actions, actions], dim=0)  # [2B, T, D]
            timesteps_tensor = torch.full(
                size=(2 * batch_size,), fill_value=t_discretized, device=device,
            )
            action_features = am.action_encoder(double_actions, timesteps_tensor)

            if am.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = am.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = am.future_tokens.weight.unsqueeze(0).expand(2 * batch_size, -1, -1)
            sa_embs = (
                torch.cat((double_state_features, future_tokens, action_features), dim=1)
                if double_state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            model_output = am.model(
                hidden_states=sa_embs,
                encoder_hidden_states=double_h,
                timestep=timesteps_tensor,
                vision_features=double_vis_feats,
            )
            double_velocity = am.action_decoder(model_output)[:, -am.action_horizon:]

            # --- Split and apply velocity-level CFG ---
            v_uncond, v_cond = torch.chunk(double_velocity, 2, dim=0)
            pred_velocity = v_uncond + current_omega * (v_cond - v_uncond)

            # --- Euler update ---
            actions = actions + dt * pred_velocity

        return actions

    # ------------------------------------------------------------------
    # Inference entry point
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        omega: Optional[float] = None,
        omega_max: Optional[float] = None,
        omega_min: Optional[float] = None,
        omega_schedule: Optional[str] = None,
        guidance_mode: Optional[str] = None,
        **kwargs,
    ) -> dict:
        if not isinstance(examples, list):
            examples = [examples]

        o_max = omega_max if omega_max is not None else self.omega_max
        o_min = omega_min if omega_min is not None else self.omega_min
        schedule = omega_schedule if omega_schedule is not None else self.omega_schedule

        # Legacy: if omega is provided as a single float, use it as omega_max
        if omega is not None and omega > 0:
            o_max = omega

        mode = guidance_mode if guidance_mode is not None else self.guidance_mode

        # --- image pre-processing ---
        batch_images = []
        for ex in examples:
            imgs = ex["image"]
            if isinstance(imgs, list):
                batch_images.append([to_pil_preserve(im) for im in imgs])
            else:
                batch_images.append([to_pil_preserve(imgs)])

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        tokenizer = self.qwen_vl_interface.processor.tokenizer

        # ============================================================
        # Conditioned pass: V + L + A
        # ============================================================
        instructions_cond = [ex["lang"] + self.latent_action_query for ex in examples]
        qwen_inputs_cond = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions_cond,
        )

        # For spatiotemporal mode, capture attention weights from the cond pass
        need_attn = (mode == "spatiotemporal")
        hook_handle, hook_captured = None, {}
        if need_attn:
            hook_handle, hook_captured = self._capture_last_layer_attn_weights(
                qwen_inputs_cond.get("input_ids"),
                qwen_inputs_cond.get("attention_mask"),
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs_cond = self.qwen_vl_interface(
                **qwen_inputs_cond,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

            if hook_handle is not None:
                hook_handle.remove()

            hidden_cond = outputs_cond.hidden_states[-1]
            action_hidden_cond, cond_action_starts = self._extract_action_query_hidden_states(
                hidden_cond, qwen_inputs_cond["input_ids"], tokenizer,
                return_starts=True,
            )

            # Compute spatial omega from attention if in spatiotemporal mode
            spatial_omega = None
            if need_attn:
                attn_weights = hook_captured["attn"]   # [B, num_heads, S, S]
                attn_avg = attn_weights.mean(dim=1)     # [B, S, S]
                spatial_omega = self._compute_spatial_omega(
                    attn_avg,
                    qwen_inputs_cond["input_ids"],
                    cond_action_starts,
                    omega_max=o_max,
                )  # [B, K, 1]
                spatial_omega = spatial_omega.to(
                    dtype=action_hidden_cond.dtype,
                    device=action_hidden_cond.device,
                )

        # ============================================================
        # Unconditioned pass: A only (no language)
        # ============================================================
        instructions_uncond = [self.latent_action_query for _ in examples]
        qwen_inputs_uncond = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions_uncond,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs_uncond = self.qwen_vl_interface(
                **qwen_inputs_uncond,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_uncond = outputs_uncond.hidden_states[-1]
            action_hidden_uncond = self._extract_action_query_hidden_states(
                hidden_uncond, qwen_inputs_uncond["input_ids"], tokenizer,
            )

        # ============================================================
        # State tensor
        # ============================================================
        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(
                action_hidden_cond.device, dtype=action_hidden_cond.dtype
            )

        # ============================================================
        # Vision features
        # ============================================================
        vis_feats = None
        if self.enable_vision_cross_attn:
            vis_feats = self._extract_vision_features(qwen_inputs_cond)

        # ============================================================
        # Action prediction with guidance
        # ============================================================
        with torch.autocast("cuda", dtype=torch.float32):
            vis_feats_f = vis_feats.float() if vis_feats is not None else None

            if mode == "spatiotemporal":
                # Core TAG-Flow innovation: spatio-temporal conditioning-level guidance
                pred_actions = self._predict_action_spatiotemporal(
                    action_hidden_cond=action_hidden_cond.float(),
                    action_hidden_uncond=action_hidden_uncond.float(),
                    spatial_omega=spatial_omega.float(),
                    state_tensor=state_tensor,
                    vis_feats=vis_feats_f,
                    omega_max=o_max,
                    omega_min=o_min,
                    schedule=schedule,
                )
            elif mode == "scheduled":
                # Time-scheduled velocity-level CFG (temporal only, uniform across tokens)
                pred_actions = self.action_model.predict_action_guided_scheduled(
                    vl_embs_cond=action_hidden_cond.float(),
                    vl_embs_uncond=action_hidden_uncond.float(),
                    state=state_tensor,
                    omega_max=o_max,
                    omega_min=o_min,
                    schedule=schedule,
                    vision_features=vis_feats_f,
                )
            elif mode == "velocity":
                # Uniform omega on each denoising step
                pred_actions = self.action_model.predict_action_guided(
                    vl_embs_cond=action_hidden_cond.float(),
                    vl_embs_uncond=action_hidden_uncond.float(),
                    state=state_tensor,
                    omega=o_max,
                    vision_features=vis_feats_f,
                )
            elif mode == "action":
                # Uniform omega on final actions
                a_cond = self.action_model.predict_action(
                    action_hidden_cond.float(), state_tensor, vision_features=vis_feats_f,
                )
                a_uncond = self.action_model.predict_action(
                    action_hidden_uncond.float(), state_tensor, vision_features=vis_feats_f,
                )
                pred_actions = a_uncond + o_max * (a_cond - a_uncond)
            else:
                # No guidance (posterior only)
                pred_actions = self.action_model.predict_action(
                    action_hidden_cond.float(), state_tensor, vision_features=vis_feats_f,
                )

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}


# ======================================================================
# Stand-alone smoke test
# ======================================================================
if __name__ == "__main__":
    from omegaconf import OmegaConf
    from PIL import Image
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero_tag_flow.yaml",
    )
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    model: TAGFlow = TAGFlow(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "Pick the red block and place it on the blue plate.",
    }
    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "Open the drawer and put the cup inside.",
    }

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Test training forward (clean: just flow_loss)
    out = model(batch)
    print(f"Action Loss: {out['action_loss'].item()}, Flow Loss: {out['flow_loss'].item()}")

    # Test spatio-temporal guidance (default, recommended)
    pred = model.predict_action([sample], omega_max=3.0, omega_min=1.0,
                                guidance_mode="spatiotemporal")
    print(f"Spatio-temporal pred shape: {pred['normalized_actions'].shape}")

    # Test scheduled guidance (temporal only)
    pred_s = model.predict_action([sample], omega_max=3.0, omega_min=1.0,
                                  guidance_mode="scheduled")
    print(f"Scheduled pred shape: {pred_s['normalized_actions'].shape}")

    # Test velocity-level guidance
    pred_v = model.predict_action([sample], omega=2.0, guidance_mode="velocity")
    print(f"Velocity-level pred shape: {pred_v['normalized_actions'].shape}")

    # Test action-level guidance
    pred_a = model.predict_action([sample], omega=2.0, guidance_mode="action")
    print(f"Action-level pred shape: {pred_a['normalized_actions'].shape}")

    print("All smoke tests passed.")
