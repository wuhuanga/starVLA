# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
TAG-Flow: Trajectory-Aligned Guided Flow Matching
==================================================
Single-branch VLA framework that replaces BayesianCAG's dual-branch architecture
with a unified Condition Dropout + Trajectory-Action Contrastive Alignment
+ Time-Scheduled CFG design.

Key innovations over BayesianCAG:
  1. **Single VLM Forward**: Uses condition dropout (randomly replacing lang with
     empty tokens during training) instead of running two separate VLM branches.
     This halves compute and memory compared to BayesianCAG.

  2. **Action-Domain Contrastive Alignment**: Aligns trajectory features from the
     DiT (action model) mid-layer with language features via InfoNCE, instead of
     BayesianCAG's hidden-state subtraction approach.

  3. **Time-Scheduled Guidance**: At inference, omega decays from omega_max
     (early denoising) to omega_min (late denoising) via cosine/linear schedule,
     ensuring coarse direction follows instructions while fine motor control
     stays smooth. Replaces BayesianCAG's unstable attention-based token-wise omega.

Training:
  - Single branch with condition dropout: p(drop) = cond_drop_rate
  - Dropped samples: instruction = action_tokens only  => p(a|v)  (prior)
  - Kept samples: instruction = lang + action_tokens    => pi(a|v,l)  (posterior)
  - Trajectory-language contrastive loss (InfoNCE) on kept samples

Inference:
  - Two VLM passes: one with lang (cond), one without (uncond)
  - Time-scheduled CFG on flow-matching velocity field
"""
import sys
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
    TAG-Flow: Trajectory-Aligned Guided Flow Matching.

    Single-branch VLA with condition dropout training + time-scheduled CFG inference.

    Config keys (under ``cfg.framework``):
        cond_drop_rate       (float): Language dropout probability.     Default 0.15.
        contrastive_weight   (float): InfoNCE loss weight.              Default 0.1.
        contrastive_tau      (float): InfoNCE temperature.              Default 0.07.
        contrastive_proj_dim (int):   Projection head output dim.       Default 256.
        omega_max            (float): Max guidance scale (early steps). Default 3.0.
        omega_min            (float): Min guidance scale (late steps).  Default 1.0.
        omega_schedule       (str):   "cosine" or "linear".            Default "cosine".
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

        # ===== Trajectory-Language Contrastive Alignment =====
        self.contrastive_weight = float(self.config.framework.get("contrastive_weight", 0.1))
        self.contrastive_tau = float(self.config.framework.get("contrastive_tau", 0.07))
        contrastive_proj_dim = int(self.config.framework.get("contrastive_proj_dim", 256))

        # Trajectory feature projection head (projects DiT hidden_size -> proj_dim)
        dit_hidden = int(self.config.framework.action_model.hidden_size)
        self.traj_proj = nn.Sequential(
            nn.Linear(dit_hidden, dit_hidden),
            nn.GELU(),
            nn.Linear(dit_hidden, contrastive_proj_dim),
        )

        # Language feature projection head (projects VLM hidden_size -> proj_dim)
        self.lang_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, contrastive_proj_dim),
        )

        # ===== Time-Scheduled Guidance =====
        self.omega_max = float(self.config.framework.get("omega_max", 3.0))
        self.omega_min = float(self.config.framework.get("omega_min", 1.0))
        self.omega_schedule = str(self.config.framework.get("omega_schedule", "cosine"))

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
            f"[TAGFlow] Single-branch with condition dropout={self.cond_drop_rate}, "
            f"contrastive_weight={self.contrastive_weight}, tau={self.contrastive_tau}, "
            f"proj_dim={contrastive_proj_dim}, "
            f"omega_max={self.omega_max}, omega_min={self.omega_min}, "
            f"schedule={self.omega_schedule}, "
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

    def _extract_action_query_hidden_states(
        self,
        hidden_states: torch.Tensor,   # [B, S, H]
        input_ids: torch.Tensor,       # [B, S]
        tokenizer,
    ):
        self._ensure_action_token_ids(tokenizer)

        B = hidden_states.shape[0]
        out = []
        for b in range(B):
            start = self._get_action_block_start(input_ids[b], tokenizer)
            assert start != -1, "No valid contiguous action token block found in the sequence."
            end = start + self.num_latent_action_query
            out.append(hidden_states[b, start:end, :])

        return torch.stack(out, dim=0)  # [B, K, H]

    # ------------------------------------------------------------------
    # Extract language hidden states (V + L + A layout)
    # ------------------------------------------------------------------
    def _find_last_pos(self, seq_1d: torch.Tensor, token_id: int) -> int:
        idx = (seq_1d == int(token_id)).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return -1
        return int(idx[-1].item())

    def _extract_lang_hidden_states(
        self,
        hidden_states: torch.Tensor,   # [B, S, H]
        input_ids: torch.Tensor,       # [B, S]
    ) -> torch.Tensor:
        """
        Extract language token hidden states from posterior layout (V + L + A)
        and mean-pool to [B, H].
        """
        B, S, H = hidden_states.shape
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        lang_embs = []
        for b in range(B):
            ids = input_ids[b]
            a_start = self._get_action_block_start(ids, tokenizer)
            v_end = self._find_last_pos(ids, VISION_END_TOKEN_INDEX)
            if v_end == -1 or a_start == -1:
                lang_embs.append(torch.zeros(H, device=hidden_states.device, dtype=hidden_states.dtype))
                continue
            lang_start = v_end + 1
            lang_end = a_start
            if lang_end <= lang_start:
                lang_embs.append(torch.zeros(H, device=hidden_states.device, dtype=hidden_states.dtype))
                continue
            lang_embs.append(hidden_states[b, lang_start:lang_end, :].mean(dim=0))
        return torch.stack(lang_embs, dim=0)  # [B, H]

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
    # Trajectory-Language contrastive loss (InfoNCE)
    # ------------------------------------------------------------------
    def _contrastive_loss(
        self,
        traj_feat: torch.Tensor,   # [B_cond, D_traj]  (from DiT mid-layer, mean-pooled)
        lang_emb: torch.Tensor,    # [B_cond, H_vlm]   (from VLM language tokens)
    ) -> torch.Tensor:
        """
        InfoNCE between trajectory features (projected from DiT) and
        language features (projected from VLM), on conditioned samples only.
        """
        B = traj_feat.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=traj_feat.device, dtype=traj_feat.dtype)

        z_traj = self.traj_proj(traj_feat.float())
        z_lang = self.lang_proj(lang_emb.float())

        z_traj = F.normalize(z_traj, dim=-1)
        z_lang = F.normalize(z_lang, dim=-1)

        logits = torch.mm(z_traj, z_lang.t()) / self.contrastive_tau
        labels = torch.arange(B, device=logits.device)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------
    # Extract trajectory features from DiT mid-layer
    # ------------------------------------------------------------------
    def _get_trajectory_feature(
        self,
        action_cond: torch.Tensor,   # [B, K, H_vlm]
        actions_target: torch.Tensor, # [B, T, action_dim]
        state: torch.Tensor = None,
        vision_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Run the action model's DiT forward and extract a pooled trajectory feature
        from the model output (before decoding to action dim).
        Returns: [B, hidden_size]
        """
        am = self.action_model

        noise = torch.randn(actions_target.shape, device=actions_target.device, dtype=actions_target.dtype)
        t = am.sample_time(actions_target.shape[0], device=actions_target.device, dtype=actions_target.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions_target
        t_discretized = (t[:, 0, 0] * am.num_timestep_buckets).long()
        action_features = am.action_encoder(noisy_trajectory, t_discretized)

        state_features = am.state_encoder(state) if state is not None else None

        if am.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=action_features.device)
            pos_embs = am.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = am.future_tokens.weight.unsqueeze(0).expand(action_cond.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)

        model_output = am.model(
            hidden_states=sa_embs,
            encoder_hidden_states=action_cond,
            timestep=t_discretized,
            return_all_hidden_states=False,
            vision_features=vision_features,
        )

        # Mean-pool over sequence to get trajectory feature [B, output_dim]
        traj_feat = model_output.mean(dim=1)
        return traj_feat

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        batch_images = [example["image"] for example in examples]
        batch_langs = [example["lang"] for example in examples]

        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        B = len(examples)

        # ===== Condition Dropout: single-branch design =====
        # With probability cond_drop_rate, drop the language instruction
        # so the model sees only action query tokens (unconditional / prior).
        drop_mask = [random.random() < self.cond_drop_rate for _ in range(B)]
        instructions = []
        for i in range(B):
            if drop_mask[i]:
                # Unconditional: action tokens only (no language)
                instructions.append(self.latent_action_query)
            else:
                # Conditional: lang + action tokens (posterior layout V + L + A)
                instructions.append(batch_langs[i] + self.latent_action_query)

        # ===== Single VLM Forward Pass =====
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

        # ===== Action head: flow matching loss =====
        with torch.autocast("cuda", dtype=torch.float32):
            actions_t = torch.tensor(
                np.array(actions), device=action_hidden.device, dtype=action_hidden.dtype
            )
            actions_target = actions_t[:, -(self.future_action_window_size + 1):, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
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

        # ===== Trajectory-Language Contrastive Alignment (on conditioned samples only) =====
        cond_indices = [i for i in range(B) if not drop_mask[i]]
        contra_loss = torch.tensor(0.0, device=action_hidden.device, dtype=torch.float32)

        if len(cond_indices) >= 2 and self.contrastive_weight > 0:
            cond_idx_t = torch.tensor(cond_indices, device=action_hidden.device, dtype=torch.long)
            cond_action_hidden = action_hidden[cond_idx_t].float()  # [B_cond, K, H]
            cond_actions_target = actions_target[cond_idx_t]
            cond_state = state_tensor[cond_idx_t] if state_tensor is not None else None
            cond_vis = None
            if self.enable_vision_cross_attn and vis_feats_repeated is not None:
                cond_vis = self._extract_vision_features(qwen_inputs)
                if cond_vis is not None:
                    cond_vis = cond_vis[cond_idx_t].float()

            with torch.autocast("cuda", dtype=torch.float32):
                # Get trajectory features from DiT mid-layer
                traj_feat = self._get_trajectory_feature(
                    cond_action_hidden, cond_actions_target, cond_state, cond_vis,
                )

                # Get language features from VLM hidden states
                lang_emb = self._extract_lang_hidden_states(
                    last_hidden[cond_idx_t],
                    qwen_inputs["input_ids"][cond_idx_t],
                )

                contra_loss = self._contrastive_loss(traj_feat, lang_emb)

        # ===== Total loss =====
        total_loss = flow_loss + self.contrastive_weight * contra_loss

        return {
            "action_loss": total_loss,
            "flow_loss": flow_loss.detach(),
            "contra_loss": contra_loss.detach(),
        }

    # ------------------------------------------------------------------
    # Inference: Time-Scheduled CFG
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

        mode = guidance_mode if guidance_mode is not None else "scheduled"

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

        # ===== Conditioned pass: V + L + A =====
        instructions_cond = [ex["lang"] + self.latent_action_query for ex in examples]
        qwen_inputs_cond = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions_cond,
        )
        tokenizer = self.qwen_vl_interface.processor.tokenizer

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs_cond = self.qwen_vl_interface(
                **qwen_inputs_cond,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_cond = outputs_cond.hidden_states[-1]
            action_hidden_cond = self._extract_action_query_hidden_states(
                hidden_cond, qwen_inputs_cond["input_ids"], tokenizer,
            )

        # ===== Unconditioned pass: A only (no language) =====
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

        # ===== State tensor =====
        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(
                action_hidden_cond.device, dtype=action_hidden_cond.dtype
            )

        # ===== Vision features =====
        vis_feats = None
        if self.enable_vision_cross_attn:
            vis_feats = self._extract_vision_features(qwen_inputs_cond)

        # ===== Action prediction with time-scheduled guidance =====
        with torch.autocast("cuda", dtype=torch.float32):
            vis_feats_f = vis_feats.float() if vis_feats is not None else None

            if mode == "scheduled":
                # Time-scheduled CFG (core TAG-Flow innovation)
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
                # Uniform omega on each denoising step (BayesianCAG-compatible)
                pred_actions = self.action_model.predict_action_guided(
                    vl_embs_cond=action_hidden_cond.float(),
                    vl_embs_uncond=action_hidden_uncond.float(),
                    state=state_tensor,
                    omega=o_max,
                    vision_features=vis_feats_f,
                )
            elif mode == "action":
                # Uniform omega on final actions (BayesianCAG-compatible)
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

    # Test training forward
    out = model(batch)
    print(f"Action Loss: {out['action_loss'].item()}, Flow Loss: {out['flow_loss'].item()}, "
          f"Contra Loss: {out['contra_loss'].item()}")

    # Test scheduled guidance (default)
    pred = model.predict_action([sample], omega_max=3.0, omega_min=1.0)
    print(f"Scheduled guidance pred shape: {pred['normalized_actions'].shape}")

    # Test velocity-level guidance (BayesianCAG-compatible)
    pred_v = model.predict_action([sample], omega=2.0, guidance_mode="velocity")
    print(f"Velocity-level pred shape: {pred_v['normalized_actions'].shape}")

    # Test action-level guidance
    pred_a = model.predict_action([sample], omega=2.0, guidance_mode="action")
    print(f"Action-level pred shape: {pred_a['normalized_actions'].shape}")

    print("All smoke tests passed.")
