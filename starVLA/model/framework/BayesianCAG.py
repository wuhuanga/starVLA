# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
BayesianCAG Framework  (v2 – Latent Contrastive + Dynamic CAG)
==============================================================
Training enhancements:
  Latent Contrastive Alignment (InfoNCE) – forces the feature delta
  (h_post − h_prior) to align with the language embedding direction.

Inference enhancements:
  Token-wise Dynamic CAG – each action query token gets its own omega_k
  derived from how much it attends to the language tokens.

Guidance modes:
  * "latent"   – token-wise dynamic omega on latent features  (new, recommended)
  * "action"   – uniform omega on final predicted actions      (original CAG)
  * "velocity" – uniform omega on each flow-matching step      (principled but slower)
"""
import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List, Optional
import torch
import torch.nn.functional as F
import numpy as np

from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.framework.LangForce import LangForce, VISION_END_TOKEN_INDEX

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("BayesianCAG")
class BayesianCAG(LangForce):
    """
    BayesianCAG = LangForce + Latent Contrastive Alignment (train)
                            + Token-wise Dynamic CAG       (inference).

    Extra config keys (all under ``cfg.framework``):
        guidance_omega      (float): Max guidance scale.         Default 2.0.
        guidance_mode       (str):   "latent"|"action"|"velocity". Default "latent".
        contrastive_weight  (float): InfoNCE loss weight.        Default 0.1.
        contrastive_tau     (float): InfoNCE temperature.        Default 0.07.
        omega_base          (float): Min per-token omega.        Default 1.0.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__(config=config, **kwargs)

        # CAG hyper-parameters
        self.guidance_omega = float(self.config.framework.get("guidance_omega", 2.0))
        self.guidance_mode = str(self.config.framework.get("guidance_mode", "latent"))
        assert self.guidance_mode in ("action", "velocity", "latent"), (
            f"guidance_mode must be 'action', 'velocity' or 'latent', got '{self.guidance_mode}'"
        )

        # Latent Contrastive Alignment
        self.contrastive_weight = float(self.config.framework.get("contrastive_weight", 0.1))
        self.contrastive_tau = float(self.config.framework.get("contrastive_tau", 0.07))

        # Token-wise dynamic omega range
        self.omega_base = float(self.config.framework.get("omega_base", 1.0))

        logger.info(
            f"[BayesianCAG] omega={self.guidance_omega}, mode={self.guidance_mode}, "
            f"contrastive_weight={self.contrastive_weight}, tau={self.contrastive_tau}, "
            f"omega_base={self.omega_base}"
        )

    # ------------------------------------------------------------------
    # Helper: extract language hidden states from posteriori branch
    # ------------------------------------------------------------------
    def _extract_lang_hidden_states(
        self,
        hidden_states: torch.Tensor,   # [B, S, H]
        input_ids: torch.Tensor,       # [B, S]
        action_starts: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        """
        Extract language token hidden states from the *posteriori* branch
        (layout: V + L + A) and mean-pool to [B, H].

        Language span in posteriori: [last(vision_end)+1 : action_start)
        """
        B, S, H = hidden_states.shape
        lang_embs = []
        for b in range(B):
            ids = input_ids[b]
            a_start = int(action_starts[b].item())
            v_end = self._find_last_pos(ids, VISION_END_TOKEN_INDEX)
            if v_end == -1:
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
    # InfoNCE contrastive loss
    # ------------------------------------------------------------------
    def _contrastive_loss(
        self,
        delta_h: torch.Tensor,   # [B, H]
        lang_emb: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:
        """
        InfoNCE: (delta_h_i, lang_emb_i) = positive pair,
        all other lang_emb_j (j != i) = negatives.
        """
        B = delta_h.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=delta_h.device, dtype=delta_h.dtype)

        delta_h_norm = F.normalize(delta_h.float(), dim=-1)
        lang_norm = F.normalize(lang_emb.float(), dim=-1)

        # [B, B] similarity matrix
        logits = torch.mm(delta_h_norm, lang_norm.t()) / self.contrastive_tau
        labels = torch.arange(B, device=logits.device)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------
    # Training forward: LangForce losses + InfoNCE contrastive
    # ------------------------------------------------------------------
    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions_priori = [self.latent_action_query + example["lang"] for example in examples]
        instructions_posteriori = [example["lang"] + self.latent_action_query for example in examples]

        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        # ===== Priori Branch (V + A + L) =====
        qwen_inputs_priori = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions_priori,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs_priori = self.qwen_vl_interface(
                **qwen_inputs_priori,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            priori_last_hidden = qwenvl_outputs_priori.hidden_states[-1]
            priori_action_hidden, priori_action_starts = self._extract_action_query_hidden_states(
                priori_last_hidden,
                qwen_inputs_priori["input_ids"],
                self.qwen_vl_interface.processor.tokenizer,
                return_starts=True,
            )
            priori_logits = qwenvl_outputs_priori.logits

        # ===== Posteriori Branch (V + L + A) =====
        qwen_inputs_posteriori = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions_posteriori,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs_posteriori = self.qwen_vl_interface(
                **qwen_inputs_posteriori,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            posteriori_last_hidden = qwenvl_outputs_posteriori.hidden_states[-1]
            posteriori_action_hidden, posteriori_action_starts = self._extract_action_query_hidden_states(
                posteriori_last_hidden,
                qwen_inputs_posteriori["input_ids"],
                self.qwen_vl_interface.processor.tokenizer,
                return_starts=True,
            )
            posteriori_logits = qwenvl_outputs_posteriori.logits.detach()

        # ===== LLR loss (from LangForce) =====
        kl_loss = self._compute_language_llr_from_boundaries(
            priori_logits=priori_logits,
            posteriori_logits=posteriori_logits,
            priori_input_ids=qwen_inputs_priori["input_ids"],
            posteriori_input_ids=qwen_inputs_posteriori["input_ids"],
            priori_action_starts=priori_action_starts,
            posteriori_action_starts=posteriori_action_starts,
        )

        # ===== Latent Contrastive Alignment (InfoNCE) =====
        # delta_h = h_post - h_prior, mean over K queries -> [B, H]
        delta_h = (posteriori_action_hidden - priori_action_hidden).mean(dim=1)
        # lang embedding: mean-pool language tokens from posteriori hidden
        lang_emb = self._extract_lang_hidden_states(
            posteriori_last_hidden,
            qwen_inputs_posteriori["input_ids"],
            posteriori_action_starts,
        )
        contra_loss = self._contrastive_loss(delta_h, lang_emb)

        # ===== Action head losses =====
        with torch.autocast("cuda", dtype=torch.float32):
            actions_t = torch.tensor(
                np.array(actions), device=priori_action_hidden.device, dtype=priori_action_hidden.dtype
            )
            actions_target = actions_t[:, -(self.future_action_window_size + 1):, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )

            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state), device=priori_action_hidden.device, dtype=priori_action_hidden.dtype
                )

            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)

            if self.detach_prior_cond:
                priori_cond_base = priori_action_hidden.detach()
            else:
                priori_cond_base = priori_action_hidden

            priori_cond = priori_cond_base.repeat(repeated_diffusion_steps, 1, 1).float()
            posteriori_cond = posteriori_action_hidden.repeat(repeated_diffusion_steps, 1, 1).float()
            state_repeated = state_tensor.repeat(repeated_diffusion_steps, 1, 1) if state_tensor is not None else None

            prior_loss = self.action_model(priori_cond, actions_target_repeated, state_repeated)
            main_loss = self.action_model(posteriori_cond, actions_target_repeated, state_repeated)

        # ===== Total loss =====
        total_loss = (
            (1.0 - self.prior_loss_weight) * main_loss
            + self.prior_loss_weight * prior_loss
            - self.kl_weight * kl_loss
            + self.contrastive_weight * contra_loss
        )

        return {
            "action_loss": total_loss,
            "main_loss": main_loss.detach(),
            "prior_loss": prior_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "contra_loss": contra_loss.detach(),
        }

    # ------------------------------------------------------------------
    # Token-wise dynamic omega from attention weights
    # ------------------------------------------------------------------
    def _compute_token_omega(
        self,
        attentions: tuple,           # tuple of per-layer [B, num_heads, S, S]
        input_ids: torch.Tensor,     # [B, S]
        action_starts: torch.Tensor, # [B]
        omega_max: float,
    ) -> torch.Tensor:
        """
        Per-action-query omega from last-layer attention over language tokens.

        For Q_k, sum attention over language positions, min-max normalise,
        linearly map to [omega_base, omega_max].

        Returns: [B, K, 1]
        """
        attn_last = attentions[-1]           # [B, num_heads, S, S]
        attn_avg = attn_last.mean(dim=1)     # [B, S, S]

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

            s_min = s_k.min()
            s_max = s_k.max()
            if s_max - s_min < 1e-8:
                s_norm = torch.ones_like(s_k)
            else:
                s_norm = (s_k - s_min) / (s_max - s_min)

            omega_k = self.omega_base + (omega_max - self.omega_base) * s_norm
            omega_vecs.append(omega_k)

        return torch.stack(omega_vecs, dim=0).unsqueeze(-1)  # [B, K, 1]

    # ------------------------------------------------------------------
    # CAG-guided inference (v2: latent + token-wise dynamic omega)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        omega: Optional[float] = None,
        guidance_mode: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Dual-path prediction with Token-wise Dynamic CAG.

        guidance_mode:
          "latent"   – token-wise dynamic omega on latent features (recommended)
          "action"   – uniform omega on final actions
          "velocity" – uniform omega on each denoising velocity step
        """
        if not isinstance(examples, list):
            examples = [examples]

        omega = omega if omega is not None else self.guidance_omega
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

        instructions_posteriori = [ex["lang"] + self.latent_action_query for ex in examples]
        instructions_priori = [self.latent_action_query + ex["lang"] for ex in examples]

        tokenizer = self.qwen_vl_interface.processor.tokenizer

        # ============================================================
        # Branch 1: Posteriori  (V + L + A)  -->  a_cond
        # ============================================================
        qwen_inputs_post = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions_posteriori,
        )
        need_attn = (mode == "latent")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs_post = self.qwen_vl_interface(
                **qwen_inputs_post,
                output_attentions=need_attn,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_post = outputs_post.hidden_states[-1]
            action_hidden_post, post_action_starts = self._extract_action_query_hidden_states(
                hidden_post,
                qwen_inputs_post["input_ids"],
                tokenizer,
                return_starts=True,
            )  # [B, K, H], [B]

            if need_attn:
                omega_vec = self._compute_token_omega(
                    outputs_post.attentions,
                    qwen_inputs_post["input_ids"],
                    post_action_starts,
                    omega_max=omega,
                )  # [B, K, 1]
                omega_vec = omega_vec.to(dtype=action_hidden_post.dtype,
                                        device=action_hidden_post.device)

        # ============================================================
        # Branch 2: Priori  (V + A + L)  -->  a_uncond
        # ============================================================
        qwen_inputs_prior = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions_priori,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs_prior = self.qwen_vl_interface(
                **qwen_inputs_prior,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_prior = outputs_prior.hidden_states[-1]
            action_hidden_prior = self._extract_action_query_hidden_states(
                hidden_prior,
                qwen_inputs_prior["input_ids"],
                tokenizer,
                return_starts=False,
            )  # [B, K, H]

        # ============================================================
        # State tensor
        # ============================================================
        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(
                action_hidden_post.device, dtype=action_hidden_post.dtype
            )

        # ============================================================
        # Action prediction with CAG guidance
        # ============================================================
        with torch.autocast("cuda", dtype=torch.float32):
            if mode == "latent":
                # --- Token-wise dynamic omega on latent features ---
                # h_final = h_prior + omega_vec * (h_post - h_prior)
                action_hidden_guided = (
                    action_hidden_prior + omega_vec * (action_hidden_post - action_hidden_prior)
                )
                pred_actions = self.action_model.predict_action(
                    action_hidden_guided.float(), state_tensor
                )
            elif mode == "velocity":
                pred_actions = self.action_model.predict_action_guided(
                    vl_embs_cond=action_hidden_post,
                    vl_embs_uncond=action_hidden_prior,
                    state=state_tensor,
                    omega=omega,
                )
            else:
                # --- Action-level guidance (uniform omega) ---
                a_cond = self.action_model.predict_action(action_hidden_post, state_tensor)
                a_uncond = self.action_model.predict_action(action_hidden_prior, state_tensor)
                pred_actions = a_uncond + omega * (a_cond - a_uncond)

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
        default="examples/LIBERO/train_files/starvla_cotrain_libero_bayesian_cag.yaml",
    )
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    model: BayesianCAG = BayesianCAG(cfg)
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

    # Test training forward (now includes contra_loss)
    out = model(batch)
    print(f"Action Loss: {out['action_loss'].item()}, KL Loss: {out['kl_loss'].item()}, "
          f"Contra Loss: {out['contra_loss'].item()}")

    # Test latent-level guidance (token-wise dynamic omega)
    pred = model.predict_action([sample], omega=2.0, guidance_mode="latent")
    print(f"Latent-level pred shape: {pred['normalized_actions'].shape}")

    # Test action-level guidance
    pred_a = model.predict_action([sample], omega=2.0, guidance_mode="action")
    print(f"Action-level pred shape: {pred_a['normalized_actions'].shape}")

    # Test velocity-level guidance
    pred_v = model.predict_action([sample], omega=2.0, guidance_mode="velocity")
    print(f"Velocity-level pred shape: {pred_v['normalized_actions'].shape}")

    print("All smoke tests passed.")
