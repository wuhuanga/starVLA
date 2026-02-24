# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
BayesianCAG Framework
=====================
Combines **BayesianVLA dual-branch decomposition** (LangForce) with
**Counterfactual Action Guidance (CAG)** at inference time.

Training:
  Identical to LangForce – Priori (V+A+L) and Posteriori (V+L+A) branches
  with LLR regularisation, action flow-matching losses, etc.

Inference (the key contribution):
  1. Run *both* branches through the VLM to obtain:
       - a_cond   (posterior: action queries attend to language)
       - a_uncond (prior:    action queries masked from language)
  2. Merge via CAG formula:
       a_final = a_uncond + omega * (a_cond - a_uncond)

  Two guidance modes are supported:
    * "action"   – apply formula on final predicted actions  (default, fast)
    * "velocity" – apply formula at every flow-matching denoising step (more principled)
"""
import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List, Optional
import torch
import numpy as np

from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.framework.LangForce import LangForce

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("BayesianCAG")
class BayesianCAG(LangForce):
    """
    BayesianCAG = LangForce training + CAG-guided inference.

    Extra config keys (all under ``cfg.framework``):
        guidance_omega  (float): Guidance scale. Default 2.0.
                                 1.0 = standard posterior-only inference.
        guidance_mode   (str):   "action" or "velocity". Default "action".
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__(config=config, **kwargs)

        # CAG hyper-parameters
        self.guidance_omega = float(self.config.framework.get("guidance_omega", 2.0))
        self.guidance_mode = str(self.config.framework.get("guidance_mode", "action"))
        assert self.guidance_mode in ("action", "velocity"), (
            f"guidance_mode must be 'action' or 'velocity', got '{self.guidance_mode}'"
        )
        logger.info(
            f"[BayesianCAG] omega={self.guidance_omega}, mode={self.guidance_mode}"
        )

    # ------------------------------------------------------------------
    # forward() is inherited from LangForce – training is unchanged
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # CAG-guided inference
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
        Dual-path prediction with Counterfactual Action Guidance.

        Args:
            examples: List of dicts with keys ``image``, ``lang``, and
                      optionally ``state``.
            omega:    Override guidance scale (uses config default if None).
            guidance_mode: Override guidance mode (uses config default if None).

        Returns:
            dict with ``normalized_actions`` np.ndarray [B, T, action_dim].
        """
        if not isinstance(examples, list):
            examples = [examples]

        omega = omega if omega is not None else self.guidance_omega
        mode = guidance_mode if guidance_mode is not None else self.guidance_mode

        # --- image pre-processing (shared across both branches) ---
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

        # --- Construct instructions for both branches ---
        instructions_posteriori = [ex["lang"] + self.latent_action_query for ex in examples]  # L + A
        instructions_priori = [self.latent_action_query + ex["lang"] for ex in examples]       # A + L

        tokenizer = self.qwen_vl_interface.processor.tokenizer

        # ============================================================
        # Branch 1: Posteriori  (V + L + A)  -->  a_cond
        # ============================================================
        qwen_inputs_post = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions_posteriori,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs_post = self.qwen_vl_interface(
                **qwen_inputs_post,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_post = outputs_post.hidden_states[-1]  # [B, S, H]
            action_hidden_post = self._extract_action_query_hidden_states(
                hidden_post,
                qwen_inputs_post["input_ids"],
                tokenizer,
                return_starts=False,
            )  # [B, K, H]

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
        # State tensor (shared)
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
            if mode == "velocity":
                # --- Velocity-level guidance (guidance at each denoising step) ---
                pred_actions = self.action_model.predict_action_guided(
                    vl_embs_cond=action_hidden_post,
                    vl_embs_uncond=action_hidden_prior,
                    state=state_tensor,
                    omega=omega,
                )
            else:
                # --- Action-level guidance (default) ---
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
        "lang": "Pick the red block and place it on the blue plate.",
    }

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Test training forward
    out = model(batch)
    print(f"Action Loss: {out['action_loss'].item()}, KL Loss: {out['kl_loss'].item()}")

    # Test action-level guidance inference
    pred = model.predict_action([sample], omega=2.0, guidance_mode="action")
    print(f"Action-level pred shape: {pred['normalized_actions'].shape}")

    # Test velocity-level guidance inference
    pred_v = model.predict_action([sample], omega=2.0, guidance_mode="velocity")
    print(f"Velocity-level pred shape: {pred_v['normalized_actions'].shape}")

    # Test with omega=1.0 (should match standard posterior)
    pred_std = model.predict_action([sample], omega=1.0, guidance_mode="action")
    print(f"Standard (omega=1.0) pred shape: {pred_std['normalized_actions'].shape}")

    print("All smoke tests passed.")
