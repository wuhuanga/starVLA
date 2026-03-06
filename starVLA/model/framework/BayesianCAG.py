# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
BayesianCAG Framework  (v2 – Latent Contrastive + Dynamic CAG)
==============================================================
Standalone dual-branch VLA framework (no LangForce inheritance).

Training:
  - Prior branch: (V + A + L) => proposal p(a|v)
  - Posterior branch: (V + L + A) => pi(a|v,l)
  - LLR regularizer with hard-token LLR + shortcut gate
  - Latent Contrastive Alignment (InfoNCE)

Inference:
  - Token-wise Dynamic CAG with per-action-query omega

Guidance modes:
  * "latent"   – token-wise dynamic omega on latent features  (recommended)
  * "action"   – uniform omega on final predicted actions
  * "velocity" – uniform omega on each flow-matching step
"""
import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List, Optional, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
IM_START_TOKEN_INDEX     = 151644  # <|im_start|>
IM_END_TOKEN_INDEX       = 151645  # <|im_end|>


@FRAMEWORK_REGISTRY.register("BayesianCAG")
class BayesianCAG(baseframework):
    """
    BayesianCAG: Dual-branch VLA with Latent Contrastive Alignment (train)
                 + Token-wise Dynamic CAG (inference).

    Config keys (under ``cfg.framework``):
        guidance_omega      (float): Max guidance scale.         Default 2.0.
        guidance_mode       (str):   "latent"|"action"|"velocity". Default "latent".
        contrastive_weight  (float): InfoNCE loss weight.        Default 0.1.
        contrastive_tau     (float): InfoNCE temperature.        Default 0.07.
        omega_base          (float): Min per-token omega.        Default 1.0.
        kl_weight           (float): LLR loss weight.            Default 0.1.
        prior_loss_weight   (float): Prior action loss weight.   Default 0.3.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # align dims
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        # Action tokens
        self.num_latent_action_query = self.config.framework.qwenvl.get("num_latent_action_query", 32)
        self.latent_action_query = "".join([f"<|action_{i}|>" for i in range(self.num_latent_action_query)])
        self.action_token_ids = None  # cached {'first','last'}

        # === Runtime action token registration ===
        self._register_action_tokens()

        # Action model
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # ===== Loss weights =====
        self.kl_weight = float(self.config.framework.get("kl_weight", 0.1))
        self.prior_loss_weight = float(self.config.framework.get("prior_loss_weight", 0.3))

        # ===== Training assert switch =====
        self.assert_lang_span_match = bool(self.config.framework.get("assert_lang_span_match", True))

        # ===== Detach prior cond switch =====
        self.detach_prior_cond = bool(self.config.framework.get("detach_prior_cond", True))

        # ===== Hard-token LLR =====
        self.use_hard_token_llr = bool(self.config.framework.get("use_hard_token_llr", True))
        self.hard_token_k = int(self.config.framework.get("hard_token_k", 16))
        assert self.hard_token_k > 0

        # ===== Shortcut gate =====
        self.use_kl_gate = bool(self.config.framework.get("use_kl_gate", True))
        self.kl_gate_momentum = float(self.config.framework.get("kl_gate_momentum", 0.99))
        self.kl_gate_temp = float(self.config.framework.get("kl_gate_temp", 0.5))
        self.kl_gate_tau_scale = float(self.config.framework.get("kl_gate_tau_scale", 0.7))
        self.kl_gate_min = float(self.config.framework.get("kl_gate_min", 0.0))
        self.kl_gate_max = float(self.config.framework.get("kl_gate_max", 1.0))

        # cache im_end token id
        self._im_end_id = None

        # EMA buffer for posterior language-span NLL
        self.register_buffer("post_nll_ema", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("post_nll_ema_inited", torch.tensor(0, dtype=torch.uint8))

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

        # Vision cross-attention injection in DiT
        self.enable_vision_cross_attn = bool(
            self.config.framework.action_model.get("enable_vision_cross_attn", False)
        )

        logger.info(
            f"[BayesianCAG] omega={self.guidance_omega}, mode={self.guidance_mode}, "
            f"contrastive_weight={self.contrastive_weight}, tau={self.contrastive_tau}, "
            f"omega_base={self.omega_base}, vision_cross_attn={self.enable_vision_cross_attn}"
        )

    # ------------------------------------------------------------------
    # Runtime action token registration
    # ------------------------------------------------------------------
    def _register_action_tokens(self):
        """
        Ensure action tokens (<|action_0|>, ..., <|action_N|>) exist in the
        tokenizer. If missing, add them and resize model embeddings.
        """
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        action_tokens = [f"<|action_{i}|>" for i in range(self.num_latent_action_query)]

        vocab = tokenizer.get_vocab()
        to_add = [t for t in action_tokens if t not in vocab]

        if to_add:
            old_embed = self.qwen_vl_interface.model.get_input_embeddings()
            old_size = old_embed.weight.shape[0]

            tokenizer.add_special_tokens({"additional_special_tokens": to_add})
            new_size = old_size + len(to_add)
            self.qwen_vl_interface.model.resize_token_embeddings(new_size)

            new_embed = self.qwen_vl_interface.model.get_input_embeddings()
            with torch.no_grad():
                ref_vec = old_embed.weight.mean(dim=0)
                for idx in range(old_size, new_size):
                    new_embed.weight[idx].copy_(ref_vec)

            logger.info(
                f"[BayesianCAG] Added {len(to_add)} action tokens to tokenizer, "
                f"resized embeddings {old_size} -> {new_size}"
            )
        else:
            logger.info("[BayesianCAG] All action tokens already present in tokenizer")

    # ------------------------------------------------------------------
    # Token id helpers (from LangForce)
    # ------------------------------------------------------------------
    def _ensure_action_token_ids(self, tokenizer):
        if self.action_token_ids is None:
            self.action_token_ids = {
                "first": tokenizer.convert_tokens_to_ids("<|action_0|>"),
                "last": tokenizer.convert_tokens_to_ids(f"<|action_{self.num_latent_action_query-1}|>"),
            }

    def _ensure_im_end_id(self, tokenizer):
        if self._im_end_id is None:
            self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _find_last_pos(self, seq_1d: torch.Tensor, token_id: int) -> int:
        idx = (seq_1d == int(token_id)).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return -1
        return int(idx[-1].item())

    def _find_first_pos_after(self, seq_1d: torch.Tensor, token_id: int, start: int) -> int:
        if start < 0:
            start = 0
        sub = seq_1d[start:]
        idx = (sub == int(token_id)).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return -1
        return int(start + idx[0].item())

    # ------------------------------------------------------------------
    # Action block helpers (from LangForce)
    # ------------------------------------------------------------------
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
    # SHIFT-correct token-level NLL span (from LangForce)
    # ------------------------------------------------------------------
    def _token_nll_span(
        self,
        logits_1d: torch.Tensor,      # [S, V]
        input_ids_1d: torch.Tensor,   # [S]
        start: int,
        end: int,
        ignore_ids: Optional[Set[int]] = None,
    ):
        if end <= start:
            return None, None
        S = int(input_ids_1d.shape[0])
        start = max(0, int(start))
        end = min(S, int(end))
        if end <= start:
            return None, None

        j = torch.arange(start, end, device=input_ids_1d.device, dtype=torch.long)
        j = j[j > 0]
        if j.numel() == 0:
            return None, None

        targets = input_ids_1d[j].long()

        if ignore_ids is not None and len(ignore_ids) > 0:
            keep = torch.ones_like(targets, dtype=torch.bool)
            for tid in ignore_ids:
                keep &= (targets != int(tid))
            j = j[keep]
            if j.numel() == 0:
                return None, None
            targets = input_ids_1d[j].long()

        pred_pos = j - 1
        pred_logits = logits_1d[pred_pos].float()  # [T, V]
        nll = F.cross_entropy(pred_logits, targets, reduction="none")  # [T]
        return nll, targets

    # ------------------------------------------------------------------
    # LLR with hard-token + shortcut gate (from LangForce)
    # ------------------------------------------------------------------
    def _compute_language_llr_from_boundaries(
        self,
        priori_logits: torch.Tensor,            # [B, S, V]
        posteriori_logits: torch.Tensor,        # [B, S, V] (detached)
        priori_input_ids: torch.Tensor,         # [B, S]
        posteriori_input_ids: torch.Tensor,     # [B, S]
        priori_action_starts: torch.Tensor,     # [B]
        posteriori_action_starts: torch.Tensor, # [B]
    ) -> torch.Tensor:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        self._ensure_im_end_id(tokenizer)

        pad_id = tokenizer.pad_token_id
        ignore_ids: Set[int] = set()
        if pad_id is not None:
            ignore_ids.add(int(pad_id))
        ignore_ids.add(int(IMAGE_TOKEN_INDEX))
        ignore_ids.add(int(VIDEO_TOKEN_INDEX))
        ignore_ids.add(int(VISION_START_TOKEN_INDEX))
        ignore_ids.add(int(VISION_END_TOKEN_INDEX))
        ignore_ids.add(int(IM_START_TOKEN_INDEX))
        ignore_ids.add(int(IM_END_TOKEN_INDEX))

        B = int(priori_input_ids.shape[0])
        K = self.num_latent_action_query

        llr_vals = []
        post_nll_means = []

        for b in range(B):
            ids_prior = priori_input_ids[b]
            ids_post  = posteriori_input_ids[b]

            a_start_prior = int(priori_action_starts[b].item())
            a_start_post  = int(posteriori_action_starts[b].item())

            # prior language span: [action_end : im_end)
            lang_start_prior = a_start_prior + K
            if lang_start_prior >= ids_prior.shape[0]:
                continue
            im_end = self._find_first_pos_after(ids_prior, self._im_end_id, lang_start_prior)
            lang_end_prior = im_end if im_end != -1 else int(ids_prior.shape[0])
            if lang_end_prior <= lang_start_prior:
                continue

            # post language span: [last(vision_end)+1 : action_start)
            v_end_post = self._find_last_pos(ids_post, VISION_END_TOKEN_INDEX)
            if v_end_post == -1:
                continue
            lang_start_post = v_end_post + 1
            lang_end_post = a_start_post
            if lang_end_post <= lang_start_post:
                continue

            # strict assertion: token-level equality
            if self.training and self.assert_lang_span_match:
                prior_span_ids = ids_prior[lang_start_prior:lang_end_prior]
                post_span_ids  = ids_post[lang_start_post:lang_end_post]

                if (prior_span_ids.numel() != post_span_ids.numel()) or (not torch.equal(prior_span_ids, post_span_ids)):
                    prior_text = tokenizer.decode(prior_span_ids.tolist())
                    post_text  = tokenizer.decode(post_span_ids.tolist())

                    raise AssertionError(
                        "\n[BayesianCAG] Language span mismatch detected!\n"
                        f"Sample b={b}\n"
                        f"PRIOR span idx: [{lang_start_prior}:{lang_end_prior}]  (len={prior_span_ids.numel()})\n"
                        f"POST  span idx: [{lang_start_post}:{lang_end_post}]  (len={post_span_ids.numel()})\n"
                        f"PRIOR span: {repr(prior_text)}\n"
                        f"POST  span: {repr(post_text)}\n"
                        f"PRIOR token ids (first 50): {prior_span_ids[:50].tolist()}\n"
                        f"POST  token ids (first 50): {post_span_ids[:50].tolist()}\n"
                        "This indicates your boundary-based language extraction is inconsistent (likely prompt/template issue)."
                    )

            # hard-token LLR
            nll_prior, tok_prior = self._token_nll_span(
                logits_1d=priori_logits[b],
                input_ids_1d=ids_prior,
                start=lang_start_prior,
                end=lang_end_prior,
                ignore_ids=ignore_ids,
            )
            nll_post, tok_post = self._token_nll_span(
                logits_1d=posteriori_logits[b],
                input_ids_1d=ids_post,
                start=lang_start_post,
                end=lang_end_post,
                ignore_ids=ignore_ids,
            )
            if nll_prior is None or nll_post is None:
                continue

            post_nll_mean = nll_post.mean().detach()
            post_nll_means.append(post_nll_mean)

            if self.use_hard_token_llr:
                if tok_prior is None or tok_post is None or tok_prior.shape != tok_post.shape or (not torch.equal(tok_prior, tok_post)):
                    llr = (nll_post.mean() - nll_prior.mean())
                else:
                    k = min(self.hard_token_k, int(nll_post.numel()))
                    if k <= 0:
                        continue
                    idx = torch.topk(nll_post.detach(), k=k, largest=True).indices
                    llr = (nll_post[idx] - nll_prior[idx]).mean()
            else:
                llr = (nll_post.mean() - nll_prior.mean())

            llr_vals.append(llr)

        if len(llr_vals) == 0:
            return torch.tensor(0.0, device=priori_logits.device, dtype=torch.float32)

        llr_vals_t = torch.stack(llr_vals).float()
        post_nll_means_t = torch.stack(post_nll_means).float()

        # shortcut gate: update EMA threshold
        if self.use_kl_gate and self.training:
            batch_mean = post_nll_means_t.mean().detach()
            with torch.no_grad():
                if int(self.post_nll_ema_inited.item()) == 0:
                    self.post_nll_ema.copy_(batch_mean)
                    self.post_nll_ema_inited.fill_(1)
                else:
                    m = self.kl_gate_momentum
                    self.post_nll_ema.copy_(m * self.post_nll_ema + (1.0 - m) * batch_mean)

        # gate computation
        if self.use_kl_gate:
            tau = (self.post_nll_ema.detach() * float(self.kl_gate_tau_scale))
            temp = max(float(self.kl_gate_temp), 1e-6)
            g = torch.sigmoid((tau - post_nll_means_t) / temp)
            if self.kl_gate_min != 0.0 or self.kl_gate_max != 1.0:
                g = float(self.kl_gate_min) + (float(self.kl_gate_max) - float(self.kl_gate_min)) * g
        else:
            g = torch.ones_like(post_nll_means_t)

        return (g * llr_vals_t).mean()

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
    # Helper: extract raw ViT features from Qwen2.5-VL visual encoder
    # ------------------------------------------------------------------
    def _extract_vision_features(
        self,
        qwen_inputs: dict,
    ) -> torch.Tensor:
        """
        Extract spatial vision features directly from Qwen2.5-VL's ViT,
        bypassing the LLM.

        Returns:
            [B, N_vis_max, D_vis]  -- pure ViT spatial features.
        """
        pixel_values = qwen_inputs.get("pixel_values", None)
        image_grid_thw = qwen_inputs.get("image_grid_thw", None)
        if pixel_values is None or image_grid_thw is None:
            return None

        with torch.no_grad():
            vit_output = self.qwen_vl_interface.model.visual(
                pixel_values, grid_thw=image_grid_thw,
            )  # [total_merged_patches, D_vis]

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
        return torch.stack(padded, dim=0)  # [B, N_vis_max, D_vis]

    # ------------------------------------------------------------------
    # InfoNCE contrastive loss
    # ------------------------------------------------------------------
    def _contrastive_loss(
        self,
        delta_h: torch.Tensor,   # [B, H]
        lang_emb: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:
        B = delta_h.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=delta_h.device, dtype=delta_h.dtype)

        delta_h_norm = F.normalize(delta_h.float(), dim=-1)
        lang_norm = F.normalize(lang_emb.float(), dim=-1)

        logits = torch.mm(delta_h_norm, lang_norm.t()) / self.contrastive_tau
        labels = torch.arange(B, device=logits.device)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------
    # Training forward
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

        # ===== LLR loss =====
        kl_loss = self._compute_language_llr_from_boundaries(
            priori_logits=priori_logits,
            posteriori_logits=posteriori_logits,
            priori_input_ids=qwen_inputs_priori["input_ids"],
            posteriori_input_ids=qwen_inputs_posteriori["input_ids"],
            priori_action_starts=priori_action_starts,
            posteriori_action_starts=posteriori_action_starts,
        )

        # ===== Latent Contrastive Alignment (InfoNCE) =====
        delta_h = (posteriori_action_hidden - priori_action_hidden).mean(dim=1)
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

            vis_feats_repeated = None
            if self.enable_vision_cross_attn:
                vis_feats = self._extract_vision_features(qwen_inputs_posteriori)
                if vis_feats is not None:
                    vis_feats_repeated = vis_feats.repeat(repeated_diffusion_steps, 1, 1).float()

            prior_loss = self.action_model(priori_cond, actions_target_repeated, state_repeated, vision_features=vis_feats_repeated)
            main_loss = self.action_model(posteriori_cond, actions_target_repeated, state_repeated, vision_features=vis_feats_repeated)

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
        # Vision features directly from ViT for DiT cross-attention
        # ============================================================
        vis_feats = None
        if self.enable_vision_cross_attn:
            vis_feats = self._extract_vision_features(qwen_inputs_post)

        # ============================================================
        # Action prediction with CAG guidance
        # ============================================================
        with torch.autocast("cuda", dtype=torch.float32):
            vis_feats_f = vis_feats.float() if vis_feats is not None else None
            if mode == "latent":
                action_hidden_guided = (
                    action_hidden_prior + omega_vec * (action_hidden_post - action_hidden_prior)
                )
                pred_actions = self.action_model.predict_action(
                    action_hidden_guided.float(), state_tensor, vision_features=vis_feats_f
                )
            elif mode == "velocity":
                pred_actions = self.action_model.predict_action_guided(
                    vl_embs_cond=action_hidden_post,
                    vl_embs_uncond=action_hidden_prior,
                    state=state_tensor,
                    omega=omega,
                    vision_features=vis_feats_f,
                )
            else:
                a_cond = self.action_model.predict_action(action_hidden_post, state_tensor, vision_features=vis_feats_f)
                a_uncond = self.action_model.predict_action(action_hidden_prior, state_tensor, vision_features=vis_feats_f)
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
