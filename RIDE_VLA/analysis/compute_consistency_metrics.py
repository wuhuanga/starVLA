"""
RIDE-VLA Consistency Analysis: D_h / D_a / eRank

For each of the four training variants (base / aug_only / output_consistency / full),
this script measures how much the action-query representation and predicted action
change when the input is perturbed (visual aug + language paraphrase).

Metrics
-------
D_h   : mean (1 - cosine_sim) between clean and perturbed action-query hidden states.
         Intuition: low D_h = the representation is invariant to perturbation.
         Expected ordering: base ≈ aug_only >> output_consistency >> full (ridevla).
         Key test: if D_h(AC) >> D_h(full), "representation-level > output-level" holds.
D_a   : mean cosine distance between clean and perturbed predicted *velocity vectors*
         (sampled once at a fixed noise level t=0.5, averaged over action tokens).
         Lower = more action-consistent.
eRank : effective rank of the clean action-query matrix [B*K, H].
         eRank = exp(-sum(p_i * log(p_i))), p_i = σ_i / Σ σ_i  (Roy & Vetterli 2007).
         Higher = richer, more diverse representation.

Usage
-----
cd /nfs/ofs-llm-ssd/user/shengrenren_i/research/chd
CUDA_VISIBLE_DEVICES=0 \
/nfs/ofs-llm-ssd/user/shengrenren_i/envs/statvla/bin/python \
    RIDE_VLA/analysis/compute_consistency_metrics.py \
    [--n_samples 256] [--seed 42] [--t_fixed 0.5]
"""

import argparse
import json
import math
import random

# ── project imports ──────────────────────────────────────────────────────────
import sys
from pathlib import Path
from typing import List

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # chd/

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.intent_vla import ParaphraseBank, RoboSafeAugment

# ── config ───────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/nfs/ofs-llm-ssd/user/shengrenren_i/research/chd/playground/Datasets/LEROBOT_LIBERO_DATA")
SUITE = "libero_spatial_no_noops_1.0.0_lerobot"

CKPT_MAP = {
    "base": "/nfs/ofs-llab-hdd/users/shengrenren_i/IntentVLA/LIBERO_base/final_model/pytorch_model.pt",
    "aug_only": "/nfs/ofs-llab-hdd/users/shengrenren_i/IntentVLA/LIBERO_aug_only/final_model/pytorch_model.pt",
    "output_consistency": "/nfs/ofs-llab-hdd/users/shengrenren_i/IntentVLA/LIBERO_output_consistency/final_model/pytorch_model.pt",
    "full": "/nfs/ofs-llab-hdd/users/shengrenren_i/IntentVLA/LIBERO_full_new/final_model/pytorch_model.pt",
    # Locus ablation variants (2026-06-12)
    "locus_visual": "/nfs/ofs-llab-hdd/users/shengrenren_i/IntentVLA/LIBERO_locus_visual/final_model/pytorch_model.pt",
    "locus_all_hidden": "/nfs/ofs-llab-hdd/users/shengrenren_i/IntentVLA/LIBERO_locus_all_hidden/final_model/pytorch_model.pt",
    "locus_output": "/nfs/ofs-llab-hdd/users/shengrenren_i/IntentVLA/LIBERO_locus_output/final_model/pytorch_model.pt",
    "full_low_distill": "/nfs/ofs-llab-hdd/users/shengrenren_i/IntentVLA/LIBERO_full_low_distill/final_model/pytorch_model.pt",
}

PARAPHRASE_BANK_PATH = "/nfs/ofs-llm-ssd/user/shengrenren_i/research/chd/processed_instructions.json"


# ── data loading ─────────────────────────────────────────────────────────────


def _read_video_frame(video_path: Path, frame_idx: int) -> Image.Image:
    """Decode exactly one frame from an av1 mp4 using PyAV."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    for i, frame in enumerate(container.decode(stream)):
        if i == frame_idx:
            container.close()
            return frame.to_image()
    container.close()
    raise IndexError(f"Frame {frame_idx} not found in {video_path}")


def load_samples(n_samples: int, seed: int) -> List[dict]:
    """
    Sample n_samples (image, lang, action) tuples from the LIBERO-Spatial dataset.
    Picks one frame from the middle of each episode to avoid run-in/run-out bias.
    """
    import pandas as pd

    rng = random.Random(seed)
    suite_root = DATASET_ROOT / SUITE

    # Load task index -> language mapping
    tasks_path = suite_root / "meta" / "tasks.jsonl"
    task_lang = {}
    with open(tasks_path) as f:
        for line in f:
            rec = json.loads(line)
            task_lang[rec["task_index"]] = rec["task"]

    # Find all episode parquet files
    data_dir = suite_root / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    sampled_files = rng.choices(parquet_files, k=n_samples)

    samples = []
    for pf in sampled_files:
        ep_idx = int(pf.stem.split("_")[1])
        df = pd.read_parquet(pf)
        mid = len(df) // 2
        row = df.iloc[mid]

        task_idx = int(row["task_index"])
        lang = task_lang[task_idx]
        action = np.array(row["action"], dtype=np.float32)

        video_path = suite_root / "videos" / "chunk-000" / "observation.images.image" / f"episode_{ep_idx:06d}.mp4"
        frame_idx = int(row["frame_index"])
        image = _read_video_frame(video_path, frame_idx).resize((224, 224))

        samples.append({"image": image, "lang": lang, "action": action})

    return samples


# ── metric computation ────────────────────────────────────────────────────────


def erank(h: torch.Tensor) -> float:
    """
    Effective rank of h [N, D] (Roy & Vetterli 2007).
    eRank = exp(H(p))  where p_i = σ_i / Σ σ_i and H is Shannon entropy.
    """
    h_fp32 = h.float()
    _, S, _ = torch.linalg.svd(h_fp32, full_matrices=False)
    S = S[S > 1e-9]
    p = S / S.sum()
    ent = -(p * p.log()).sum().item()
    return math.exp(ent)


@torch.no_grad()
def compute_metrics_for_model(
    model,
    samples: List[dict],
    aug: RoboSafeAugment,
    paraphrase: ParaphraseBank,
    t_fixed: float,
    batch_size: int = 8,
) -> dict:
    """
    Run clean and perturbed forward passes; compute D_h, D_a, eRank.
    """
    tokenizer = model.qwen_vl_interface.processor.tokenizer
    device = next(model.qwen_vl_interface.parameters()).device

    all_h_clean, all_h_pert = [], []
    all_v_clean, all_v_pert = [], []  # predicted velocities at t_fixed

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]

        images_clean = [[s["image"]] for s in batch]
        images_pert = [[aug(s["image"])] for s in batch]
        lang_clean = [s["lang"] + model.latent_action_query for s in batch]
        lang_pert = [paraphrase.sample(s["lang"]) + model.latent_action_query for s in batch]
        actions = np.array([s["action"] for s in batch])  # [B, 7]

        # Encode clean
        h_c = model._encode_action_queries(
            model.qwen_vl_interface, images_clean, lang_clean, tokenizer, no_grad=True
        )  # [B, K, H]

        # Encode perturbed
        h_p = model._encode_action_queries(
            model.qwen_vl_interface, images_pert, lang_pert, tokenizer, no_grad=True
        )  # [B, K, H]

        all_h_clean.append(h_c.cpu())
        all_h_pert.append(h_p.cpu())

        # Velocity at fixed t for D_a
        B = h_c.shape[0]
        T = model.future_action_window_size + 1
        act_np = actions[:, np.newaxis, :].repeat(T, axis=1)  # [B, T, 7]
        act_t = torch.tensor(act_np, device=device, dtype=h_c.dtype)

        eps = torch.randn_like(act_t)
        s = torch.full((B,), t_fixed, device=device, dtype=h_c.dtype)
        x_s = (1.0 - t_fixed) * eps + t_fixed * act_t

        with torch.autocast("cuda", dtype=torch.float32):
            v_c = model.action_model.predict_velocity(h_c.float(), x_s, s, state=None)
            v_p = model.action_model.predict_velocity(h_p.float(), x_s, s, state=None)

        all_v_clean.append(v_c.cpu())
        all_v_pert.append(v_p.cpu())

    H_clean = torch.cat(all_h_clean, dim=0)  # [N, K, H]
    H_pert = torch.cat(all_h_pert, dim=0)

    V_clean = torch.cat(all_v_clean, dim=0)  # [N, 1, 7] or similar
    V_pert = torch.cat(all_v_pert, dim=0)

    # D_h: mean (1 - cosine) over all (sample, token) pairs
    cos_h = F.cosine_similarity(H_clean, H_pert, dim=-1)  # [N, K]
    d_h = (1.0 - cos_h).mean().item()

    # D_a: mean (1 - cosine) over all (sample, token) pairs of velocity
    cos_a = F.cosine_similarity(
        V_clean.view(-1, V_clean.shape[-1]),
        V_pert.view(-1, V_pert.shape[-1]),
        dim=-1,
    )
    d_a = (1.0 - cos_a).mean().item()

    # eRank on clean hidden states: flatten to [N*K, H]
    N, K, Hdim = H_clean.shape
    er = erank(H_clean.view(N * K, Hdim))

    return {"D_h": d_h, "D_a": d_a, "eRank": er}


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t_fixed", type=float, default=0.5, help="Noise level t for velocity comparison (D_a)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "base",
            "aug_only",
            "output_consistency",
            "full",
            "locus_visual",
            "locus_all_hidden",
            "locus_output",
            "full_low_distill",
        ],
        help="Which variants to evaluate",
    )
    args = parser.parse_args()

    print(f"Loading {args.n_samples} samples from {SUITE} (seed={args.seed})...")
    samples = load_samples(args.n_samples, args.seed)
    print(f"  Loaded {len(samples)} samples.")

    aug = RoboSafeAugment(p_apply=1.0)
    paraphrase = ParaphraseBank(PARAPHRASE_BANK_PATH if Path(PARAPHRASE_BANK_PATH).exists() else None)

    results = {}
    for name in args.models:
        ckpt = CKPT_MAP[name]
        print(f"\n[{name}] Loading from {ckpt} ...")
        model = baseframework.from_pretrained(ckpt)
        model = model.cuda().eval()

        print(f"[{name}] Computing metrics on {len(samples)} samples ...")
        metrics = compute_metrics_for_model(
            model,
            samples,
            aug,
            paraphrase,
            t_fixed=args.t_fixed,
            batch_size=args.batch_size,
        )
        results[name] = metrics
        print(f"[{name}]  D_h={metrics['D_h']:.4f}  D_a={metrics['D_a']:.4f}  eRank={metrics['eRank']:.1f}")

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # ── summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Method':<22} {'D_h':>8} {'D_a':>8} {'eRank':>8}")
    print("-" * 60)
    for name in args.models:
        m = results[name]
        print(f"{name:<22} {m['D_h']:>8.4f} {m['D_a']:>8.4f} {m['eRank']:>8.1f}")
    print("=" * 60)

    # Save JSON
    out_path = Path(__file__).parent / "consistency_metrics.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
