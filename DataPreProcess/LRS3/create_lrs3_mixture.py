#!/usr/bin/env python
"""
Create 2-speaker mixtures for the LRS3 dataset using the same mixture pair
list as ClearerVoice-Studio, enabling fair audio-only vs audio-visual
comparison.

The CSV file (from ClearerVoice) has columns:
    split, src1_subset, src1_speaker, src1_clip, offset,
          src2_subset, src2_speaker, src2_clip, snr, duration

Audio paths follow ClearerVoice convention:
    {lrs3_root}/wav/{subset}/{speaker}/{clip}.wav

Mixing logic mirrors ClearerVoice's dataset_lip.py:
    1. Load both speakers' dry audio.
    2. Power-normalise interference to match target power.
    3. SNR scale: snr_0 = 10^(offset/20), snr_1 = 10^(snr/20).
    4. Normalise by max_snr, apply individual SNR, sum.

With --reverb, an additional synthetic RIR convolution + LUFS normalisation
step is applied before mixing (SonicSim-style).

Output files (per mixture pair directory):
    mix.wav          – the 2-speaker mixture
    s1.wav          – speaker 1 (target), reverberant if --reverb
    s2.wav          – speaker 2 (interference), reverberant if --reverb

Usage
-----
    # Dry mixtures (matches ClearerVoice exactly)
    python create_lrs3_mixture.py \\
        --lrs3_root /mnt/e/data/LRS3 \\
        --csv_input /path/to/mixture_data_list_2mix.csv \\
        --out_root /mnt/e/data/LRS3/LRS3-2mix \\
        --splits train val

    # With synthetic reverberation
    python create_lrs3_mixture.py \\
        --lrs3_root /mnt/e/data/LRS3 \\
        --csv_input /path/to/mixture_data_list_2mix.csv \\
        --out_root /mnt/e/data/LRS3/LRS3-2mix-reverb \\
        --splits train val \\
        --reverb

    # Quick test
    python create_lrs3_mixture.py \\
        ... --splits train --max_pairs 5
"""

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import pyloudnorm as pyln
import warnings
import pyroomacoustics as pra
warnings.filterwarnings("ignore", message="Possible clipped samples in output.")

# ----------------------------------------------------------------------- #
#  Constants
# ----------------------------------------------------------------------- #
SAMPLE_RATE = 16000
LUFS_TARGET = -17.0          # SonicSim uses -17 LUFS for speech sources
LUFS_JITTER = 2.0            # ±2 dB random variation (SonicSim style)


# -----------------------------------------------------------------------
#  RIR generation using pyroomacoustics (diverse room types)
# --------------------------------------------------------------------- #
# Room type presets mimicking Echo2Mix/SonicSim scene diversity
ROOM_PRESETS = [
    # (name, dim_range, rt60_range, materials)
    ("living_room",    ((4, 5, 2.6), (8, 7, 3.5)), (0.3, 0.6), "medium"),
    ("bedroom",        ((3, 3, 2.4), (5, 4, 2.8)), (0.2, 0.4), "soft"),
    ("kitchen",        ((3, 3, 2.5), (6, 5, 3.0)), (0.15, 0.35), "hard"),
    ("bathroom",       ((2, 2, 2.3), (4, 3, 2.6)), (0.4, 0.8), "hard"),
    ("dining_room",    ((4, 4, 2.6), (7, 6, 3.2)), (0.3, 0.55), "medium"),
    ("meeting_room",   ((5, 4, 2.8), (10, 8, 3.5)), (0.4, 0.7), "medium"),
    ("hallway",        ((2, 4, 2.4), (3, 12, 2.8)), (0.3, 0.6), "hard"),
    ("lounge",         ((5, 5, 2.8), (9, 8, 3.5)), (0.35, 0.65), "soft"),
    ("conference_room", ((6, 5, 2.8), (12, 10, 3.5)), (0.4, 0.7), "medium"),
    ("office",         ((3, 3, 2.5), (6, 5, 3.0)), (0.25, 0.5), "medium"),
]

# Material absorption coefficients (low, mid, high frequency bands)
MATERIAL_BANK = {
    "hard":   {  # bathroom, kitchen — tiles, concrete
        "walls": 0.05, "floor": 0.02, "ceiling": 0.10,
    },
    "medium": {  # living room, dining room — drywall, wood floor
        "walls": 0.15, "floor": 0.10, "ceiling": 0.20,
    },
    "soft":   {  # bedroom, lounge — carpet, curtains
        "walls": 0.30, "floor": 0.40, "ceiling": 0.30,
    },
}


def generate_synthetic_rir(
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate a realistic Room Impulse Response using pyroomacoustics.

    Randomly selects from diverse room types (living room, bedroom, kitchen,
    bathroom, hallway, etc.) with appropriate materials and dimensions to
    match the diversity of Echo2Mix/SonicSim scenes.

    Returns
    -------
    np.ndarray, shape (L,)
        RIR with direct path at sample 0, normalized so direct path = 1.0.
    """
    # --- Randomly select room type ---
    name, dim_range, rt60_range, mat_key = random.choice(ROOM_PRESETS)
    mats = MATERIAL_BANK[mat_key]

    # --- Random room dimensions ---
    room_dim = np.array([
        random.uniform(dim_range[0][i], dim_range[1][i]) for i in range(3)
    ])

    # --- Random source and receiver positions ---
    src_pos = np.array([
        random.uniform(0.5, room_dim[i] - 0.5) for i in range(3)
    ])
    rcv_pos = np.array([
        random.uniform(0.5, room_dim[i] - 0.5) for i in range(3)
    ])
    # Ensure minimum distance
    while np.linalg.norm(src_pos - rcv_pos) < 1.0:
        rcv_pos = np.array([
            random.uniform(0.5, room_dim[i] - 0.5) for i in range(3)
        ])

    # --- Target RT60 ---
    target_rt60 = random.uniform(*rt60_range)

    # --- Create room with materials ---
    # Build absorption per surface (6 walls: -x, +x, -y, +y, -z floor, +z ceiling)
    abs_walls = mats["walls"]
    abs_floor = mats["floor"]
    abs_ceil = mats["ceiling"]

    try:
        room = pra.ShoeBox(
            room_dim,
            fs=sample_rate,
            materials=pra.Material(abs_walls),
            max_order=5,
        )
        # Override floor and ceiling materials
        room.wall_materials[4] = pra.Material(abs_floor)   # floor (-z)
        room.wall_materials[5] = pra.Material(abs_ceil)    # ceiling (+z)
    except Exception:
        # Fallback: uniform absorption
        room = pra.ShoeBox(
            room_dim,
            fs=sample_rate,
            materials=pra.Material(0.15),
            max_order=5,
        )

    # --- Add source and microphone ---
    room.add_source(src_pos)
    room.add_microphone(rcv_pos)

    # --- Compute RIR ---
    room.compute_rir()
    rir = room.rir[0][0]  # mic 0, source 0

    # --- Find direct path and shift to sample 0 ---
    direct_idx = int(round(
        np.linalg.norm(src_pos - rcv_pos) / 343.0 * sample_rate
    ))
    if direct_idx > 0 and direct_idx < len(rir):
        rir = np.concatenate([rir[direct_idx:], np.zeros(direct_idx)])

    # --- Normalize: direct path = 1.0, reflections ≤ 0.3x ---
    direct_amp = abs(rir[0])
    if direct_amp > 0:
        max_other = np.max(np.abs(rir[1:])) if len(rir) > 1 else 0.0
        if max_other > 0:
            rir[1:] *= (direct_amp * 0.3) / max_other
        rir = rir / direct_amp
    else:
        # Fallback: peak normalize
        peak = np.max(np.abs(rir))
        if peak > 0:
            rir = rir / peak

    # --- Truncate to target RT60 length ---
    target_len = int(math.ceil(target_rt60 * sample_rate))
    if len(rir) > target_len:
        rir = rir[:target_len]

    return rir.astype(np.float32)


# ----------------------------------------------------------------------- #
#  FFT convolution (same as SonicSim)
# ----------------------------------------------------------------------- #
def fft_convolve(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a 1-D signal with a 1-D kernel using FFT."""
    s = torch.from_numpy(signal.astype(np.float32))
    k = torch.from_numpy(kernel.astype(np.float32))
    padded_signal = F.pad(s.reshape(-1), (0, k.size(-1) - 1))
    padded_kernel = F.pad(k.reshape(-1), (0, s.size(-1) - 1))
    signal_fr = torch.fft.rfftn(padded_signal, dim=-1)
    kernel_fr = torch.fft.rfftn(padded_kernel, dim=-1)
    output = torch.fft.irfftn(signal_fr * kernel_fr, dim=-1)
    return output.numpy()


# ----------------------------------------------------------------------- #
#  LUFS normalisation (SonicSim style, used only with --reverb)
# ----------------------------------------------------------------------- #
def lufs_normalize(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float = LUFS_TARGET,
    jitter: float = LUFS_JITTER,
) -> np.ndarray:
    """LUFS-normalise audio to a target loudness with random jitter."""
    audio_f64 = audio.astype(np.float64)
    block_size = 0.4 if len(audio_f64) / sample_rate >= 0.4 else len(audio_f64) / sample_rate
    meter = pyln.Meter(rate=sample_rate, block_size=block_size)
    loudness = meter.integrated_loudness(audio_f64)
    if math.isinf(loudness):
        loudness = -40.0
    random_target = random.uniform(target_lufs - jitter, target_lufs + jitter)
    norm_audio = pyln.normalize.loudness(audio_f64, loudness, random_target)
    return norm_audio.astype(np.float32)


# ----------------------------------------------------------------------- #
#  CSV parsing (ClearerVoice format)
# ----------------------------------------------------------------------- #
def parse_csv(csv_path: Path, split: str) -> List[dict]:
    """
    Parse the ClearerVoice mixture_data_list_2mix CSV and return entries
    matching the requested split.

    Each returned dict has:
        s1_subset, s1_speaker, s1_clip, offset,
        s2_subset, s2_speaker, s2_clip, snr, duration
    """
    entries = []
    with csv_path.open("r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 10:
                continue
            if row[0].strip() != split:
                continue
            entries.append({
                "s1_subset": row[1].strip(),
                "s1_speaker": row[2].strip(),
                "s1_clip": row[3].strip(),
                "offset": float(row[4]),
                "s2_subset": row[5].strip(),
                "s2_speaker": row[6].strip(),
                "s2_clip": row[7].strip(),
                "snr": float(row[8]),
                "duration": float(row[9]),
            })
    return entries


# ----------------------------------------------------------------------- #
#  Audio loading
# ----------------------------------------------------------------------- #
def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file, resample if needed, convert to mono."""
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != sample_rate:
        t = torch.from_numpy(audio).float()
        if t.ndim == 1:
            t = t.unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=None,
                          scale_factor=sample_rate / sr,
                          mode="linear", align_corners=False)
        audio = t.squeeze().numpy()
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


# ----------------------------------------------------------------------- #
#  Mixture creation for a single pair
# ----------------------------------------------------------------------- #
def create_mixture(
    s1_path: Path,
    s2_path: Path,
    offset: float,
    snr: float,
    out_dir: Path,
    sample_rate: int = SAMPLE_RATE,
    use_reverb: bool = False,
) -> bool:
    """
    Create a single 2-speaker mixture.

    Mixing logic mirrors ClearerVoice's dataset_lip.py:
      1. Load both speakers' dry audio.
      2. (Optional) Apply synthetic RIR convolution + LUFS normalise.
      3. Power-normalise interference to match target power.
      4. SNR scale: snr_0 = 10^(offset/20), snr_1 = 10^(snr/20).
      5. Normalise by max_snr, apply individual SNR, sum.
      6. Peak-normalise if clipping.

    Returns True on success, False on failure.
    """
    try:
        # --- Load dry audio ---
        audio1 = load_audio(s1_path, sample_rate)   # target (s1)
        audio2 = load_audio(s2_path, sample_rate)   # interference (s2)

        # --- Pad shorter audio to match the longer one ---
        len1, len2 = len(audio1), len(audio2)
        if len1 < len2:
            audio1 = np.pad(audio1, (0, len2 - len1), mode="constant")
        elif len2 < len1:
            audio2 = np.pad(audio2, (0, len1 - len2), mode="constant")

        # --- Optional reverberation ---
        if use_reverb:
            rir1 = generate_synthetic_rir(sample_rate)
            rir2 = generate_synthetic_rir(sample_rate)
            audio1 = fft_convolve(audio1, rir1)[:max(len1, len2)]
            audio2 = fft_convolve(audio2, rir2)[:max(len1, len2)]
            # LUFS normalise each source (SonicSim style)
            audio1 = lufs_normalize(audio1, sample_rate)
            audio2 = lufs_normalize(audio2, sample_rate)

        # --- ClearerVoice mixing logic ---
        # Power-normalise interference to match target power
        target_power = np.linalg.norm(audio1, 2) ** 2 / audio1.size
        intef_power = np.linalg.norm(audio2, 2) ** 2 / audio2.size
        if intef_power > 0:
            audio2 = audio2 * np.sqrt(target_power / intef_power)

        # SNR scaling (matching ClearerVoice dataset_lip.py lines 133-153)
        snr_0 = 10.0 ** (offset / 20.0)   # target gain (offset is typically 0 → 1.0)
        snr_1 = 10.0 ** (snr / 20.0)      # interference gain

        max_snr = max(snr_0, snr_1)
        if max_snr > 0:
            audio1 = audio1 / max_snr
            audio2 = audio2 / max_snr

        audio1 = audio1 * snr_0
        audio2 = audio2 * snr_1

        mixture = audio1 + audio2

        # --- Peak-normalise if clipping (ClearerVoice style) ---
        max_val = np.max(np.abs(mixture))
        if max_val > 1.0:
            mixture = mixture / max_val
            audio1 = audio1 / max_val
            audio2 = audio2 / max_val

        # --- Save outputs ---
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_dir / "mix.wav"), mixture.astype(np.float32), sample_rate)
        sf.write(str(out_dir / "s1.wav"), audio1.astype(np.float32), sample_rate)
        sf.write(str(out_dir / "s2.wav"), audio2.astype(np.float32), sample_rate)

        return True

    except Exception as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        return False


# ----------------------------------------------------------------------- #
#  Main
# ----------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Create LRS3-2mix mixtures from ClearerVoice CSV pair list."
    )
    parser.add_argument("--lrs3_root", type=Path, required=True,
                        help="LRS3 data root (e.g. /mnt/e/data/LRS3).")
    parser.add_argument("--csv_input", type=Path, required=True,
                        help="ClearerVoice mixture_data_list_2mix.csv path.")
    parser.add_argument("--out_root", type=Path, required=True,
                        help="Output root for LRS3-2mix wav data.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Splits to process (train, val). No test in CSV.")
    parser.add_argument("--max_pairs", type=int, default=0,
                        help="Max pairs per split (0 = all). For quick testing.")
    parser.add_argument("--reverb", action="store_true",
                        help="Apply synthetic RIR reverberation before mixing.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (RIR generation).")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    lrs3_root = args.lrs3_root.resolve()
    csv_path = args.csv_input.resolve()
    out_root = args.out_root.resolve()

    if not csv_path.is_file():
        print(f"[Error] CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    reverb_tag = " (with reverb)" if args.reverb else " (dry, no reverb)"
    print(f"\nLRS3-2mix mixture generation{reverb_tag}")
    print(f"  CSV:       {csv_path}")
    print(f"  LRS3 root: {lrs3_root}")
    print(f"  Output:    {out_root}")
    print(f"  Splits:    {args.splits}")
    print(f"  Reverb:    {args.reverb}")

    total_created = 0
    total_skipped = 0

    for split in args.splits:
        entries = parse_csv(csv_path, split)
        if not entries:
            print(f"\n[Warning] No '{split}' entries in CSV, skipping.")
            continue

        if args.max_pairs > 0:
            entries = entries[:args.max_pairs]

        print(f"\n{'='*60}")
        print(f"Processing split: {split}  ({len(entries)} pairs)")
        print(f"{'='*60}")

        split_created = 0
        split_skipped = 0

        for idx, entry in enumerate(entries):
            # Build wav paths following ClearerVoice convention:
            #   {lrs3_root}/wav/{subset}/{speaker}/{clip}.wav
            wav1 = lrs3_root / "wav" / entry["s1_subset"] / entry["s1_speaker"] / f"{entry['s1_clip']}.wav"
            wav2 = lrs3_root / "wav" / entry["s2_subset"] / entry["s2_speaker"] / f"{entry['s2_clip']}.wav"

            # Check files exist
            missing = []
            if not wav1.is_file():
                missing.append(str(wav1))
            if not wav2.is_file():
                missing.append(str(wav2))
            if missing:
                if idx < 5:
                    print(f"  [{idx+1}/{len(entries)}] SKIP (missing: {missing[0]})")
                split_skipped += 1
                continue

            # Output directory: {out_root}/{split}/{s1_speaker}_{s1_clip}-{s2_speaker}_{s2_clip}
            name1 = f"{entry['s1_speaker']}_{entry['s1_clip']}"
            name2 = f"{entry['s2_speaker']}_{entry['s2_clip']}"
            pair_dir = out_root / split / f"{name1}-{name2}"

            # Skip if already exists
            if (pair_dir / "mix.wav").is_file():
                if idx < 5:
                    print(f"  [{idx+1}/{len(entries)}] EXISTS {pair_dir.name}")
                split_created += 1
                continue

            ok = create_mixture(
                wav1, wav2,
                offset=entry["offset"],
                snr=entry["snr"],
                out_dir=pair_dir,
                use_reverb=args.reverb,
            )
            if ok:
                split_created += 1
                if (idx + 1) % 100 == 0 or idx < 5:
                    print(f"  [{idx+1}/{len(entries)}] OK   {pair_dir.name}")
            else:
                split_skipped += 1
                print(f"  [{idx+1}/{len(entries)}] FAIL  {pair_dir.name}")

        print(f"\n  {split}: created={split_created}, skipped={split_skipped}")
        total_created += split_created
        total_skipped += split_skipped

    print(f"\n{'='*60}")
    print(f"All splits done. Total created={total_created}, skipped={total_skipped}")
    print(f"Output root: {out_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
