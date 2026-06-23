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
warnings.filterwarnings("ignore", message="Possible clipped samples in output.")

# ----------------------------------------------------------------------- #
#  Constants
# ----------------------------------------------------------------------- #
SAMPLE_RATE = 16000
LUFS_TARGET = -17.0          # SonicSim uses -17 LUFS for speech sources
LUFS_JITTER = 2.0            # ±2 dB random variation (SonicSim style)


# ----------------------------------------------------------------------- #
#  Synthetic RIR generation (image-source method, simplified)
# ----------------------------------------------------------------------- #
def generate_synthetic_rir(
    sample_rate: int = SAMPLE_RATE,
    room_dim_range: tuple = ((3, 4, 2.5), (10, 8, 6)),
    rt60_range: tuple = (0.2, 0.7),
) -> np.ndarray:
    """
    Generate a synthetic monaural Room Impulse Response using a simplified
    image-source method.

    The RIR consists of:
      - Direct path (delta with attenuation based on distance)
      - Early reflections (image sources up to order 3)
      - Late reverberation (exponentially decaying white noise)

    Returns
    -------
    np.ndarray, shape (L,)
        Synthetic RIR of length L = ceil(rt60 * sample_rate).
    """
    # --- Random room parameters ---
    min_d, max_d = room_dim_range
    room = np.array([
        random.uniform(min_d[i], max_d[i]) for i in range(3)
    ])
    rt60 = random.uniform(*rt60_range)

    # --- Random source and receiver positions ---
    src = np.array([random.uniform(0.5, d - 0.5) for d in room])
    rcv = np.array([random.uniform(0.5, d - 0.5) for d in room])
    while np.linalg.norm(src - rcv) < 1.0:
        rcv = np.array([random.uniform(0.5, d - 0.5) for d in room])

    # --- Reflection coefficient from RT60 (Sabine equation) ---
    V = np.prod(room)
    S = 2 * (room[0] * room[1] + room[0] * room[2] + room[1] * room[2])
    alpha = 0.161 * V / (S * rt60)
    alpha = np.clip(alpha, 0.01, 0.99)
    beta = math.sqrt(1.0 - alpha)

    # --- RIR length ---
    rir_length = int(math.ceil(rt60 * sample_rate))
    rir = np.zeros(rir_length, dtype=np.float64)

    c = 343.0  # speed of sound

    # --- Direct path ---
    dist_direct = np.linalg.norm(src - rcv)
    delay_direct = int(round(dist_direct / c * sample_rate))
    if delay_direct < rir_length:
        rir[delay_direct] = 1.0 / (4 * math.pi * dist_direct)

    # --- Early reflections (image sources up to order 3) ---
    max_order = 3
    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                # Image source position (standard rectangular room formula):
                #   even n:  img = n * L + src
                #   odd  n:  img = n * L + (L - src)
                img = np.array([
                    nx * room[0] + src[0] if nx % 2 == 0 else nx * room[0] + (room[0] - src[0]),
                    ny * room[1] + src[1] if ny % 2 == 0 else ny * room[1] + (room[1] - src[1]),
                    nz * room[2] + src[2] if nz % 2 == 0 else nz * room[2] + (room[2] - src[2]),
                ])
                dist = np.linalg.norm(img - rcv)
                if dist < 0.5:
                    continue
                delay = int(round(dist / c * sample_rate))
                if delay >= rir_length:
                    continue
                attenuation = (1.0 / (4 * math.pi * dist)) * (
                    beta ** (abs(nx) + abs(ny) + abs(nz))
                )
                rir[delay] += attenuation

    # --- Late reverberation tail (exponentially decaying noise) ---
    tail_start = int(0.05 * sample_rate)
    if tail_start < rir_length:
        tail_len = rir_length - tail_start
        t = np.arange(tail_len) / sample_rate
        decay = np.exp(-6.91 * t / rt60)
        noise = np.random.randn(tail_len) * decay
        if tail_start > 0 and abs(rir[tail_start - 1]) > 0:
            noise_scale = abs(rir[tail_start - 1]) * 0.5
        else:
            noise_scale = 0.01
        noise = noise * noise_scale / (np.std(noise) + 1e-10)
        rir[tail_start:] += noise

    # --- Normalise peak to 1.0 ---
    peak = np.max(np.abs(rir))
    if peak > 0:
        rir = rir / peak

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
