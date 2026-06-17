#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC_ROOT = Path("/mnt/e/data/Echo2Mix")
DEFAULT_OUT_ROOT = PROJECT_ROOT / "DataPreProcess"
DEFAULT_SPLITS = ["train", "val", "test"]


SOURCE_FILES = {
    "mix": "mix.wav",
    "s1": "spk1_reverb.wav",
    "s2": "spk2_reverb.wav",
}


OUTPUT_FILES = {
    "mix": "mix.json",
    "s1": "s1.json",
    "s2": "s2.json",
}


def wav_info(path):
    info = sf.info(str(path))
    return [str(path.resolve()), int(info.frames)]


def collect_split_infos(split_dir):
    mix_infos = []
    s1_infos = []
    s2_infos = []
    missing = []

    mix_paths = sorted(split_dir.rglob(SOURCE_FILES["mix"]))
    for mix_path in mix_paths:
        sample_dir = mix_path.parent
        s1_path = sample_dir / SOURCE_FILES["s1"]
        s2_path = sample_dir / SOURCE_FILES["s2"]
        if not s1_path.exists() or not s2_path.exists():
            missing.append(str(sample_dir))
            continue

        mix_infos.append(wav_info(mix_path))
        s1_infos.append(wav_info(s1_path))
        s2_infos.append(wav_info(s2_path))

    return {"mix": mix_infos, "s1": s1_infos, "s2": s2_infos}, missing


def write_split_jsons(split, infos, out_root):
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, out_name in OUTPUT_FILES.items():
        out_path = out_dir / out_name
        with out_path.open("w") as f:
            json.dump(infos[key], f, indent=4)
        print(f"Wrote {out_path}: {len(infos[key])} items")


def main():
    parser = argparse.ArgumentParser(description="Create Echo2Mix JSON manifests for SPMamba.")
    parser.add_argument("--src_root", type=Path, default=DEFAULT_SRC_ROOT, help="Echo2Mix wav root containing train/val/test.")
    parser.add_argument(
        "--out_root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Output root. Defaults to SPMamba/DataPreProcess to match spmamba-echo2mix.yml.",
    )
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS, help="Splits to process, e.g. train val test.")
    args = parser.parse_args()

    src_root = args.src_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    for split in args.splits:
        split_dir = src_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing Echo2Mix split directory: {split_dir}")

        print(f"Processing {split}: {split_dir}")
        infos, missing = collect_split_infos(split_dir)
        counts = {key: len(value) for key, value in infos.items()}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"Mismatched counts in {split}: {counts}")
        if missing:
            print(f"Warning: skipped {len(missing)} sample dir(s) with missing source wav files.")

        write_split_jsons(split, infos, out_root)


if __name__ == "__main__":
    main()
