import json
from pathlib import Path

import soundfile as sf

SRC_ROOT = "/mnt/e/data/LibriMix/Libri2Mix/wav8k/min"
OUT_ROOT = "/mnt/e/project/prjANS/src/AVSE/SPMamba/DataPreProcess/Libri2Mix"
SPLITS = ["train-100", "dev", "test"]
SUBDIRS = ["mix_clean", "s1", "s2"]


def collect_wav_infos(wav_dir: Path):
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"Missing wav directory: {wav_dir}")

    infos = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        wav_info = sf.info(str(wav_path))
        infos.append([str(wav_path.resolve()), int(wav_info.frames)])
    return infos


def main():
    for split in SPLITS:
        split_out_dir = OUT_ROOT / split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        split_counts = {}
        for subdir in SUBDIRS:
            infos = collect_wav_infos(SRC_ROOT / split / subdir)
            out_name = "mix_clean.json" if subdir == "mix_clean" else f"{subdir}.json"
            out_path = split_out_dir / out_name
            with out_path.open("w") as f:
                json.dump(infos, f, indent=4)
            split_counts[subdir] = len(infos)
            print(f"Wrote {out_path}: {len(infos)} items")

        if len(set(split_counts.values())) != 1:
            raise RuntimeError(f"Mismatched counts in {split}: {split_counts}")


if __name__ == "__main__":
    main()
