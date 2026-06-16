#!/usr/bin/env python
import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch
import yaml

from infer_two_sources import (
    load_model,
    load_mono,
    print_metrics_file,
    resample_if_needed,
    write_single_target_metrics,
)


def parse_scene_number(scene_id):
    if len(scene_id) > 1 and scene_id[0].upper() == "S" and scene_id[1:].isdigit():
        return int(scene_id[1:])
    return None


def find_pairs(scenes_dir, mixed_suffix, target_suffix, start_id=None, end_id=None):
    mixed_files = sorted(scenes_dir.glob(f"*{mixed_suffix}"))
    pairs = []
    missing_targets = []

    for mixed_path in mixed_files:
        scene_id = mixed_path.name[: -len(mixed_suffix)]
        scene_number = parse_scene_number(scene_id)
        if start_id is not None and scene_number is not None and scene_number < start_id:
            continue
        if end_id is not None and scene_number is not None and scene_number > end_id:
            continue

        target_path = mixed_path.with_name(f"{scene_id}{target_suffix}")
        if target_path.exists():
            pairs.append((scene_id, mixed_path, target_path))
        else:
            missing_targets.append((scene_id, mixed_path, target_path))

    return pairs, missing_targets


def read_metric_row(metrics_path):
    with open(metrics_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def get_pair_frames(pair):
    _, mixed_path, target_path = pair
    try:
        return min(sf.info(str(mixed_path)).frames, sf.info(str(target_path)).frames)
    except RuntimeError:
        return 0


def load_pair_audio(scene_id, mixed_path, target_path, sample_rate):
    mix, mix_sr = load_mono(str(mixed_path))
    mix = resample_if_needed(mix, mix_sr, sample_rate, f"{scene_id} mix")

    target, target_sr = load_mono(str(target_path))
    target = resample_if_needed(target, target_sr, sample_rate, f"{scene_id} s1")

    length = min(len(mix), len(target))
    return mix[:length], target[:length], length


def pad_audio_batch(items):
    max_len = max(item["length"] for item in items)
    mix_batch = np.zeros((len(items), max_len), dtype=np.float32)
    for idx, item in enumerate(items):
        mix_batch[idx, : item["length"]] = item["mix"]
    return mix_batch


def process_batch(batch_pairs, output_root, model, sample_rate, device, save_mix=True, print_metrics=False):
    items = []
    for scene_id, mixed_path, target_path in batch_pairs:
        mix, target, length = load_pair_audio(scene_id, mixed_path, target_path, sample_rate)
        items.append(
            {
                "scene_id": scene_id,
                "mixed_path": mixed_path,
                "target_path": target_path,
                "mix": mix,
                "target": target,
                "length": length,
            }
        )

    mix_batch = torch.from_numpy(pad_audio_batch(items)).to(device)
    with torch.inference_mode():
        estimates = model(mix_batch)
    if estimates.ndim == 2:
        estimates = estimates[:, None, :]

    rows = []
    for batch_idx, item in enumerate(items):
        scene_id = item["scene_id"]
        output_dir = output_root / scene_id
        wav_dir = output_dir / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)

        length = min(item["length"], estimates.shape[-1])
        mix_tensor = torch.from_numpy(item["mix"][:length]).to(device)
        target_tensor = torch.from_numpy(item["target"][:length]).to(device)
        estimate = estimates[batch_idx, :, :length]

        if save_mix:
            sf.write(wav_dir / "mix.wav", mix_tensor.cpu().numpy(), sample_rate)
        for idx, source in enumerate(estimate.detach().cpu().numpy(), start=1):
            sf.write(wav_dir / f"estimate_s{idx}.wav", source, sample_rate)

        metrics_path = output_dir / "metrics.csv"
        best_idx = write_single_target_metrics(mix_tensor, target_tensor, estimate, "s1", str(metrics_path))
        sf.write(wav_dir / "best_s1.wav", estimate[best_idx - 1].detach().cpu().numpy(), sample_rate)

        row = read_metric_row(metrics_path)
        row["scene_id"] = scene_id
        row["mix"] = str(item["mixed_path"])
        row["target"] = str(item["target_path"])
        row["output_dir"] = str(output_dir)
        rows.append(row)
        if print_metrics:
            print_metrics_file(str(metrics_path))

    return rows


def write_summary(summary_path, rows):
    if not rows:
        return
    keys = [
        "scene_id",
        "best_estimate",
        "sdr",
        "sdr_i",
        "si-snr",
        "si-snr_i",
        "mix",
        "target",
        "output_dir",
    ]
    extra_keys = sorted({key for row in rows for key in row.keys()} - set(keys))
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + extra_keys)
        writer.writeheader()
        writer.writerows(rows)


def write_failures(failures_path, failures):
    if not failures:
        return
    with open(failures_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "mix", "target", "error"])
        writer.writeheader()
        writer.writerows(failures)


def main():
    parser = argparse.ArgumentParser(description="Batch SPMamba inference for AVSEC scene wav pairs.")
    parser.add_argument("--scenes_dir", default="/mnt/e/data/AVSEC/avsec3/scenes")
    parser.add_argument("--output_root", default="/mnt/e/data/AVSEC/avsec3/spmamba")
    parser.add_argument("--conf_dir", default="configs/spmamba-echo2mix.yml")
    parser.add_argument("--mixed_suffix", default="_mixed.wav")
    parser.add_argument("--target_suffix", default="_target.wav")
    parser.add_argument("--start_id", type=int, default=None, help="Optional numeric scene id start, e.g. 50000.")
    parser.add_argument("--end_id", type=int, default=None, help="Optional numeric scene id end, e.g. 50100.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of pairs to process.")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of scenes to infer per model forward pass.")
    parser.add_argument("--no_sort_by_length", action="store_true", help="Disable length sorting before batching.")
    parser.add_argument("--no_save_mix", action="store_true", help="Do not write copied mix.wav files; saves disk I/O.")
    parser.add_argument("--print_each_metrics", action="store_true", help="Print metrics for every scene; slower for large batches.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess scenes that already have metrics.csv.")
    parser.add_argument("--dry_run", action="store_true", help="Only list matched pairs without running inference.")
    args = parser.parse_args()

    script_dir = PROJECT_ROOT
    os.chdir(script_dir)

    scenes_dir = Path(args.scenes_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    conf_path = Path(args.conf_dir).expanduser()
    if not conf_path.is_absolute():
        conf_path = script_dir / conf_path

    pairs, missing_targets = find_pairs(
        scenes_dir,
        args.mixed_suffix,
        args.target_suffix,
        start_id=args.start_id,
        end_id=args.end_id,
    )
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not args.no_sort_by_length:
        pairs = sorted(pairs, key=get_pair_frames)

    print(f"Found {len(pairs)} matched pair(s) in: {scenes_dir}")
    if missing_targets:
        print(f"Warning: {len(missing_targets)} mixed file(s) have no matching target file.")
    if args.dry_run:
        for scene_id, mixed_path, target_path in pairs[:20]:
            print(f"{scene_id}: {mixed_path.name} + {target_path.name}")
        if len(pairs) > 20:
            print(f"... {len(pairs) - 20} more pair(s)")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    with open(conf_path, "rb") as f:
        conf = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device} with config: {conf_path}")
    model, sample_rate = load_model(conf, device)

    summary_rows = []
    failures = []
    total = len(pairs)

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    pending_pairs = []
    for index, pair in enumerate(pairs, start=1):
        scene_id, mixed_path, target_path = pair
        output_dir = output_root / scene_id
        metrics_path = output_dir / "metrics.csv"
        if metrics_path.exists() and not args.overwrite:
            print(f"[{index}/{total}] Skip existing: {scene_id}")
            row = read_metric_row(metrics_path)
            row["scene_id"] = scene_id
            row["mix"] = str(mixed_path)
            row["target"] = str(target_path)
            row["output_dir"] = str(output_dir)
            summary_rows.append(row)
            continue
        pending_pairs.append(pair)

    pending_total = len(pending_pairs)
    for start in range(0, pending_total, args.batch_size):
        batch_pairs = pending_pairs[start : start + args.batch_size]
        batch_label = f"{batch_pairs[0][0]}..{batch_pairs[-1][0]}" if len(batch_pairs) > 1 else batch_pairs[0][0]
        print(f"Batch {start // args.batch_size + 1}/{(pending_total + args.batch_size - 1) // args.batch_size}: {batch_label}")
        try:
            rows = process_batch(
                batch_pairs,
                output_root,
                model,
                sample_rate,
                device,
                save_mix=not args.no_save_mix,
                print_metrics=args.print_each_metrics,
            )
            summary_rows.extend(rows)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[Batch Error] {batch_label}: {exc}")
            print("Retrying this batch one scene at a time.")
            for scene_id, mixed_path, target_path in batch_pairs:
                try:
                    rows = process_batch(
                        [(scene_id, mixed_path, target_path)],
                        output_root,
                        model,
                        sample_rate,
                        device,
                        save_mix=not args.no_save_mix,
                        print_metrics=args.print_each_metrics,
                    )
                    summary_rows.extend(rows)
                except Exception as item_exc:
                    print(f"[Error] {scene_id}: {item_exc}")
                    failures.append(
                        {
                            "scene_id": scene_id,
                            "mix": str(mixed_path),
                            "target": str(target_path),
                            "error": repr(item_exc),
                        }
                    )
        except Exception as exc:
            print(f"[Batch Error] {batch_label}: {exc}")
            for scene_id, mixed_path, target_path in batch_pairs:
                failures.append(
                    {
                        "scene_id": scene_id,
                        "mix": str(mixed_path),
                        "target": str(target_path),
                        "error": repr(exc),
                    }
                )

    summary_path = output_root / "summary_metrics.csv"
    failures_path = output_root / "failed_scenes.csv"
    write_summary(summary_path, summary_rows)
    write_failures(failures_path, failures)

    print(f"Done. Successful or skipped scenes: {len(summary_rows)}")
    print(f"Summary metrics: {summary_path}")
    if failures:
        print(f"Failures: {failures_path}")


if __name__ == "__main__":
    main()
