#!/usr/bin/env python
import argparse
import csv
import gc
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


def make_dynamic_batches(pairs, max_batch_size, max_batch_frames):
    batches = []
    current = []
    current_max_frames = 0

    for pair in pairs:
        pair_frames = get_pair_frames(pair)
        next_max_frames = max(current_max_frames, pair_frames)
        next_padded_frames = next_max_frames * (len(current) + 1)
        exceeds_size = len(current) >= max_batch_size
        exceeds_frames = current and next_padded_frames > max_batch_frames
        if exceeds_size or exceeds_frames:
            batches.append(current)
            current = []
            current_max_frames = 0

        current.append(pair)
        current_max_frames = max(current_max_frames, pair_frames)

    if current:
        batches.append(current)
    return batches


def is_cuda_oom(exc):
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def is_fatal_cuda_state(exc):
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    fatal_markers = [
        "device not ready",
        "illegal memory access",
        "unspecified launch failure",
        "device-side assert",
    ]
    return any(marker in message for marker in fatal_markers)


def safe_cuda_cleanup(device):
    gc.collect()
    if device.type != "cuda":
        return
    try:
        torch.cuda.empty_cache()
    except RuntimeError as cleanup_exc:
        print(f"[Warning] CUDA cleanup failed after OOM: {cleanup_exc}")


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


def infer_one_audio(model, mix_audio, device, chunk_frames):
    estimates = []
    total_frames = len(mix_audio)
    if chunk_frames <= 0 or total_frames <= chunk_frames:
        mix_tensor = torch.from_numpy(mix_audio).to(device)
        with torch.inference_mode():
            estimate = model(mix_tensor[None]).squeeze(0).detach().cpu()
        del mix_tensor
        return estimate

    for start in range(0, total_frames, chunk_frames):
        end = min(start + chunk_frames, total_frames)
        chunk_tensor = torch.from_numpy(mix_audio[start:end]).to(device)
        with torch.inference_mode():
            chunk_estimate = model(chunk_tensor[None]).squeeze(0).detach().cpu()
        estimates.append(chunk_estimate)
        del chunk_tensor, chunk_estimate
        safe_cuda_cleanup(device)

    min_sources = min(estimate.shape[0] for estimate in estimates)
    estimates = [estimate[:min_sources] for estimate in estimates]
    return torch.cat(estimates, dim=-1)


def process_batch(batch_pairs, output_root, model, sample_rate, device, save_mix=True, print_metrics=False, chunk_frames=0):
    rows = []
    for scene_id, mixed_path, target_path in batch_pairs:
        mix, target, length = load_pair_audio(scene_id, mixed_path, target_path, sample_rate)
        mix = mix[:length]
        target = target[:length]

        estimate = infer_one_audio(model, mix, device, chunk_frames)
        length = min(length, estimate.shape[-1])
        mix_tensor = torch.from_numpy(mix[:length])
        target_tensor = torch.from_numpy(target[:length])
        estimate = estimate[:, :length]

        output_dir = output_root / scene_id
        wav_dir = output_dir / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)

        if save_mix:
            sf.write(wav_dir / "mix.wav", mix_tensor.numpy(), sample_rate)
        for idx, source in enumerate(estimate.numpy(), start=1):
            sf.write(wav_dir / f"estimate_s{idx}.wav", source, sample_rate)

        metrics_path = output_dir / "metrics.csv"
        best_idx = write_single_target_metrics(mix_tensor, target_tensor, estimate, "s1", str(metrics_path))
        sf.write(wav_dir / "best_s1.wav", estimate[best_idx - 1].numpy(), sample_rate)

        row = read_metric_row(metrics_path)
        row["scene_id"] = scene_id
        row["mix"] = str(mixed_path)
        row["target"] = str(target_path)
        row["output_dir"] = str(output_dir)
        rows.append(row)
        if print_metrics:
            print_metrics_file(str(metrics_path))

        del estimate, mix_tensor, target_tensor
        safe_cuda_cleanup(device)

    return rows


def read_scene_id_filter(path):
    if not path:
        return None
    filter_path = Path(path).expanduser()
    if not filter_path.exists():
        return set()

    scene_ids = set()
    with open(filter_path, "r", newline="") as f:
        sample = f.read(1024)
        f.seek(0)
        sample_lines = sample.splitlines()
        has_scene_id_header = bool(sample_lines) and "scene_id" in sample_lines[0]
        if has_scene_id_header:
            for row in csv.DictReader(f):
                scene_id = row.get("scene_id")
                if scene_id:
                    scene_ids.add(scene_id.strip())
        else:
            for line in f:
                scene_id = line.strip().split(",")[0]
                if scene_id:
                    scene_ids.add(scene_id)
    return scene_ids


def filter_pairs_by_scene_ids(pairs, include_ids=None, exclude_ids=None):
    filtered = []
    for pair in pairs:
        scene_id = pair[0]
        if include_ids is not None and scene_id not in include_ids:
            continue
        if exclude_ids is not None and scene_id in exclude_ids:
            continue
        filtered.append(pair)
    return filtered


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
    parser.add_argument("--batch_size", type=int, default=1, help="Maximum number of scenes per scheduling group. Use 1 for best stability on small GPUs.")
    parser.add_argument(
        "--max_batch_seconds",
        type=float,
        default=12.0,
        help="Maximum padded audio seconds per scheduling group. Lower this if CUDA OOM occurs.",
    )
    parser.add_argument(
        "--chunk_seconds",
        type=float,
        default=6.0,
        help="Split each file into this many seconds per model forward pass. Use 0 to disable chunking.",
    )
    parser.add_argument("--include_scene_ids", default=None, help="Optional file containing scene IDs to process, one per line or CSV with scene_id.")
    parser.add_argument("--exclude_scene_ids", default=None, help="Optional file containing scene IDs to skip, one per line or CSV with scene_id.")
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
    include_ids = read_scene_id_filter(args.include_scene_ids)
    exclude_ids = read_scene_id_filter(args.exclude_scene_ids)
    pairs = filter_pairs_by_scene_ids(pairs, include_ids=include_ids, exclude_ids=exclude_ids)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    print(f"Found {len(pairs)} matched pair(s) in: {scenes_dir}")
    print("Processing order: filename order from input directory.")
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
    summary_path = output_root / "summary_metrics.csv"
    failures_path = output_root / "failed_scenes.csv"

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.max_batch_seconds <= 0:
        raise ValueError("--max_batch_seconds must be > 0")
    if args.chunk_seconds < 0:
        raise ValueError("--chunk_seconds must be >= 0")

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
    max_batch_frames = int(args.max_batch_seconds * sample_rate)
    chunk_frames = int(args.chunk_seconds * sample_rate) if args.chunk_seconds > 0 else 0
    dynamic_batches = make_dynamic_batches(pending_pairs, args.batch_size, max_batch_frames)
    print(
        f"Processing {pending_total} pending scene(s) in {len(dynamic_batches)} scheduling group(s) "
        f"with max_group_size={args.batch_size}, max_group_seconds={args.max_batch_seconds}, "
        f"chunk_seconds={args.chunk_seconds}."
    )

    for batch_index, batch_pairs in enumerate(dynamic_batches, start=1):
        batch_label = f"{batch_pairs[0][0]}..{batch_pairs[-1][0]}" if len(batch_pairs) > 1 else batch_pairs[0][0]
        batch_frames = max(get_pair_frames(pair) for pair in batch_pairs)
        batch_seconds = len(batch_pairs) * batch_frames / sample_rate
        print(f"Group {batch_index}/{len(dynamic_batches)}: {batch_label} ({len(batch_pairs)} file(s), {batch_seconds:.1f} padded sec)")
        try:
            rows = process_batch(
                batch_pairs,
                output_root,
                model,
                sample_rate,
                device,
                save_mix=not args.no_save_mix,
                print_metrics=args.print_each_metrics,
                chunk_frames=chunk_frames,
            )
            summary_rows.extend(rows)
            write_summary(summary_path, summary_rows)
        except RuntimeError as exc:
            if is_fatal_cuda_state(exc):
                print(f"[Fatal CUDA] {batch_label}: {exc}")
                print("CUDA is not recoverable in this Python process. Progress has been written; restart the command to continue.")
                for scene_id, mixed_path, target_path in batch_pairs:
                    failures.append(
                        {
                            "scene_id": scene_id,
                            "mix": str(mixed_path),
                            "target": str(target_path),
                            "error": repr(exc),
                        }
                    )
                write_summary(summary_path, summary_rows)
                write_failures(failures_path, failures)
                raise SystemExit(1)
            if is_cuda_oom(exc):
                print(f"[OOM] {batch_label}: {exc}")
                safe_cuda_cleanup(device)
            else:
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
                        chunk_frames=chunk_frames,
                    )
                    summary_rows.extend(rows)
                    write_summary(summary_path, summary_rows)
                except Exception as item_exc:
                    if is_fatal_cuda_state(item_exc):
                        print(f"[Fatal CUDA] {scene_id}: {item_exc}")
                        failures.append(
                            {
                                "scene_id": scene_id,
                                "mix": str(mixed_path),
                                "target": str(target_path),
                                "error": repr(item_exc),
                            }
                        )
                        write_summary(summary_path, summary_rows)
                        write_failures(failures_path, failures)
                        raise SystemExit(1)
                    if is_cuda_oom(item_exc):
                        safe_cuda_cleanup(device)
                    print(f"[Error] {scene_id}: {item_exc}")
                    failures.append(
                        {
                            "scene_id": scene_id,
                            "mix": str(mixed_path),
                            "target": str(target_path),
                            "error": repr(item_exc),
                        }
                    )
                    write_failures(failures_path, failures)
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
            write_failures(failures_path, failures)

    write_summary(summary_path, summary_rows)
    write_failures(failures_path, failures)

    print(f"Done. Successful or skipped scenes: {len(summary_rows)}")
    print(f"Summary metrics: {summary_path}")
    if failures:
        print(f"Failures: {failures_path}")


if __name__ == "__main__":
    main()
