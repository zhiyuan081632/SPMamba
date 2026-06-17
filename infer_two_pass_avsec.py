#!/usr/bin/env python
import argparse
import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
INFER_SCRIPT = PROJECT_ROOT / "infer_batch_avsec.py"


def read_failed_scene_ids(path):
    failed_path = Path(path)
    if not failed_path.exists():
        return set()

    scene_ids = set()
    with open(failed_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        scene_id = row.get("scene_id")
        if scene_id:
            scene_ids.add(scene_id.strip())
    return scene_ids


def write_scene_id_file(path, scene_ids):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        for scene_id in sorted(scene_ids):
            f.write(f"{scene_id}\n")


def remove_if_exists(path):
    path = Path(path)
    if path.exists():
        path.unlink()


def build_common_args(args):
    common = [
        sys.executable,
        str(INFER_SCRIPT),
        "--scenes_dir",
        args.scenes_dir,
        "--output_root",
        args.output_root,
        "--conf_dir",
        args.conf_dir,
        "--mixed_suffix",
        args.mixed_suffix,
        "--target_suffix",
        args.target_suffix,
    ]
    if args.start_id is not None:
        common += ["--start_id", str(args.start_id)]
    if args.end_id is not None:
        common += ["--end_id", str(args.end_id)]
    if args.limit is not None:
        common += ["--limit", str(args.limit)]
    if args.no_save_mix:
        common.append("--no_save_mix")
    return common


def run_child(command, failures_path):
    remove_if_exists(failures_path)
    return subprocess.run(command, cwd=PROJECT_ROOT).returncode


def main():
    parser = argparse.ArgumentParser(description="Two-pass AVSEC inference: fast first, then conservative retry for failures.")
    parser.add_argument("--scenes_dir", default="/mnt/e/data/AVSEC/avsec3/scenes")
    parser.add_argument("--output_root", default="/mnt/e/data/AVSEC/avsec3/spmamba")
    parser.add_argument("--conf_dir", default="configs/spmamba-echo2mix.yml")
    parser.add_argument("--mixed_suffix", default="_mixed.wav")
    parser.add_argument("--target_suffix", default="_target.wav")
    parser.add_argument("--start_id", type=int, default=None)
    parser.add_argument("--end_id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_save_mix", action="store_true", default=True)

    parser.add_argument("--fast_batch_size", type=int, default=4)
    parser.add_argument("--fast_max_batch_seconds", type=float, default=45.0)
    parser.add_argument("--fast_chunk_seconds", type=float, default=0.0)
    parser.add_argument("--max_fast_restarts", type=int, default=80)

    parser.add_argument("--retry_batch_size", type=int, default=1)
    parser.add_argument("--retry_max_batch_seconds", type=float, default=12.0)
    parser.add_argument("--retry_chunk_seconds", type=float, default=3.0)
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    failures_path = output_root / "failed_scenes.csv"
    fast_failed_ids_path = output_root / "fast_failed_scene_ids.txt"

    common = build_common_args(args)
    failed_scene_ids = set()

    print("=== Pass 1: fast inference, skipping scenes that crash/fail ===")
    for restart_index in range(1, args.max_fast_restarts + 1):
        write_scene_id_file(fast_failed_ids_path, failed_scene_ids)
        fast_command = common + [
            "--batch_size",
            str(args.fast_batch_size),
            "--max_batch_seconds",
            str(args.fast_max_batch_seconds),
            "--chunk_seconds",
            str(args.fast_chunk_seconds),
            "--exclude_scene_ids",
            str(fast_failed_ids_path),
        ]
        print(f"Fast attempt {restart_index}/{args.max_fast_restarts}; excluded failed scenes: {len(failed_scene_ids)}")
        return_code = run_child(fast_command, failures_path)
        new_failed = read_failed_scene_ids(failures_path)
        before_count = len(failed_scene_ids)
        failed_scene_ids.update(new_failed)
        if len(failed_scene_ids) > before_count:
            write_scene_id_file(fast_failed_ids_path, failed_scene_ids)
            print(f"Recorded fast-pass failures: {len(failed_scene_ids)}")

        if return_code == 0:
            print("Fast pass completed without a fatal CUDA exit.")
            break
        if not new_failed or len(failed_scene_ids) == before_count:
            print("Fast pass stopped but no new failed scene IDs were recorded; stopping to avoid a loop.")
            return return_code
    else:
        print("Reached --max_fast_restarts before fast pass completed.")
        return 1

    if not failed_scene_ids:
        print("No failed scenes to retry.")
        return 0

    print(f"=== Pass 2: conservative retry for {len(failed_scene_ids)} failed scene(s) ===")
    retry_command = common + [
        "--batch_size",
        str(args.retry_batch_size),
        "--max_batch_seconds",
        str(args.retry_max_batch_seconds),
        "--chunk_seconds",
        str(args.retry_chunk_seconds),
        "--include_scene_ids",
        str(fast_failed_ids_path),
    ]
    return run_child(retry_command, failures_path)


if __name__ == "__main__":
    raise SystemExit(main())
