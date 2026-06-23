import argparse
import csv
import gc
import os
from datetime import datetime

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import resample_poly

import look2hear.models
from look2hear.metrics import MetricsTracker


def load_mono(path):
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


def resample_if_needed(audio, src_sr, target_sr, label):
    if src_sr == target_sr:
        return audio
    print(f"[Warning] {label} sample rate is {src_sr} Hz, resampling to {target_sr} Hz.")
    gcd = np.gcd(src_sr, target_sr)
    return resample_poly(audio, target_sr // gcd, src_sr // gcd).astype(np.float32)


def load_model(conf, device):
    train_conf = conf
    train_conf.setdefault("main_args", {})
    train_conf["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", train_conf["exp"]["exp_name"]
    )
    model_path = os.path.join(train_conf["main_args"]["exp_dir"], "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    model = getattr(look2hear.models, train_conf["audionet"]["audionet_name"]).from_pretrain(
        model_path,
        sample_rate=train_conf["datamodule"]["data_config"]["sample_rate"],
        **train_conf["audionet"]["audionet_config"],
    )
    model.to(device).eval()
    return model, train_conf["datamodule"]["data_config"]["sample_rate"]


def calc_sdr(estimate, target, eps=1e-8):
    return 10 * torch.log10((target ** 2).sum() / (((estimate - target) ** 2).sum() + eps) + eps)


def calc_si_snr(estimate, target, eps=1e-8):
    projection = (estimate * target).sum() * target / ((target ** 2).sum() + eps)
    noise = estimate - projection
    return 10 * torch.log10((projection ** 2).sum() / ((noise ** 2).sum() + eps) + eps)


def write_single_target_metrics(mix, target, estimates, target_label, save_file):
    scores = []
    baseline_sdr = calc_sdr(mix, target)
    baseline_si_snr = calc_si_snr(mix, target)
    for idx, estimate in enumerate(estimates, start=1):
        sdr = calc_sdr(estimate, target)
        si_snr = calc_si_snr(estimate, target)
        scores.append((idx, sdr, sdr - baseline_sdr, si_snr, si_snr - baseline_si_snr))

    best_idx, best_sdr, best_sdr_i, best_si_snr, best_si_snr_i = max(scores, key=lambda item: item[3])
    with open(save_file, "w") as f:
        f.write("target,best_estimate,sdr,sdr_i,si-snr,si-snr_i\n")
        f.write(
            f"{target_label},estimate_s{best_idx},{best_sdr.item()},{best_sdr_i.item()},"
            f"{best_si_snr.item()},{best_si_snr_i.item()}\n"
        )
    return best_idx


def print_metrics_file(metrics_path):
    with open(metrics_path, "r") as f:
        rows = list(csv.DictReader(f))
    row = next((item for item in rows if item.get("snt_id") == "avg"), rows[0])
    print("Metrics:")
    for key, value in row.items():
        if key in {"snt_id", "target", "best_estimate"}:
            continue
        print(f"  {key}: {float(value):.4f}")
    if "best_estimate" in row:
        print(f"  best_estimate: {row['best_estimate']}")


def safe_cuda_cleanup(device):
    gc.collect()
    if device.type != "cuda":
        return
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass


def infer_audio(model, mix, device, chunk_frames, sample_rate):
    total_frames = len(mix)
    if chunk_frames <= 0 or total_frames <= chunk_frames:
        mix_tensor = torch.from_numpy(mix).to(device)
        with torch.inference_mode():
            estimate = model(mix_tensor[None]).squeeze(0).detach().cpu()
        del mix_tensor
        safe_cuda_cleanup(device)
        return estimate

    estimates = []
    for start in range(0, total_frames, chunk_frames):
        end = min(start + chunk_frames, total_frames)
        print(f"Processing chunk {start // chunk_frames + 1}/{(total_frames + chunk_frames - 1) // chunk_frames}: {start / sample_rate:.1f}s-{end / sample_rate:.1f}s")
        chunk_tensor = torch.from_numpy(mix[start:end]).to(device)
        with torch.inference_mode():
            chunk_estimate = model(chunk_tensor[None]).squeeze(0).detach().cpu()
        estimates.append(chunk_estimate)
        del chunk_tensor, chunk_estimate
        safe_cuda_cleanup(device)

    min_sources = min(chunk_estimate.shape[0] for chunk_estimate in estimates)
    estimates = [chunk_estimate[:min_sources] for chunk_estimate in estimates]
    return torch.cat(estimates, dim=-1)

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", default=None, help="Path to mixed wav. If omitted, mix is created from --s1 + --s2.")
    parser.add_argument("--s1", default=None, help="Path to clean source 1 wav")
    parser.add_argument("--s2", default=None, help="Path to clean source 2 wav")
    parser.add_argument("--conf_dir", default="configs/spmamba-echo2mix.yml")
    parser.add_argument("--output_dir", "--output", default=None)
    parser.add_argument("--chunk_seconds", type=float, default=6.0, help="Split long audio into chunks for inference. Use 0 to disable chunking.")
    args = parser.parse_args()

    if args.mix is None and (args.s1 is None or args.s2 is None):
        parser.error("Without --mix, provide both --s1 and --s2 so the script can create a mixture.")

    print("conf_dir:", args.conf_dir)
    with open(args.conf_dir, "rb") as f:
        conf = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_sample_rate = load_model(conf, device)

    clean_tensor = None
    target_tensor = None
    target_label = None
    if args.mix:
        mix, mix_sr = load_mono(args.mix)
        mix = resample_if_needed(mix, mix_sr, model_sample_rate, "mix")
        refs = []
        if args.s1:
            s1, sr1 = load_mono(args.s1)
            s1 = resample_if_needed(s1, sr1, model_sample_rate, "s1")
            refs.append(("s1", s1))
        if args.s2:
            s2, sr2 = load_mono(args.s2)
            s2 = resample_if_needed(s2, sr2, model_sample_rate, "s2")
            refs.append(("s2", s2))
        if len(refs) == 2:
            length = min(len(mix), len(refs[0][1]), len(refs[1][1]))
            mix = mix[:length]
            clean_tensor = torch.from_numpy(np.stack([refs[0][1][:length], refs[1][1][:length]]))
        elif len(refs) == 1:
            target_label, target_audio = refs[0]
            length = min(len(mix), len(target_audio))
            mix = mix[:length]
            target_tensor = torch.from_numpy(target_audio[:length])
    else:
        s1, sr1 = load_mono(args.s1)
        s2, sr2 = load_mono(args.s2)
        s1 = resample_if_needed(s1, sr1, model_sample_rate, "s1")
        s2 = resample_if_needed(s2, sr2, model_sample_rate, "s2")
        length = min(len(s1), len(s2))
        s1, s2 = s1[:length], s2[:length]
        mix = s1 + s2
        clean_tensor = torch.from_numpy(np.stack([s1, s2]))

    output_dir = args.output_dir or os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = os.path.abspath(output_dir)
    wav_dir = os.path.join(output_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    chunk_frames = int(args.chunk_seconds * model_sample_rate) if args.chunk_seconds > 0 else 0
    if chunk_frames > 0:
        print(f"Using chunked inference: {args.chunk_seconds}s per chunk")

    mix_tensor = torch.from_numpy(mix)
    estimate = infer_audio(model, mix, device, chunk_frames, model_sample_rate)

    min_len = min(mix_tensor.shape[-1], estimate.shape[-1])
    if clean_tensor is not None:
        min_len = min(min_len, clean_tensor.shape[-1])
        clean_tensor = clean_tensor[:, :min_len]
    if target_tensor is not None:
        min_len = min(min_len, target_tensor.shape[-1])
        target_tensor = target_tensor[:min_len]
    mix_tensor = mix_tensor[:min_len]
    estimate = estimate[:, :min_len]

    sf.write(os.path.join(wav_dir, "mix.wav"), mix_tensor.cpu().numpy(), model_sample_rate)
    for idx, source in enumerate(estimate.numpy(), start=1):
        sf.write(os.path.join(wav_dir, f"estimate_s{idx}.wav"), source, model_sample_rate)

    print(f"Saved wav files to: {wav_dir}")
    if clean_tensor is not None:
        metrics_path = os.path.join(output_dir, "metrics.csv")
        metrics = MetricsTracker(save_file=metrics_path)
        metrics(mix=mix_tensor, clean=clean_tensor, estimate=estimate, key="input_pair")
        metrics.final()
        print(f"Saved metrics to: {metrics_path}")
        print_metrics_file(metrics_path)
    elif target_tensor is not None:
        metrics_path = os.path.join(output_dir, "metrics.csv")
        best_idx = write_single_target_metrics(mix_tensor, target_tensor, estimate, target_label, metrics_path)
        sf.write(os.path.join(wav_dir, f"best_{target_label}.wav"), estimate[best_idx - 1].numpy(), model_sample_rate)
        print(f"Saved single-target metrics to: {metrics_path}")
        print(f"Best estimate for {target_label}: estimate_s{best_idx}.wav")
        print_metrics_file(metrics_path)
    else:
        print("[Warning] No clean --s1/--s2 references provided; metrics.csv was not generated.")


if __name__ == "__main__":
    main()
