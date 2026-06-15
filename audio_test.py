###
# Author: Kai Li
# Date: 2021-06-21 23:29:31
# LastEditors: Please set LastEditors
# LastEditTime: 2022-09-26 11:14:20
###

import os
import random
from typing import Union
import soundfile as sf
import torch
import yaml
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from scipy.io import wavfile
import warnings
# import torchaudio
warnings.filterwarnings("ignore")
import look2hear.models
import look2hear.datas
from look2hear.metrics import MetricsTracker
from look2hear.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


parser = argparse.ArgumentParser()
parser.add_argument("--conf_dir",
                    default="local/mixit_conf.yml",
                    help="Full path to save best validation model")
parser.add_argument("--output_dir",
                    default=None,
                    help="Directory to save metrics.csv and wav outputs. Defaults to output/<timestamp>.")


compute_metrics = ["si_sdr", "sdr"]

def main(config):
    metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
    progress = Progress(
        TextColumn("[bold blue]Testing", justify="right"),
        BarColumn(bar_width=None),
        "•",
        BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress), 
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        metricscolumn
    )
    # import pdb; pdb.set_trace()
    train_conf = config["train_conf"]
    train_conf.setdefault("main_args", {})
    train_conf["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", train_conf["exp"]["exp_name"]
    )
    model_path = os.path.join(train_conf["main_args"]["exp_dir"], "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}. Run training first or pass the saved training conf.yml from an existing experiment."
        )
    # import pdb; pdb.set_trace()
    # conf["train_conf"]["masknet"].update({"n_src": 2})
    model =  getattr(look2hear.models, train_conf["audionet"]["audionet_name"]).from_pretrain(
        model_path,
        sample_rate=train_conf["datamodule"]["data_config"]["sample_rate"],
        **train_conf["audionet"]["audionet_config"],
    )
    device = torch.device(
        "cuda" if train_conf["training"].get("gpus") and torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    model_device = next(model.parameters()).device
    datamodule: object = getattr(look2hear.datas, train_conf["datamodule"]["data_name"])(
        **train_conf["datamodule"]["data_config"]
    )
    try:
        datamodule.setup("test")
    except TypeError:
        datamodule.setup()
    _, _ , test_set = datamodule.make_sets
   
    # Randomly choose the indexes of sentences to save.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ex_save_dir = config.get("output_dir") or os.path.join("output", timestamp)
    ex_save_dir = os.path.abspath(ex_save_dir)
    os.makedirs(ex_save_dir, exist_ok=True)
    audio_save_dir = os.path.join(ex_save_dir, "audio")
    os.makedirs(audio_save_dir, exist_ok=True)
    sample_rate = train_conf["datamodule"]["data_config"]["sample_rate"]
    metrics = MetricsTracker(
        save_file=os.path.join(ex_save_dir, "metrics.csv"))
    torch.no_grad().__enter__()
    with progress:
        for idx in progress.track(range(len(test_set))):
            # Forward the network on the mixture.
            mix, sources, key = tensors_to_device(test_set[idx],
                                                    device=model_device)
            est_sources = model(mix[None])
            mix_np = mix
            sources_np = sources
            est_sources_np = est_sources.squeeze(0)
            audio_key = os.path.splitext(os.path.basename(key))[0]
            est_sources_audio = est_sources_np.detach().cpu().numpy()
            if est_sources_audio.ndim == 1:
                sf.write(os.path.join(audio_save_dir, f"{audio_key}.wav"), est_sources_audio, sample_rate)
            else:
                for src_idx, est_source in enumerate(est_sources_audio, start=1):
                    sf.write(os.path.join(audio_save_dir, f"{audio_key}_s{src_idx}.wav"), est_source, sample_rate)
            metrics(mix=mix_np,
                    clean=sources_np,
                    estimate=est_sources_np,
                    key=key)
            if idx % 50 == 0:
                metricscolumn.update(metrics.update())
    metrics.final()


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf
    # print(arg_dic)
    main(arg_dic)
