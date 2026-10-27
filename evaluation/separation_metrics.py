# libraries
from pathlib import Path, os
from scipy.io import wavfile
import numpy as np
import torch
from torchmetrics.audio import SignalDistortionRatio, SignalNoiseRatio
import mir_eval
import warnings
warnings.filterwarnings("ignore")

## FUNCTIONS

# SDR
def SDR(target, estimate):
    target_source = torch.tensor(target, dtype=torch.float32)
    estimated_source = torch.tensor(estimate, dtype=torch.float32)

    metric = SignalDistortionRatio()
    sdr = metric(estimated_source, target_source)

    print(f"SDR = {sdr.item():.2f} dB")
    return sdr

# SNR
def SNR(target, estimate):
    target_source = torch.tensor(target, dtype=torch.float32)
    estimated_source = torch.tensor(estimate, dtype=torch.float32)

    metric = SignalNoiseRatio()
    snr = metric(estimated_source, target_source)

    print(f"SNR = {snr.item():.2f} dB")
    return snr

# SAR
def SAR(target, estimate):

    _, _, sar, _ = mir_eval.separation.bss_eval_sources(
    target,
    estimate
    )

    print(f"SAR = {sar.item():.2f} dB")
    return sar


# prepare data for separation metrics
def process_data(target_source, estimated_path):

    BASE_DIR = Path.cwd().parents[2]
    ground_truth_folder = BASE_DIR / "Dataset" / "Evaluation Grpund Truth" / "DCASE2025Task4EvaluationGroundTruth" / "eval_set" / "oracle_target"
    
    target_path = os.path.join(
    ground_truth_folder,
    target_source
    )
    
    fs_target, target = wavfile.read(target_path)
    fs_est, estimated = wavfile.read(estimated_path)

    # Convert to mono if stereo
    if target.ndim > 1:
        target = target[:, 0]

    if estimated.ndim > 1:
        estimated = estimated[:, 0]

    min_len = min(len(target), len(estimated))

    target = target[:min_len]
    estimated = estimated[:min_len]


    # Convert to float
    target = target.astype(np.float64)
    estimated = estimated.astype(np.float64)


    # Remove DC offset
    target = target - np.mean(target)
    estimated = estimated - np.mean(estimated)

    return target, estimated