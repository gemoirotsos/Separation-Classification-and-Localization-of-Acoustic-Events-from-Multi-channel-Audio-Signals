# import libraries
import librosa 
import librosa.display
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from IPython.display import Audio
from scipy.io import wavfile
import warnings
warnings.filterwarnings("ignore")
from importlib import reload

# reload class IVA
import IVA_Class
reload(IVA_Class)
from IVA_Class import IVA

# path for mixture signals (Evaluation dataset from DCASE 2025 Task 4 Challenge)
BASE_DIR = Path.cwd().parents[2]
folder = BASE_DIR / "Dataset" / "Evaluation Dataset" / "DCASE2025Task4EvaluationDataset" / "eval_set" / "soundscape"
mixture = sorted(folder.glob("*.wav"))

# path to save separated signals (create the following folders)
output_dir = Path.home() / "BSS" / "IVA" / "separated_outputs" / "separated_outputs_aux_2048_1024"
output_dir.mkdir(parents=True, exist_ok=True)

costs = {}
separated_outputs = {}

# RUN auxIVA for N mixtures (i.e. N=100 stands for the first 100 mixtures from the evaluation set)
N = 100
for i in range(N):

    print(f"Running auxIVA for {mixture[i].name}")
    print()
    # auxIVA
    iva = IVA(
        n_fft=2048,
        hop_length=1024,
        window="hamming",
        num_iters=20,
        eps=1e-12
    )

    X = iva.compute_multichannel_STFT(mixture[i])
    W = iva.initialize_demixing_matrix(X)
    W_final, cost_history = iva.run_auxIVA()
    separated = iva.reconstruct_time_signals()

    costs[i] = cost_history

    # save the separated outputs in .wav file
    for q in range(separated.shape[1]):
        sig = separated[:, q]

        sig = sig - np.mean(sig)
        sig = 0.95 * sig / (np.max(np.abs(sig)) + 1e-12)

        filename = output_dir / f"{mixture[i].stem}_sep_output{q+1}_aux_20_2048_1024.wav"
        sf.write(filename, sig, iva.fs)

    print(f"End of {mixture[i].name}")
    print()

