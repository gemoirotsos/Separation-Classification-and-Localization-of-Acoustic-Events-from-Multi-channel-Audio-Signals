# libraries
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# path
BASE_DIR = Path.cwd().parents[2]

folder = BASE_DIR / "Dataset" / "Evaluation Dataset" / "DCASE2025Task4EvaluationDataset" / "eval_set" / "soundscape"
mixture = sorted(folder.glob("*.wav"))

# FUNCTIONS
def reconstruct_source_images(W, Y):
    """
    W: (K, M, M)
       Demixing matrices

    Y: (M, K, N)
       AuxIVA separated STFT outputs

    Returns
    -------
    source_images: (M, M, K, N)

    source_images[q, m, k, n]
        contribution of separated source q
        at spatial channel m.
    """

    num_sources, num_bins, num_frames = Y.shape
    num_channels = W.shape[1]

    source_images = np.zeros(
        (num_sources, num_channels, num_bins, num_frames),
        dtype=np.complex128
    )

    for k in range(num_bins):

        # Estimated mixing matrix
        A_k = np.linalg.pinv(W[k])

        for q in range(num_sources):

            # Spatial signature of source q
            a_q = A_k[:, q]

            # Source q at this frequency
            y_q = Y[q, k, :]

            source_images[q, :, k, :] = (
                a_q[:, np.newaxis] *
                y_q[np.newaxis, :]
            )

    return source_images

def source_channel_energy(source_images):

    # sum over frequency and time
    energies = np.mean(
        np.abs(source_images)**2,
        axis=(2, 3)
    )

    return energies

def compute_source_covariances(source_images):
    Q, M, K, N = source_images.shape

    R = np.zeros(
        (Q,K,M,M), 
        dtype=np.complex128
    )

    for q in range(Q):
        for k in range(K):

            S_qk = source_images[q, :,k, :]

            R[q,k] = (
                S_qk @ S_qk.conj().T
            ) / N

    return R

# DOA estimation: Returns azimuth and elevation coordinates
def estimate_doa(source_images, q):
    """
    Ch1=W, Ch2=Y, Ch3=Z, Ch4=X
    """

    S = source_images[q]   # shape (4, K, N)

    W = S[0]
    Y = S[1]
    Z = S[2]
    X = S[3]

    Ix = np.real(np.conj(W) * X)
    Iy = np.real(np.conj(W) * Y)
    Iz = np.real(np.conj(W) * Z)

    Ix_mean = np.sum(Ix)
    Iy_mean = np.sum(Iy)
    Iz_mean = np.sum(Iz)

    azimuth = np.degrees(
        np.arctan2(Iy_mean, Ix_mean)
    )

    elevation = np.degrees(
        np.arctan2(
            Iz_mean,
            np.sqrt(Ix_mean**2 + Iy_mean**2)
        )
    )

    return azimuth, elevation

# load class IVA
iva_folder = BASE_DIR / "Blind Source Separation" / "IVA" / "coding"
sys.path.insert(0, str(iva_folder))
from importlib import reload

import IVA_Class
reload(IVA_Class)
from IVA_Class import IVA

# RUN the IVA algorithm
# run AuxIVA for the first mixture from the evaluation set
mix = mixture[0] 
for i in range(1):

    print(f"Running auxIVA for {mix.name}")
    print()
    # auxIVA
    iva = IVA(
        n_fft=2048,
        hop_length=1024,
        window="hamming",
        num_iters=20,
        eps=1e-12
    )

    X = iva.compute_multichannel_STFT(mix)
    W = iva.initialize_demixing_matrix(X)
    W_final, cost_history = iva.run_auxIVA()

    print(f"End of {mix}")
    print()

    break


source_images = reconstruct_source_images(
    W_final,
    iva.Y
)

energies = source_channel_energy(source_images)

normalized_energy = (
    energies /
    np.sum(energies, axis=1, keepdims=True)
)

normalized_energy *= 100

# eval_0000, Sources: Buzzer, Typing
q_buzzer = 0
q_typing = 3

print( "Buzzer channel energies:", normalized_energy[q_buzzer])

strongest = np.argmax(normalized_energy[q_buzzer]) + 1

print("Strongest spatial channel:",strongest)

print("typing channel energies:", normalized_energy[q_typing])

strongest = np.argmax( normalized_energy[q_typing]) + 1

print("Strongest spatial channel:", strongest)

# estimate DOAs (θ,φ) for eval_0000 (sources "Buzzer" and "Typing")
az_buzzer, el_buzzer = estimate_doa(
    source_images,
    q=0
)

az_typing, el_typing = estimate_doa(
    source_images,
    q=3
)

print("buzzer azimuth:", az_buzzer)
print("buzzer elevation:", el_buzzer)
print()
print("typing azimuth:", az_typing)
print("typing elevation:", el_typing)

labels = ["Buzzer", "Dishes"]
azimuth = [az_buzzer, az_typing]
elevation = [el_buzzer, el_typing]

fig, ax = plt.subplots(figsize=(12, 5))

for lab, az, el in zip(labels, azimuth, elevation):
    ax.scatter(az, el, s=90, label=lab)
    ax.annotate(f"{lab}\n({az:.1f}°, {el:.1f}°)",
                (az, el),
                xytext=(8, 8),
                textcoords="offset points")

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax.set_xlabel("Azimuth (°)")
ax.set_ylabel("Elevation (°)")
ax.set_title("Estimated DOA of separated sources")
ax.grid(True, alpha=0.3)
ax.legend(title="Estimated sources")

plt.show()