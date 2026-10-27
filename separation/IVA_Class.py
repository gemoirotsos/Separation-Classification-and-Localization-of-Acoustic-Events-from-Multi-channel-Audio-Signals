# import libraries
import soundfile as sf
from scipy.signal import stft, istft
import matplotlib.pyplot as plt
import numpy as np
import math

class IVA:
    def __init__(self, n_fft=2048, hop_length=None, window="hamming", learning_rate=None, factor=0.001, alpha=2, num_iters=100, beta=0.2, eps=1e-12):
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 2
        self.window = window
        self.learning_rate = None
        self.num_iters = num_iters
        self.eps = eps

        self.fs = None
        self.X = None
        self.W = None
        self.Y = None
        self.Y_vectors = None
        self.r = None
        self.J_IVA = None
        self.separated_signals = None
        self.cost_history = []
        self.gamma = 1
        self.beta = beta
        self.factor = factor
        self.alpha = alpha


    # Step 1 - STFT grad
    def compute_multichannel_STFT_grad(self, x, fs):
        if x.ndim == 1:
            x = x[np.newaxis, :]

        # Αν έρθει ως (samples, channels), κάν' το (channels, samples)
        if x.shape[0] > x.shape[1]:
            x = x.T

        self.fs = fs

        num_channels, num_samples = x.shape

        X_list = []
        
        for ch in range(num_channels):
            f, t, Z = stft(
                x[ch, :],
                fs=fs,
                window=self.window,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length,
                nfft=self.n_fft,
                boundary=None,
                padded=False
            )

            X_list.append(Z)
    
        self.X = np.stack(X_list, axis=0)

        return self.X

    # Step 1 - STFT aux
    def compute_multichannel_STFT(self, audio):
        x, fs = sf.read(audio)

        if x.ndim == 1:
            x = x[:, np.newaxis]

        self.fs = fs
        num_samples, num_channels = x.shape
        X_list = []
        
        for ch in range(num_channels):
            f, t, Z = stft(
                x[:,ch],
                fs=fs,
                window=self.window,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length,
                nfft=self.n_fft,
                boundary=None,
                padded=False
            )

            X_list.append(Z)
    
        self.X = np.stack(X_list, axis=0)

        return self.X
    
    # Step 2 - Initialize W
    def initialize_demixing_matrix(self, X):
        num_channels, num_bins, num_frames = X.shape

        self.W = np.zeros((num_bins, num_channels, num_channels), dtype=np.complex128)

        for k in range(num_bins):
            self.W[k] = np.eye(num_channels, dtype=np.complex128)
            
        return self.W
    
    # Step 3 - Demixing Operation
    def compute_output_signal_vector(self, X, W):
        num_channels, num_bins, num_frames = X.shape

        self.Y = np.zeros((num_channels,num_bins,num_frames), dtype=np.complex128)
        for k in range(num_bins):
            for n in range(num_frames):
                self.Y[:, k, n] = W[k] @ X[:,k,n]

        return self.Y
    
    # Step 4 - Frequency Vector y_q
    def get_frequency_vector(self):
        self.Y_vectors = np.transpose(self.Y,(0,2,1))
        return self.Y_vectors
    
    # Step 5 - Capture Statistical Dependencies
    def compute_laplacian_radii(self):
        self.r = np.sqrt(np.sum(np.abs(self.Y_vectors)**2, axis=2) + self.eps)

        return self.r
    
    # Step 6 - Cost Function grad
    def Cost_Function_grad(self):
    
        source_term = np.sum(np.mean((self.r + self.eps), axis=1))

        det_W = np.linalg.det(self.W)
        log_determinant_term = -2.0 * np.sum(np.log(np.abs(det_W) + self.eps))
        
        self.J_IVA = source_term + log_determinant_term
        self.J_IVA = np.real_if_close(self.J_IVA).item()

        return self.J_IVA
    
    # Step 6 - Cost Function aux
    def Cost_Function_aux(self):
    
        source_term = np.sum(np.mean((self.r + self.eps)**self.beta, axis=1))

        det_W = np.linalg.det(self.W)
        log_determinant_term = -2.0 * np.sum(np.log(np.abs(det_W) + self.eps))
        
        self.J_IVA = source_term + log_determinant_term
        self.J_IVA = np.real_if_close(self.J_IVA).item()

        return self.J_IVA
    
    # iterative factor improvement - for grad
    def iterative_factor(self, k):
        self.learning_rate = self.factor / (np.log(k * self.alpha) / np.log(self.alpha))

        return self.learning_rate
    
    # Step 7 - Update W using gradIVA
    def update_W_gradIVA(self):
    
        num_channels, num_bins, num_frames = self.X.shape

        W_new = self.W.copy()
        I = np.eye(num_channels, dtype=np.complex128)

        for k in range(num_bins):
            A_k = np.zeros((num_channels,num_channels), dtype=np.complex128)

            for n in range(num_frames):
                y_kn = self.Y[:,k,n]
                phi_kn = y_kn / (self.r[:,n] + self.eps)
                A_k += np.outer(phi_kn, np.conj(y_kn))
        
            A_k /= num_frames

            W_new[k] = self.W[k] + self.learning_rate * (I - A_k) @ self.W[k]

        self.W = W_new
        return self.W
    
    # Step 7 - Update W using auxIVA
    def update_W_auxIVA(self):
        num_channels, num_bins, num_frames = self.X.shape
        W_new = self.W.copy()
        phi0 = 1000.0
        reg = 1e-6

        for k in range(num_bins):
            X_k = self.X[:, k, :]   # (C, N)
            W_k = W_new[k].copy()
            

            for q in range(num_channels):
                U_q = np.zeros((num_channels, num_channels), dtype=np.complex128)

                for n in range(num_frames):
                    x_kn = X_k[:, n]
                    weight = self.gamma * (self.r[q, n] + self.eps) **(self.beta-2)
                    weight = np.minimum(weight, phi0)
                    U_q += weight * np.outer(x_kn, np.conj(x_kn))
                   
                
                U_q /= num_frames
                U_q += reg * np.eye(num_channels, dtype=np.complex128)

                e_q = np.zeros(num_channels, dtype=np.complex128)
                e_q[q] = 1.0 + 0j

                V = W_k @ U_q
                V += reg * np.eye(num_channels, dtype=np.complex128)
                w_q = np.linalg.solve(V, e_q)

                denom = np.sqrt(
                    np.maximum(
                        np.real(np.conj(w_q).T @ U_q @ w_q),
                        self.eps
                    )
                )
                
                w_q = w_q / denom

                W_k[q, :] = np.conj(w_q)
            
            W_new[k] = W_k

        self.W = W_new
        return self.W
    
    # Step 8 - Run using gradIVA
    def run_gradIVA(self):
        
        self.cost_history = []

        for i in range(self.num_iters):
            self.Y = self.compute_output_signal_vector(self.X,self.W)
            # Step 4
            self.Y_vectors = self.get_frequency_vector()
            # Step 5
            self.r = self.compute_laplacian_radii()
            # Step 6
            cost = self.Cost_Function_grad()
            cost = np.real(cost)
            self.cost_history.append(cost)

            # learning rate
            k_iter = i + 1
            self.learning_rate = self.iterative_factor(k_iter)
            # Step 7
            self.W = self.update_W_gradIVA()

            print(f"Iteration {k_iter}/{self.num_iters}, Cost = {cost:.10f}")

        self.Y = self.compute_output_signal_vector(self.X, self.W)
        self.Y_vectors = self.get_frequency_vector()
        self.r = self.compute_laplacian_radii()
    
        return self.W, self.cost_history
    
    # Step 8 - Run using auxIVA
    def run_auxIVA(self):
        self.cost_history = []

        for i in range(self.num_iters):
            self.Y = self.compute_output_signal_vector(self.X,self.W)
            # Step 4
            self.Y_vectors = self.get_frequency_vector()
            # Step 5
            self.r = self.compute_laplacian_radii()
            # Step 6
            cost = self.Cost_Function_aux()
            cost = np.real(cost)
            self.cost_history.append(cost)
            # Step 7
            self.W = self.update_W_auxIVA()
            #scale = np.sqrt(np.mean(self.r ** 2) + self.eps)
            #self.W = self.W / scale

            print(f"Iteration {i+1}/{self.num_iters}, Cost = {cost:.10f}")

        self.Y = self.compute_output_signal_vector(self.X, self.W)
        self.Y_vectors = self.get_frequency_vector()
        self.r = self.compute_laplacian_radii()
    
        return self.W, self.cost_history

    
    # Step 9 - ISTFT
    def reconstruct_time_signals(self):
        if self.hop_length is None:
            self.hop_length = self.n_fft // 2

        num_channels, num_bins, num_frames = self.Y.shape
        reconstructed = []

        for q in range(num_channels):
            _, y_time = istft(
                self.Y[q, :, :],
                fs=self.fs,
                window=self.window,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length,
                nfft=self.n_fft,
                input_onesided=True,
                boundary=None
            )
            reconstructed.append(np.real(y_time))

        # Make all channels same length
        min_len = min(len(sig) for sig in reconstructed)
        reconstructed = [sig[:min_len] for sig in reconstructed]

        self.separated_signals = np.stack(reconstructed, axis=1)  # (samples, channels)
        return self.separated_signals