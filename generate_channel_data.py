import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / 'tools'
OUT_DIR.mkdir(parents=True, exist_ok=True)

L = 16
N_TRAIN = 100_000
N_TEST = 390_000

pdp = np.exp(-np.arange(L) / 3.0).astype(np.float32)
pdp = pdp / pdp.sum()
std = np.sqrt(pdp / 2.0).astype(np.float32)

def make(n, seed):
    rng = np.random.default_rng(seed)
    real = rng.normal(size=(n, L)).astype(np.float32) * std[None, :]
    imag = rng.normal(size=(n, L)).astype(np.float32) * std[None, :]
    return (real + 1j * imag).astype(np.complex64)

np.save(OUT_DIR / 'channel_train.npy', make(N_TRAIN, 1))
np.save(OUT_DIR / 'channel_test.npy', make(N_TEST, 2))
print('generated replacement channel_train.npy and channel_test.npy')
