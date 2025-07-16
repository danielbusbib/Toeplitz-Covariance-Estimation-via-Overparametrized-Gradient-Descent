import torch
import numpy as np
from scipy.linalg import sqrtm
import numpy


# ATOM Data
def atom_data(num_test, N, d=15, on_grid_freq=True):
    torch.manual_seed(0)
    np.random.seed(0)
    # Given frequencies and amplitudes
    omegas = torch.tensor(
        [0.2167, 0.6500, 1.0833, 1.3, 1.5166, 1.9500, 2.3833, 2.8166, 3.2499, 3.6832, 4.1166, 4.5499, 4.9832, 5.4165,
         5.8499][:d]).unsqueeze(0)
    if not on_grid_freq:
        omegas[:, 3] = 1.25
        omegas[:, 7] = 3.01
        omegas[:, 12] = 5.20
        omegas[:, 14] = 5.8
    amplitudes = torch.sqrt(torch.tensor(list(range(1, d + 1)), dtype=torch.cfloat).unsqueeze(0))
    grid_t = torch.arange(d)
    z = amplitudes[:, :, None] * torch.exp(1j * (grid_t[None, None, :] * omegas[:, :, None]))
    cn = np.sqrt(.5) * torch.randn(num_test, N, d) + 1j * np.sqrt(.5) * torch.randn(num_test, N, d)
    data = torch.einsum('esf,efd->esd', cn, z)
    cov = torch.einsum('efd,efr->edr', z, z.conj())
    return data, cov.repeat(num_test, 1, 1), omegas


# Random Toeplitz
def generate_data_toplitz_batch(num_examples=1, num_samples=20, dimension=6, num_freq=6, sigma2_noise=0.15,
                                ret_f=False):
    torch.manual_seed(1)
    omegas = torch.sort(torch.rand(1, num_freq))[0] * 2 * np.pi
    amplitudes = torch.sort(torch.rand(1, num_freq) * 2.5)[0]
    grid_t = torch.arange(dimension)
    z = amplitudes[:, :, None] * torch.exp(1j * (grid_t[None, None, :] * omegas[:, :, None]))
    cn = np.sqrt(.5) * torch.randn(num_examples, num_samples, num_freq) + 1j * np.sqrt(.5) * torch.randn(num_examples,
                                                                                                         num_samples,
                                                                                                         num_freq)
    noise = np.sqrt(.5) * torch.randn(num_examples, num_samples, dimension) + 1j * np.sqrt(.5) * torch.randn(
        num_examples, num_samples, dimension)
    data = torch.einsum('esf,efd->esd', cn, z) + sigma2_noise * noise
    cov = torch.einsum('efd,efr->edr', z, z.conj()) + sigma2_noise ** 2 * torch.eye(dimension)[None, :, :]
    if ret_f:
        return data, cov, omegas.squeeze(0)
    return data, cov


# AR Data
def TriaToepMulShort(c, P, rev):
    """
    Compute the matrix multiplication of a triangular Toeplitz matrix
    with its transpose.

    Parameters:
        c (numpy.ndarray): First column of the Toeplitz matrix.
        P (int): Matrix dimension.
        rev (bool): If True, reverse computation is applied.

    Returns:
        numpy.ndarray: Resulting matrix.
    """
    if not rev:
        M = np.zeros((P, P), dtype=np.complex128)
        for i in range(len(c)):
            for j in range(len(c)):
                if i == 0:
                    M[i, i + j] = c[i] * c[i + j]
                    M[i + j, i] = M[i, i + j]
                else:
                    M[i, min(i + j, len(c) - 1)] = M[i - 1, min(i + j - 1, len(c) - 2)] + c[i] * c[
                        min(i + j, len(c) - 1)]
                    M[min(i + j, len(c) - 1), i] = M[i, min(i + j, len(c) - 1)]
        for i in range(len(c), P):
            for j in range(len(c)):
                M[min(i - j, P - 1), i] = M[min(i - j - 1, P - 2), i - 1]
                M[i, min(i - j, P - 1)] = M[min(i - j, P - 1), i]
        return M
    else:
        cflip = np.flip(c[1:])
        M = np.zeros((P, P), dtype=np.complex128)
        Mfull = np.zeros((len(c) - 1, len(c) - 1), dtype=np.complex128)
        for i in range(len(c) - 1):
            for j in range(i, len(c) - 1):
                if i == 0:
                    Mfull[i, j] = cflip[i] * cflip[j]
                    Mfull[j, i] = Mfull[i, j]
                else:
                    Mfull[i, j] = Mfull[i - 1, j - 1] + cflip[i] * cflip[j]
                    Mfull[j, i] = Mfull[i, j]
        M[-len(c) + 1:, -len(c) + 1:] = Mfull
        return M


def gen_Gamma_varA(alpha, P):
    """
    Generate the precision matrix Gamma.

    Parameters:
        alpha (numpy.ndarray): Vector alpha.
        P (int): Dimension.

    Returns:
        numpy.ndarray: Precision matrix Gamma.
    """
    B_m = TriaToepMulShort(alpha, P, False)
    C_m = TriaToepMulShort(alpha, P, True)
    return (1 / alpha[0]) * (B_m - C_m)


def generate_AR_cov(N, sigma, a):
    a = np.asarray(a)
    p = len(a)
    B = np.eye(p)
    B = B[:-1, :]
    B = np.vstack([a, B])

    eigvals = np.linalg.eigvals(B)
    if np.all(np.abs(eigvals) < 1):
        print("stable AR process")

    # Create precision matrix via Gohberg-Semencul
    a0 = 1 / (sigma ** 2)
    ar = -a * a0
    alpha = np.concatenate([[a0], ar])
    G = gen_Gamma_varA(alpha, N)
    C = np.linalg.inv(G)
    return C


def generate_ar1(num_examples=1, M=None, dimension=10):
    torch.manual_seed(1)
    np.random.seed(1)
    a = [0.5, 0.2, 0.05]
    MM = generate_AR_cov(dimension, 0.8, a)
    Msqrt = sqrtm(MM)  # Matrix square root of the covariance matrix
    num_samples = M
    X0 = np.random.randn(num_examples, num_samples, dimension)  # Base normal samples
    data = X0 @ Msqrt.T  # Apply square root of covariance matrix to generate data
    data = torch.tensor(data.astype(np.complex128), dtype=torch.complex64)
    true_covariances = torch.tensor(MM, dtype=torch.complex64).repeat(num_examples, 1, 1)
    return data, true_covariances
