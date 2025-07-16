import torch


def RSCM(x, M, mu=1, alpha=0.01):
    b, n, d = x.shape
    cov = x.swapaxes(2, 1) @ torch.conj(x) / M
    cov = mu * alpha * torch.eye(d)[None] + (1 - alpha) * cov
    return cov


def KL(ahat, a):
    K = torch.linalg.pinv(ahat)
    v = .5 * (torch.real(torch.vmap(torch.trace)(K @ a)) - torch.linalg.slogdet(K @ a)[1] - a.shape[-1])
    return v.mean().item()


def RMSE(ahat, a):
    return torch.sum(torch.abs(ahat[:, 0, :] - a[:, 0, :]) ** 2, 1).mean()


def RMSE_VEC(ahat, a):
    return torch.sum(torch.abs(ahat[:, 0, :] - a[:, 0, :]) ** 2, 1)


def sample_from_spectrum(p, K):
    p = p / p.sum()
    indices = torch.multinomial(p, K, replacement=True)
    return indices


def get_init_spect2(scm, K, N, L=1000, power=1.0):
    grid_o = torch.linspace(0, 2 * torch.pi, L)
    n = torch.arange(N).unsqueeze(1)  # (N, 1)
    A = torch.exp(1j * n * grid_o)  # (N, L)
    spectrum = torch.abs(torch.sum(A.conj() * (scm @ A), dim=0))
    spectrum = spectrum.pow(power)

    spectrum = spectrum / spectrum.sum()

    indices = sample_from_spectrum(spectrum, K)
    quantized_f = indices / L * 2 * torch.pi
    return quantized_f


def toeplitz_from_first_row(X):
    B, D = X.shape
    i = torch.arange(D, device=X.device).view(-1, 1)  # shape (D, 1)
    j = torch.arange(D, device=X.device).view(1, -1)  # shape (1, D)
    idx = (j - i).abs()  # shape (D, D), values in [0, D-1]
    idx = idx.unsqueeze(0).expand(B, -1, -1)  # (B, D, D)
    b_idx = torch.arange(B, device=X.device).view(-1, 1, 1).expand(-1, D, D)  # (B, D, D)
    T = X[b_idx, idx]  # shape (B, D, D)
    return T


def quantize_indices(p, K):
    p = p / p.sum()
    F = torch.cumsum(p, dim=0)
    quantile_centers = (torch.arange(K, dtype=p.dtype) + 0.5) / K
    indices = torch.searchsorted(F, quantile_centers)
    return indices


def get_init_spect(scm, K, N, L=10000):
    grid_o = torch.linspace(0, 2 * torch.pi, L)
    n = torch.arange(N).unsqueeze(1)  # (N, 1)
    A = torch.exp(1j * n * grid_o)  # (N, L)

    spectrum = torch.abs(torch.sum(A.conj() * (scm @ A), dim=0))
    spectrum = spectrum / spectrum.sum()
    quantized_f = quantize_indices(spectrum, K) / L * 2 * torch.pi
    return quantized_f


def generate_toeplitz_basis(m):
    basis_matrices = []
    for g in range(1, m + 1):
        B = torch.zeros((m, m), dtype=torch.cfloat)
        for i in range(m):
            for k in range(m):
                if i - k == g - 1:
                    if g - 1 == 0:
                        B[i, k] = 1 + 1j
                    else:
                        B[i, k] = 1 - 1j
                elif k - i == g - 1:
                    B[i, k] = 1 + 1j
        basis_matrices.append(B)
    return basis_matrices


def compute_crb(R, n):
    """
    Compute the CRB for Toeplitz covariance matrices parameterized by the real and imaginary
    parts of the first row.

    Parameters:
        R (torch.Tensor): Toeplitz covariance matrix of shape (m, m).
        n (int): Number of independent samples.
        verbose (bool): If True, prints the 1/n constant.

    Returns:
        torch.Tensor: Sum of per-parameter CRBs (lower bound on MSE).
    """
    m = R.shape[0]
    P = m
    theta_dim = 2 * P - 1
    basis_matrices = generate_toeplitz_basis(P)
    R_inv = torch.linalg.inv(R)
    # Build derivatives dR/dθ
    dR_dtheta = []
    for i in range(P):
        dR_dtheta.append(basis_matrices[i].real)
    for i in range(1, P):
        dR_dtheta.append(1j * basis_matrices[i].imag)

    fim = torch.zeros((theta_dim, theta_dim), dtype=torch.complex64, device=R.device)

    # Slepian-Bangs formula
    for i in range(theta_dim):
        for j in range(theta_dim):
            dRi = dR_dtheta[i].to(torch.complex64)
            dRj = dR_dtheta[j].to(torch.complex64)
            term = R_inv @ dRi @ R_inv @ dRj
            fim[i, j] = torch.trace(term)

    fim_inv = torch.linalg.inv(fim)
    fim_inv = (1 / n) * fim_inv
    # Compute per-parameter CRB
    crb_params = torch.zeros(P, dtype=R.dtype, device=R.device)
    for i in range(P):
        if i == 0:
            crb_params[i] = fim_inv[i, i]
        else:
            crb_params[i] = fim_inv[i, i] + fim_inv[i + P - 1, i + P - 1]

    total_crb = torch.abs(crb_params).sum()
    return total_crb
