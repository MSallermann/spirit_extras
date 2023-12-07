from spirit import constants
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class htst_quantities:
    hessian_min_2n = None
    det_min = None
    evals_min = None
    basis_min = None

    hessian_sp_2n = None
    det_sp = None
    basis_sp = None
    evals_sp = None
    q = None


def get_htst_quantities(
    p_state,
    file_min: Path,
    file_sp: Path,
    workdir: Path,
    triplet_format=False,
):
    from spirit import io

    workdir.mkdir(exist_ok=True, parents=True)

    res = htst_quantities()

    # ---------------
    # Minimum
    # ---------------
    io.image_read(p_state, file_min.as_posix())

    # Write Hessian
    io.write_hessian_geodesic(
        p_state,
        (workdir / "hessian_min_2n.txt").as_posix(),
        triplet_format=triplet_format,
    )
    res.hessian_min_2n = np.loadtxt(workdir / "hessian_min_2n.txt")
    res.det_min = np.linalg.det(res.hessian_min_2n)
    res.evals_min, evecs = np.linalg.eigh(res.hessian_min_2n)

    # Write basis at minimum
    io.write_basis(
        p_state,
        (workdir / "basis_min.txt").as_posix(),
        triplet_format=triplet_format,
    )
    res.basis_min = np.loadtxt(workdir / "basis_min.txt")

    # ---------------
    # Saddlepoint
    # ---------------
    io.image_read(p_state, file_sp.as_posix())

    # Write Hessian
    io.write_hessian_geodesic(
        p_state,
        (workdir / "hessian_sp_2n.txt").as_posix(),
        triplet_format=triplet_format,
    )
    res.hessian_sp_2n = np.loadtxt(workdir / "hessian_sp_2n.txt")
    res.evals_sp, evecs = np.linalg.eigh(res.hessian_sp_2n)
    res.det_sp = np.prod(res.evals_sp[1:])

    # Write basis
    io.write_basis(
        p_state,
        (workdir / "basis_sp.txt").as_posix(),
        triplet_format=triplet_format,
    )
    res.basis_sp = np.loadtxt(workdir / "basis_sp.txt")
    res.q = res.basis_sp @ evecs[:, 0]

    return res


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def delta(i, j):
    return 1 if i == j else 0


def compute_a_vector(q, s, H, alpha, mu, T):
    gamma_prime = constants.gamma / ((1 + alpha**2) * constants.mu_B * mu)

    nos = int(H.shape[0] / 3)
    a = np.zeros(3 * nos)
    for i in range(nos):
        ai = np.zeros(3)
        for j in range(nos):
            qj = q[3 * j : 3 * j + 3]
            sj = s[3 * j : 3 * j + 3]
            Hij = H[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
            ai += Hij @ np.cross(sj, qj)
        a[3 * i, 3 * i + 3] = ai
    return gamma_prime * a


def compute_b_vector(q, s, alpha, mu, T):
    gamma_prime = constants.gamma / ((1 + alpha**2) * constants.mu_B * mu)
    D = alpha * constants.k_B * T * constants.mu_B * mu / constants.gamma
    nos = int(s.shape[0] / 3)

    b = np.zeros(3 * nos)

    for i in range(nos):
        qi = q[3 * i : 3 * i + 3]
        si = s[3 * i : 3 * i + 3]
        b[3 * i : 3 * i + 3] = np.cross(si, qi) + alpha * qi

    return -gamma_prime * np.sqrt(2 * D) * b


def compute_Q_matrix(q, s, alpha, mu, T):
    gamma_prime = constants.gamma / ((1 + alpha**2) * constants.mu_B * mu)
    D = alpha * constants.k_B * T * constants.mu_B * mu / constants.gamma
    nos = int(s.shape[0] / 3)

    Q = np.zeros(shape=(3 * nos, 3 * nos))

    for i in range(nos):
        qi = q[3 * i : 3 * i + 3]
        Q[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = skew(qi)

    return -gamma_prime * np.sqrt(2 * D) * Q


def compute_dyn_contribution(q, s, H_2N, basis, alpha, mu, T):
    a = compute_a_vector(q, s, basis.T @ H_2N @ basis, alpha, mu)
    b = compute_b_vector(q, s, alpha, mu, T)
    Q = compute_Q_matrix(q, s, alpha, mu)

    # covariance matrix of boltzmann distribution projected to 3N space
    Sigma = constants.k_B * T * basis.T @ np.linalg.inv(H_2N) @ basis

    variance = a.T @ Sigma @ a + b.T @ b - np.trace(Sigma @ Q @ Q)

    dyn_factor = np.sqrt(variance / (2.0 * np.pi))
    return dyn_factor


def compute_entropic_factor(Vsp, Vm, N0m, N0sp, detHm, detHsp, delta_e, T):
    return (
        Vsp
        / Vm
        * np.sqrt(2 * np.pi * constants.k_B * T) ** (N0m - N0sp - 1)
        * np.sqrt(detHm / detHsp)
        * np.exp(-delta_e / (constants.k_B * T))
    )
