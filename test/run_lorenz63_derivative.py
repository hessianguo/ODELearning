#%%

import os
import sys
import argparse

path1 = os.path.abspath('..')
sys.path.append(path1 + '/src/')

import numpy as np
import matplotlib.pyplot as plt

from gen_data import gen_observ
from denoise import denoise_vrkhs
from fd import compute_nonuniform_fd, compute_l2norm
from tvregdiff import TVRegDiff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsr", type=float, required=True)
    args = parser.parse_args()

    nsr = args.nsr

    examp_type = 'lorenz63'
    paras = [10, 28, 8/3]
    x0 = [1, 1, 1]
    time_interval = [0, 30]
    pts_type = 'uniform'
    pts_num = 6000
    ns_type = 2

    # generate data
    X_ns, X_data, T, Dx, sol = gen_observ(
        examp_type, paras, x0, time_interval,
        pts_type, pts_num, nsr, ns_type
    )
    T1 = T[1:]

    # finite difference
    X_dot_fd = compute_nonuniform_fd(X_ns, T)
    err_fd = compute_l2norm(X_dot_fd[:, 1:] - Dx[:, 1:], T1) / compute_l2norm(Dx[:, 1:], T1)

    # vRKHS with L-curve
    kernel_type = 'gauss'
    X_dot_rkhs, X_fit, lamb_lc = denoise_vrkhs(
        T, X_ns, None, 'auto', kernel_type, (0.02,)
    )
    err_rkhs = compute_l2norm(X_dot_rkhs - Dx[:, 1:], T1) / compute_l2norm(Dx[:, 1:], T1)

    # vRKHS with QOC
    X_dot_rkhs2, X_fit2, lamb_qoc = denoise_vrkhs(
        T, X_ns, None, 'auto-qoc', kernel_type, (0.02,)
    )
    err_rkhs2 = compute_l2norm(X_dot_rkhs2 - Dx[:, 1:], T1) / compute_l2norm(Dx[:, 1:], T1)

    # TV regularization
    dt = 0.005
    X_dot_tv_1 = TVRegDiff(X_ns[0, :], 50, 2e-2, scale='small', ep=1e+12, dx=dt, plotflag=0)
    X_dot_tv_2 = TVRegDiff(X_ns[1, :], 50, 2e-2, scale='small', ep=1e+12, dx=dt, plotflag=0)
    X_dot_tv_3 = TVRegDiff(X_ns[2, :], 50, 2e-2, scale='small', ep=1e+12, dx=dt, plotflag=0)

    X_dot_tv = np.vstack((X_dot_tv_1, X_dot_tv_2, X_dot_tv_3))
    err_tv = compute_l2norm(X_dot_tv[:, 1:] - Dx[:, 1:], T1) / compute_l2norm(Dx[:, 1:], T1)

    print('Errors of fd, rkhs-lc, rkhs-qoc, tv:',
          err_fd, err_rkhs, err_rkhs2, err_tv)

    err = np.array([err_fd, err_rkhs, err_rkhs2, err_tv, lamb_lc, lamb_qoc])
    filename = f"numerical_errors_lrz_nsr_{nsr:.2e}.txt"
    np.savetxt(
        filename,
        err.reshape(1, -1),
        fmt="%.6e",
        header="err_fd err_rkhs_lc err_rkhs_qoc err_tv lambda_lc lambda_qoc"
    )


if __name__ == "__main__":
    main()
# %%
