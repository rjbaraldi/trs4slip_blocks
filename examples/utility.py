import collections
import numpy as np


DiscretizationInfo = collections.namedtuple("DiscretizationInfo",
    ("t0 tf N h "
     "t_coarse t_ctr int_order "
     "lg_t lg_c lg_t_vec lg_c_vec "
     "lg_t_mat lg_c_mat"))


def create_discretization_info(t0, tf, N, int_order):
    h = (tf - t0) / N
    lg_t, lg_c = np.polynomial.legendre.leggauss(int_order)
    t_coarse = np.linspace(t0, tf, N + 1)
    t_ctr = .5 * (t_coarse[1:] + t_coarse[:-1])

    t_ctr_reps = np.repeat(t_ctr[np.newaxis, :], int_order, axis=0)
    lg_t_reps = np.repeat(lg_t[:, np.newaxis], N, axis=1) * .5 * h
    lg_t_mat = t_ctr_reps + lg_t_reps
    lg_t_vec = lg_t_mat.flatten('F')
    lg_c_vec = np.tile(lg_c, N)
    lg_c_mat = np.reshape(lg_c_vec, lg_t_vec.shape, 'F')

    return DiscretizationInfo(
        t0=t0, tf=tf, N=N, h=h,
        t_coarse=t_coarse, t_ctr=t_ctr, int_order=int_order,
        lg_t=lg_t, lg_c=lg_c, lg_t_vec=lg_t_vec, lg_c_vec=lg_c_vec,
        lg_t_mat=lg_t_mat, lg_c_mat=lg_c_mat)


class LgConvPwcMat:
    def __init__(self, lg_x, di):
        self.mat = self.lg_cm_create(lg_x, di.N)
        self.mat_short = self.mat[:di.int_order * di.N]

    def lg_cm_create(self, x, Ny):
        x = np.asarray(x)
        assert x.ndim == 2
        Nx = x.shape[1]
        M = x.shape[0]
        Nzi = Nx + Ny - 1
        Nz = M * Nzi
        lo_xi_stack = []
        for i in range(M):
            xi_stack = np.hstack([x[i], np.zeros(Nzi)])
            n = np.arange(Nzi)[:, np.newaxis]
            m = np.arange(Ny)
            lo_xi_stack.append(xi_stack[n - m])
        return np.hstack(lo_xi_stack).reshape((Nz, Nx))

    def conv(self, y):
        return self.mat_short.dot(y)


A = 0.1
omega0 = np.pi


def a(t):
    val = A * (1. - np.sqrt(2) * np.exp(-omega0 * t / np.sqrt(2)) * np.cos(omega0 * t / np.sqrt(2) - np.pi / 4.))
    return val


def mkappa(t):

    val = A*np.sqrt(2)*(np.exp(-omega0*t/np.sqrt(2))*np.cos(omega0*t/np.sqrt(2) - np.pi/4.)*(-omega0/np.sqrt(2))
                         + np.exp(-omega0*t/np.sqrt(2))*(-np.sin(omega0*t/np.sqrt(2) - np.pi/4.)*(omega0/np.sqrt(2))))
    return val


def f(t):
    return .2 * np.cos(2. * t * np.pi - 0.25) * np.exp(t)


def lg_int_mda(di):
    t = np.linspace(di.t0 - di.h, di.tf, di.N + 2)
    t_ctr = .5 * (t[1:] + t[:-1])
    t_ctr_mat = np.repeat(t_ctr[np.newaxis, :], di.int_order, axis=0)
    lg_t_reps = np.repeat(di.lg_t[:, np.newaxis], di.N + 1, axis=1) * .5 * di.h
    t_mat = t_ctr_mat + lg_t_reps
    a0c_stack = a(t_mat)
    int_mda_stack = -(a0c_stack[:, 1:] - a0c_stack[:, :-1])
    assert np.allclose(t_mat[:, 1:], di.lg_t_mat)
    return int_mda_stack


def lg_jacobian(x, lg_cm, di, f_vec):
    y_vec = lg_cm.conv(x)
    return .5 * .5 * di.h * lg_cm.mat_short.transpose().dot(np.multiply(di.lg_c_vec, 2. * (y_vec - f_vec)))


def lg_objective(x, lg_cm, di, f_vec):
    y_vec = lg_cm.conv(x)
    return .5 * .5 * di.h * np.inner(di.lg_c_vec, (y_vec - f_vec)**2)
