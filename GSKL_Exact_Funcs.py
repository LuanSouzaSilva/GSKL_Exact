
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import kron, csc_matrix, eye


def Pauli(N):
    s0 = np.eye(2, dtype = complex)
    sx = np.array([[0, 1], [1, 0]], dtype = complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype = complex)
    sz = np.array([[1, 0], [0, -1]], dtype = complex)

    sp = (sx + 1j*sy)/2
    sm = (sx - 1j*sy)/2
    if N >= 6:
        return csc_matrix(s0), csc_matrix(sx), csc_matrix(sy), csc_matrix(sz), csc_matrix(sp), csc_matrix(sm)
    else:
        return s0, sx, sy, sz, sp, sm

def anticommutator(A, B):
    return A@B + B@A

def commutator(A, B):
    return A@B - B@A

def create_string_sparse(N, site, op, s0):
    if site == 0:
        string = op
        for i in range(N-1):
            string = kron(string, s0, format = 'csc')
        return string
    else:
        string = s0
        for i in range(1, N):
            if i == site:
                string = kron(string, op, format = 'csc')
            else:
                string = kron(string, s0, format = 'csc')
    return string

def create_string(N, site, op, s0):
    if N < 6:
        if site == 0:
            string = op
            for i in range(N-1):
                string = np.kron(string, s0)
            return string
        else:
            string = s0
            for i in range(1, N):
                if i == site:
                    string = np.kron(string, op)
                else:
                    string = np.kron(string, s0)
        return string
    else:
        return create_string_sparse(N, site, op, s0)

def hamiltonian(N, J, D, s0, sx, sy, sz):
    sxi = create_string(N, 0, sx, s0)
    sxip1 = create_string(N, 0+1, sx, s0)

    syi = create_string(N, 0, sy, s0)
    syip1 = create_string(N, 0+1, sy, s0)

    szi = create_string(N, 0, sz, s0)
    szip1 = create_string(N, 0+1, sz, s0)

    Ham = J*(sxi@sxip1 + syi@syip1 + D*(szi@szip1))
    for i in range(1, N-1):
        sxi = create_string(N, i, sx, s0)
        sxip1 = create_string(N, i+1, sx, s0)

        syi = create_string(N, i, sy, s0)
        syip1 = create_string(N, i+1, sy, s0)

        szi = create_string(N, i, sz, s0)
        szip1 = create_string(N, i+1, sz, s0)
        Ham = Ham + J*(sxi@sxip1 + syi@syip1 + D*(szi@szip1))
    return Ham

def density_matrix(N, probs, states):
    if N <= 6:
        rho = np.zeros((len(states[0]), len(states[0])), dtype = complex)
        for i in range(len(probs)):
            rho = rho + probs[i]*np.outer(states[i], states[i])
        return rho

def cal_D(L, rho):
    Ldag = L.T.conj()
    aux1 = L@rho@Ldag
    aux2 = -anticommutator(Ldag@L, rho)/2

    return aux1 + aux2

def dissipator(rho, op1, op2, gamma, f):
    return gamma*(f*cal_D(op1, rho) + (1-f)*cal_D(op2, rho))

def GSKL(H, rho, dissipators):
    return -1j*commutator(H, rho) + dissipators



def time_evolution(N, Nt, dt, H0, rho0, gammas, fs, op_dis, observable, eps):
    op01 = op_dis[0]
    op02 = op_dis[1]

    op11 = op_dis[2]
    op12 = op_dis[3]

    rho = rho0#np.zeros((Nt, len(rho0), len(rho0)), dtype = complex)
    rho_tr = np.zeros(Nt, dtype = complex)
    Opmean = np.zeros(Nt, dtype = complex)
    liouvillian_mean = np.zeros(Nt, dtype = complex)

    rho = rho0
    rho_tr[0] = rho.trace()
    Opmean[0] = (rho@observable).trace()

    Dfirst = dissipator(rho, op01, op02, gammas[0], fs[0])
    Dlast = dissipator(rho, op11, op12, gammas[1], fs[1])
    liouvillian_mean[0] = np.abs(GSKL(H0, rho, Dfirst + Dlast)).mean()
    for i in range(Nt-1):
        Dfirst = dissipator(rho, op01, op02, gammas[0], fs[0])
        Dlast = dissipator(rho, op11, op12, gammas[1], fs[1])

        rho = rho + GSKL(H0, rho, Dfirst + Dlast)*dt
        rho_tr[i+1] = rho.trace()
        rho = rho/rho_tr[i+1]
        Opmean[i+1] = (rho@observable).trace()
        liouvillian_mean[i+1] = np.abs(GSKL(H0, rho, Dfirst + Dlast)).mean()
        if np.abs(GSKL(H0, rho, Dfirst + Dlast)).mean() <= eps and i > 10:
            print(f'Rodou {i} iteracoes de tempo.')
            break

    return rho_tr, Opmean, liouvillian_mean

def time_evolution2(N, Nt, dt, H0, rho0, gammas, fs, op_dis, observable, eps):
    op01 = op_dis[0]
    op02 = op_dis[1]

    op11 = op_dis[2]
    op12 = op_dis[3]

    rho = rho0#np.zeros((Nt, len(rho0), len(rho0)), dtype = complex)
    rho_tr = np.zeros(Nt, dtype = complex)
    Opmean = np.zeros(Nt, dtype = complex)
    liouvillian_mean = np.zeros(Nt, dtype = complex)

    rho = rho0
    rho_tr[0] = rho.trace()
    Opmean[0] = (rho@observable).trace()

    Dfirst = dissipator(rho, op01, op02, gammas[0], fs[0])
    Dlast = dissipator(rho, op11, op12, gammas[1], fs[1])
    k1 = GSKL(H0, rho, Dfirst + Dlast)

    liouvillian_mean[0] = np.abs(GSKL(H0, rho, Dfirst + Dlast)).mean()
    for i in range(Nt-1):
        Dfirst = dissipator(rho, op01, op02, gammas[0], fs[0])
        Dlast = dissipator(rho, op11, op12, gammas[1], fs[1])
        k1 = GSKL(H0, rho, Dfirst + Dlast)

        Dfirst = dissipator(rho + k1*dt/2, op01, op02, gammas[0], fs[0])
        Dlast = dissipator(rho + k1*dt/2, op11, op12, gammas[1], fs[1])
        k2= GSKL(H0, rho + k1*dt/2, Dfirst + Dlast)

        Dfirst = dissipator(rho + k2*dt/2, op01, op02, gammas[0], fs[0])
        Dlast = dissipator(rho + k2*dt/2, op11, op12, gammas[1], fs[1])
        k3= GSKL(H0, rho + k2*dt/2, Dfirst + Dlast)

        Dfirst = dissipator(rho + k3*dt, op01, op02, gammas[0], fs[0])
        Dlast = dissipator(rho + k3*dt, op11, op12, gammas[1], fs[1])
        k4= GSKL(H0, rho + k3*dt, Dfirst + Dlast)

        rho = rho + (k1 + 2*(k2 + k3) + k4)*dt/6
        rho_tr[i+1] = rho.trace()
        rho = rho/rho_tr[i+1]
        Opmean[i+1] = (rho@observable).trace()
        liouvillian_mean[i+1] = np.abs(GSKL(H0, rho, Dfirst + Dlast)).mean()
        if np.abs(GSKL(H0, rho, Dfirst + Dlast)).mean() <= eps and i > 10:
            print(f'Rodou {i} iteracoes de tempo.')
            break

    return rho_tr, Opmean, liouvillian_mean