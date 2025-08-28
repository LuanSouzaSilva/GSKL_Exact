# Solve steady state of Lindblad master equation by trace-constrained linear solve
# Uses orbital bit basis (spinful), sparse Liouvillian construction, and replaces one row with trace constraint:
#   L_mod vec(rho) = b, where L_mod[0,:] = trace_row, b[0]=1 and other entries 0.
# This avoids eigen-solvers and is robust for small-medium systems.
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

# ---- Parameters (small system for demo) ----
L = 2                 # sites (spinful)
t = 1.0
U = 4.0
gamma_in = 1.0
gamma_out = 1.0
V_field = 1.0         # example voltage per site
# --------------------------------------------

n_orbitals = 2 * L
dim = 2 ** n_orbitals

def occ_bit(state, orb):
    return (state >> orb) & 1
def flip_bit(state, orb):
    return state ^ (1 << orb)
def parity_sign(state, orb):
    mask = (1 << orb) - 1
    n_before = bin(state & mask).count("1")
    return -1 if (n_before % 2 == 1) else 1

def build_c_operator(orb):
    rows = []
    cols = []
    data = []
    for n in range(dim):
        if occ_bit(n, orb) == 1:
            m = flip_bit(n, orb)
            sign = parity_sign(n, orb)
            rows.append(m)
            cols.append(n)
            data.append(sign)
    return sp.csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=complex)

# Build operators
c = [build_c_operator(p) for p in range(n_orbitals)]
cdag = [op.getH() for op in c]
n_op_orb = [cdag[p].dot(c[p]) for p in range(n_orbitals)]
def site_orb_index(site, spin):
    return 2 * site + spin
n_site = [n_op_orb[site_orb_index(i,0)] + n_op_orb[site_orb_index(i,1)] for i in range(L)]

# Hamiltonian builder
def build_hamiltonian(V_field=0.0):
    H = sp.csr_matrix((dim, dim), dtype=complex)
    for i in range(L-1):
        for spin in (0,1):
            p = site_orb_index(i, spin)
            q = site_orb_index(i+1, spin)
            H += -t * (cdag[p].dot(c[q]) + cdag[q].dot(c[p]))
    for i in range(L):
        p_up = site_orb_index(i,0)
        p_dn = site_orb_index(i,1)
        H += U * (cdag[p_up].dot(c[p_up])).dot(cdag[p_dn].dot(c[p_dn]))
    if V_field != 0.0:
        for i in range(L):
            H += V_field * i * n_site[i]
    return H

# Liouvillian builder (robust sparse)
I = sp.eye(dim, dtype=complex, format='csr')
def liouvillian(H, jump_list):
    H = sp.csr_matrix(H, dtype=complex)
    I = sp.eye(H.shape[0], dtype=complex, format='csr')
    Lh = -1j * (sp.kron(I, H, format='csr') - sp.kron(H.T.conj(), I, format='csr'))
    Ltot = Lh.copy()
    for (Lop, rate) in jump_list:
        Lop = sp.csr_matrix(Lop, dtype=complex)
        Lop_dag = Lop.getH()
        left = sp.kron(Lop, Lop.conj(), format='csr')
        M = sp.csr_matrix(Lop_dag.dot(Lop), dtype=complex)
        right1 = sp.kron(I, M, format='csr')
        right2 = sp.kron(M.T, I, format='csr')
        Ltot = Ltot + rate * (left - 0.5 * right1 - 0.5 * right2)
    return Ltot

# Jumps: injection at site 0 (cdag), removal at site L-1 (c)
def build_jumps(gamma_in, gamma_out):
    jumps = []
    for spin in (0,1):
        p = site_orb_index(0, spin)
        jumps.append((cdag[p], gamma_in))
    for spin in (0,1):
        p = site_orb_index(L-1, spin)
        jumps.append((c[p], gamma_out))
    return jumps

# Trace-constrained linear solve for steady state
def steady_state_linear_solve(H, jumps):
    Lmat = liouvillian(H, jumps)  # sparse (dim^2 x dim^2)
    N = Lmat.shape[0]
    # build trace row: ones at positions i + i*dim  (column-stacking vec -> diag positions every (dim+1))
    trace_row = sp.lil_matrix((1, N), dtype=complex)
    diag_positions = np.arange(0, N, dim+1)
    trace_row[0, diag_positions] = 1.0
    # convert Lmat to lil to replace first row safely
    L_lil = Lmat.tolil()
    L_lil[0,:] = trace_row
    L_mod = L_lil.tocsr()
    b = np.zeros(N, dtype=complex)
    b[0] = 1.0
    # solve sparse linear system
    try:
        vec = spla.spsolve(L_mod, b)
    except Exception as e:
        # fallback to dense solve
        Ldense = L_mod.toarray()
        vec = la.solve(Ldense, b)
    rho = vec.reshape((dim, dim), order='F')
    # enforce hermiticity, normalize
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / np.trace(rho)
    return rho

# Build H and jumps and solve
H = build_hamiltonian(V_field=V_field)
jumps = build_jumps(gamma_in, gamma_out)
rho_ss = steady_state_linear_solve(H, jumps)

# Compute current on bond 0
def build_current_operator(bond_i):
    J = sp.csr_matrix((dim, dim), dtype=complex)
    for spin in (0,1):
        p = site_orb_index(bond_i, spin)
        q = site_orb_index(bond_i+1, spin)
        J += -1j * t * (cdag[p].dot(c[q]) - cdag[q].dot(c[p]))
    return J

Jop = build_current_operator(0)
I = np.real(np.trace(rho_ss.dot(Jop.toarray())))

print("Steady-state current I =", I)
print("Trace of rho_ss =", np.trace(rho_ss))
print("Eigenvalues of rho_ss (should be >=0):", np.linalg.eigvals(rho_ss))
