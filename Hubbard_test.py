import numpy as np
import matplotlib.pyplot as plt
from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator, get_ground_state
import qutip
# NEW: Import the data module for the inspect function
from qutip.core import data as qutip_data
from scipy.integrate import solve_ivp
import time as time

# --- Helper Functions (Unchanged) ---
def creation_operator(site, spin):
    qubit_index = 2 * site + spin
    return FermionOperator(f'{qubit_index}^', 1.0)

def annihilation_operator(site, spin):
    qubit_index = 2 * site + spin
    return FermionOperator(f'{qubit_index}', 1.0)

def build_hubbard_hamiltonian(num_sites, t, U, mu=0.0, periodic=False):
    # This function is unchanged.
    hamiltonian = FermionOperator()
    # 1. Termo de Hopping (-t)
    for i in range(num_sites):
        # Condições de contorno
        if periodic:
            # Pula a última conexão para evitar contagem dupla (L-1 -> 0 é contado por 0 -> L-1)
            if i == num_sites - 1:
                j = 0
            else:
                j = i + 1
        else: # Contorno aberto
            if i == num_sites - 1:
                continue
            j = i + 1

        for spin in [0, 1]:  # 0 para down, 1 para up
            # Cria os operadores para o salto i -> j e seu conjugado hermitiano j -> i
            # Termo: c_i^\dagger c_j
            c_i_dag = creation_operator(i, spin)
            c_j = annihilation_operator(j, spin)
            
            # Termo conjugado: c_j^\dagger c_i
            c_j_dag = creation_operator(j, spin)
            c_i = annihilation_operator(i, spin)

            # Adiciona ambos os termos para garantir que o Hamiltoniano seja Hermitiano
            hamiltonian += -t * (c_i_dag * c_j + c_j_dag * c_i)
    for i in range(num_sites):
        n_up = creation_operator(i, 1) * annihilation_operator(i, 1)
        n_down = creation_operator(i, 0) * annihilation_operator(i, 0)
        hamiltonian += U * (n_up * n_down)
    for i in range(num_sites):
        for spin in [0, 1]:
            num_op = creation_operator(i, spin) * annihilation_operator(i, spin)
            hamiltonian += -mu * num_op
    return hamiltonian

def build_current_operator(num_sites, t):
    """Builds the bond current operator j between site i and i+1."""
    # We measure the current in the middle of the chain for stability
    op_list = []
    for i in range(num_sites -1):
        j_op = FermionOperator()
        for spin in [0, 1]:
            term = 1j * t * (
                creation_operator(i + 1, spin) * annihilation_operator(i, spin) -
                creation_operator(i, spin) * annihilation_operator(i + 1, spin)
            )
            j_op += term
        op_list.append(j_op)
    return op_list

# --- NEW FUNCTION for Driven-Dissipative Nonlinear Conductivity ---

def calculate_driven_dissipative_current(H_hubbard_sparse, j_matrices_sparse, ground_state, gamma_in, gamma_out, t_max, num_steps):
    """
    Calculates time-dependent current in a driven-dissipative system using GKSL.
    """
    num_sites = int(np.log2(H_hubbard_sparse.shape[0]) / 2)
    num_qubits = 2 * num_sites
    
    # 1. Convert operators to QuTiP format
    H_qutip = qutip.Qobj(H_hubbard_sparse)
    j_qutip_ops = [qutip.Qobj(j) for j in j_matrices_sparse]
    
    # 2. Define the Lindblad (collapse) operators for boundary driving
    c_ops = []
    
    # Driving at the left boundary (site 0) - Particle INJECTION
    # Models a high-potential lead
    for spin in [0, 1]:
        # L_in = sqrt(gamma_in) * c_0^dagger
        op = creation_operator(0, spin)
        op_sparse = get_sparse_operator(op, n_qubits=num_qubits)
        c_ops.append(np.sqrt(gamma_in) * qutip.Qobj(op_sparse))

    # Draining at the right boundary (site L-1) - Particle EXTRACTION
    # Models a low-potential lead
    for spin in [0, 1]:
        # L_out = sqrt(gamma_out) * c_{L-1}
        op = annihilation_operator(num_sites - 1, spin)
        op_sparse = get_sparse_operator(op, n_qubits=num_qubits)
        c_ops.append(np.sqrt(gamma_out) * qutip.Qobj(op_sparse))
        
    # 3. Set the initial state
    # We start from the ground state of the closed system
    rho0 = qutip.ket2dm(qutip.Qobj(ground_state))
    
    # 4. Evolve the system using the master equation solver
    times = np.linspace(0, t_max, num_steps)
    
    # mesolve solves d(rho)/dt = -i[H, rho] + L(rho)
    # e_ops is the list of operators for which we want to calculate expectation values
    result = qutip.mesolve(H_qutip, rho0, times, c_ops, e_ops=j_qutip_ops)
    
    # result.expect contains a list of arrays, one for each operator in e_ops
    return times, result.expect

def Calculate_Density(L, H_qutip, rho0, gamma, t_list):
    num_qubits = 2 * L

    c_ops = []
    cdag_ops = []
    loss_op_ferm = annihilation_operator(0, 0)
    loss_op_sparse = get_sparse_operator(loss_op_ferm, n_qubits=num_qubits)

    loss1_op_ferm = annihilation_operator(0, 1)
    loss1_op_sparse = get_sparse_operator(loss1_op_ferm, n_qubits=num_qubits)

    creation_op_ferm = creation_operator(L-1, 0)
    creation_op_sparse = get_sparse_operator(creation_op_ferm, n_qubits=num_qubits)

    creation1_op_ferm = creation_operator(L-1, 1)
    creation1_op_sparse = get_sparse_operator(creation1_op_ferm, n_qubits=num_qubits)


    c_ops.append(np.sqrt(gamma) * (qutip.Qobj(loss_op_sparse) + qutip.Qobj(loss1_op_sparse)))
    cdag_ops.append(np.sqrt(gamma) * (qutip.Qobj(creation_op_sparse) + qutip.Qobj(creation1_op_sparse)))


    # Define observable to track
    nup_ops = []
    ndn_ops = []

    for i in range(L):
        ndn_ops.append(creation_operator(i, 0) * annihilation_operator(i, 0))
        nup_ops.append(creation_operator(i, 1) * annihilation_operator(i, 1))

    expn_values = []
    for  i in range(L):
        numdn_op_qutip = qutip.Qobj(get_sparse_operator(ndn_ops[i], n_qubits=num_qubits))
        numup_op_qutip = qutip.Qobj(get_sparse_operator(nup_ops[i], n_qubits=num_qubits))

        # --- 2. Solve the Master Equation using qutip.mesolve ---
        print("\n2. Solving with qutip.mesolve...")

        start_time = time.time()
        # This is the master equation solver. It handles all the complexity internally.
        result_dn = qutip.mesolve(H_qutip, rho0, t_list, [c_ops, cdag_ops], e_ops=[numdn_op_qutip])
        result_up = qutip.mesolve(H_qutip, rho0, t_list, [c_ops, cdag_ops], e_ops=[numup_op_qutip])
        print(f"   -> Evolution took {time.time() - start_time:.2f} seconds.")

        expn_values.append(result_up.expect[0])
        expn_values.append(result_dn.expect[0])

    return np.array(expn_values)
