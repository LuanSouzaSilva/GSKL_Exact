from Hubbard_test import *
import time as time
import pandas as pd
import numpy as np

def Calc_Current(L, params, gamma):
    # --- 1. Build operators and find initial state ---
    start_time = time.time()
    print("1. Building operators and finding ground state...")
    num_qubits = 2 * L
    hubbard_hamiltonian_ferm = build_hubbard_hamiltonian(L, params[0], params[1], params[2])
    hamiltonian_matrix_sparse = get_sparse_operator(hubbard_hamiltonian_ferm, n_qubits=num_qubits)

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix_sparse.toarray())
    ground_state = eigenvectors[:, 0]
    print(f"   -> Ground state energy: {eigenvalues[0]:.6f}")

    # Build the current operator to measure the flow
    j_op_ferm_list = build_current_operator(L, params[0])
    j_matrices_sparse = [get_sparse_operator(j, n_qubits=num_qubits) for j in j_op_ferm_list]
    print(f"   -> Setup took {time.time() - start_time:.2f} seconds.")

    t_max = 40.0            # Maximum evolution time
    Nt = 401         # Number of time steps

    # --- 2. Nonlinear Response via GKSL Evolution ---
    print("\n2. Calculating driven-dissipative response via GKSL time evolution...")

    currents1 = []
    for i in range(len(gamma)):
        gammas = [gamma[i], gamma[i]]

        #start_time = time.time()
        times, current_t_list = calculate_driven_dissipative_current(
            hamiltonian_matrix_sparse, j_matrices_sparse, ground_state, 
            gammas[0], gammas[1], t_max, Nt
        )

        currents = np.array(current_t_list)

        currents1.append(currents[L//2-1, -1])

    return currents1

def Calc_Density(L, params, gamma):
    # --- 1. Build operators and find initial state ---
    start_time = time.time()
    print("1. Building operators and finding ground state...")
    num_qubits = 2 * L
    hubbard_hamiltonian_ferm = build_hubbard_hamiltonian(L, params[0], params[1], params[2])
    hamiltonian_matrix_sparse = get_sparse_operator(hubbard_hamiltonian_ferm, n_qubits=num_qubits)

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix_sparse.toarray())
    ground_state = eigenvectors[:, 0]
    print(f"   -> Ground state energy: {eigenvalues[0]:.6f}")

    t_max = 80.0            # Maximum evolution time
    Nt = 401         # Number of time steps
    t_list = np.linspace(0, t_max, Nt)

    # --- 2. Nonlinear Response via GKSL Evolution ---
    print("\n2. Calculating driven-dissipative response via GKSL time evolution...")

    # Convert operators to QuTiP's Qobj format
    H_qutip = qutip.Qobj(hamiltonian_matrix_sparse)

    # Initial state is the density matrix of the ground state
    rho0 = qutip.ket2dm(qutip.Qobj(ground_state))

    nups = []
    ndns = []
    for i in range(len(gamma)):
        ndn = Calculate_Density(L, H_qutip, rho0, gamma[i], t_list)

        ndns.append(ndn)

    return np.array(ndns[:][0])

# --- Parameters ---
L = int(input('Numero de sitios: '))

currents_input = input('Calcular corrente? (s/n): ').lower().strip()
Currents = (currents_input == 's') 

densities_input = input('Calcular densidades? (s/n): ').lower().strip()
Densities = (densities_input == 's') 

t_hopping = 1.0 #params[0]
U_interaction = 5.0 #params[1]
mu_potential = U_interaction/2 #params[2]

gamma = [0.5]#np.arange(0, 1+0.01, 0.1)

print(f"Driven-Dissipative Hubbard Model for L={L}, t={t_hopping}, U={U_interaction}")
print("="*80)

if Densities == True:
    print(Densities)
    ndns = Calc_Density(L, [t_hopping, U_interaction, mu_potential], gamma)
if Currents == True:
    currents1 = Calc_Current(L, [t_hopping, U_interaction, mu_potential], gamma)

#print(nups.shape)
df = pd.DataFrame(ndns.T)

df.to_csv(f'GSKL_Exact_gamma_N{L}_U5_densityxt.csv')
