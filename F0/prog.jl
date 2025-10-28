using Pkg
using TensorMixedStates, .Electrons
using ITensors
using ITensorMPS
using LinearAlgebra

#Maximum bond dimension:
global MAXDIM = 128
#SVD truncation level:
global CUTOFF = 1e-30

Threads.nthreads() = 8 #Number of threads 

#function to calculate the ground state using ITensor's DMRG
function Hubbard_DMRG(Nsites, t, U, ed)
    site_types = [Electron() for i in 1:Nsites] #Vector of Electron()
    sys = System(site_types) #system using the Vector created above

    #sistes is a Vector{Index}, used for the DMRG
    sites = sys.pure_indices

    #Build of the hamiltonian as a sum of operators
    os = OpSum()
    #local energy/chemical potential
    for j = 1:Nsites
        os += -ed, "Nup", j
        os += -ed, "Ndn", j
    end

    #hopping terms
    for j = 1:Nsites-1
        os += -t, "Cdagup", j + 1, "Cup", j
        os += -t, "Cdagup", j, "Cup", j + 1

        os += -t, "Cdagdn", j + 1, "Cdn", j
        os += -t, "Cdagdn", j, "Cdn", j + 1
    end

    #Local Coulomb interaction
    for j = 1:Nsites
        os += U, "Nupdn", j
    end

    H = MPO(os, sites) #Conversion to MPO

    psi0 = randomMPS(sites; linkdims=MAXDIM) #Random_MPS works only if conserve_qns = false

    #state = [isodd(n) ? "Up" : "Dn" for n=1:Nsites] #Creates a Neel State if conserve_ne = true
    #psi0 = MPS(sites,state) #use if conserve_qns = true

    #number of sweeps. Uncomment the lines below if the system size is big enough
    nsweeps = 5#100
    #maxdim1 = ones(Int, 40) .* 10
    #maxdim2 = ones(Int, 50) .* 20
    #maxdim3 = [40, 40, 80, 80, 100, 100, 200, 200, 400, 500]
    maxdim = [10, 10, 16, 16, 32]#[maxdim1; maxdim2; maxdim3]
    cutoff = [1E-12] #cutoff of DMRG
    noise = [1E-5, 1E-7, 1E-8, 1E-10, 1E-12] #noise. This parameter helps DMRG to not get stucked in local minima

    GS_energy1, GS0 = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise, eigsolve_krylovdim=4, outputlevel=1) #DMRG calculation

    #Conversion of the Ground State found by DMRG (type MPS) to type {Pure}State (from TMS)
    GS = State{Pure}(sys, GS0)
    
    #Conversion of the {Pure}State to a Density Matrix in Mixed Space (which is d^2 dimensional, where d is the dimension of the original Hilbert Space)
    rho0 = mix(GS)

    return rho0, sys #Return of both initial state and system
end


#Limits of the MPS/MPO used in TMS
limits = Limits(
    cutoff=CUTOFF,
    maxdim=MAXDIM,
)

#A list of outputs which will be saved in a .dat file
output(n) = [
    #"density_up.dat" => Nup, #Local density of spin up electrons
    #"density_dn.dat" => Ndn, #Local density of spin down electrons
    "total_number_up.dat" => sum(Nup(i) for i in 1:n), #Total number of spin up electrons
    "total_number_dn.dat" => sum(Ndn(i) for i in 1:n), #Total number of spin down electrons
    "OSEE.dat" => EE(1 + n ÷ 2, 4), #Entanglement entropy of a bipartite system (parts are separeted by the "center" of the systems)
    "purity.dat" => Purity, #Purity (\Tr \rho^2) of the system
    "trace.dat" => Trace, #Trace of the density matrix during time evolution. We need to ensure that \Tr \rho = 1
    "bond_dim.dat" => Linkdim #Bond dimension
]


#High level interface of TMS
sim_data(n, step, duration, alg, Gamma, init_sys, init_rho, t, U, mu) = SimData(
    name="F0",
    description="""
      Hubbard model with $n sites and a particle source (sink) in the right (left) 
      cutoff = $CUTOFF
      maxdim = $MAXDIM
      algo #$alg W2o2
      time_step = $step
      duration = $duration
      electron injection (absorption) at the right (left) = $Gamma
      initial state: Ground state
  """,
    phases=[
        CreateState(
            name="Initialization - finding the ground state of the Hubbard model",
            type=Mixed(),
            system=init_sys,
            state=init_rho,
            final_measures=output(n),
        ),
        Evolve(
            algo=(
                (alg == 0) ? Tdvp() : (
                    (alg == 1) ? ApproxW(order=4, w=1, n_hermitianize=5) : ApproxW(order=4, w=2, n_hermitianize=5)
                )),
            limits=limits,
            duration=duration,
            time_step=step,
            # The evolver follows the Lindblad (GSKL) quantum master equation
            evolver=-im * (
                -t*sum(dag(Cup)(i) * Cup(i + 1) + dag(Cup)(i + 1) * Cup(i)
                + dag(Cdn)(i) * Cdn(i + 1) + dag(Cdn)(i + 1) * Cdn(i) for i in 1:n-1)
                -mu*sum(Ntot(i) for i in 1:n-1)
                + U*sum(Nupdn(i) for i in 1:n-1)
            ) 
            + Dissipator(sqrt(Gamma) * (dag(Cup) + dag(Cdn)))(n)
            + Dissipator(sqrt(Gamma) * (Cup + Cdn))(1), #dissipative term injecting fermions in the center of the chain
            measures=output(n),
            measures_period=2,
        ),
        #Calculation of the steady state from the Liouvillian (Need to understand this)
        SteadyState(
        lindbladian = -im * (
                -t*sum(dag(Cup)(i) * Cup(i + 1) + dag(Cup)(i + 1) * Cup(i)
                + dag(Cdn)(i) * Cdn(i + 1) + dag(Cdn)(i + 1) * Cdn(i) for i in 1:n-1)
                -mu*sum(Ntot(i) for i in 1:n-1)
                + U*sum(Nupdn(i) for i in 1:n-1)
            ) 
            + Dissipator(sqrt(Gamma) * (dag(Cup) + dag(Cdn)))(n)
            + Dissipator(sqrt(Gamma) * (Cup + Cdn))(1),
        limits = Limits(cutoff = 1e-20, maxdim = [10, 20, 50]),
        nsweeps = 10,
        tolerance = 1e-5,
        )
        ]
)


#Definition of parameters
n = 4 #number of sites

t = 1. #hopping amplitude
U = 5. #local Coulomb interaction
mu = U/2 #Chemical Potential

step = 0.1 #timestep
duration = 10 #Tmax
#alg = 0 # TDVP
#alg=1 # W1 order 4
alg=2 # W2 order 4
Gamma = 0.5 #System-Environment coupling 

rho, sys = Hubbard_DMRG(n, 1., 5., 2.5) #Calculation of the ground state

#Simulation using TMS
runTMS(sim_data(n, step, duration, alg, Gamma, sys, rho, t, U, mu), restart=true)