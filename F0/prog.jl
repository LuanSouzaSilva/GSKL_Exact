using Pkg
using TensorMixedStates, .Electrons
using ITensors
using ITensorMPS
using LinearAlgebra

#Maximum bond dimension:
MAXDIM = 16
#SVD truncation level:
CUTOFF = 1e-30

Threads.nthreads() = 4

function Hubbard_DMRG(Nsites, t, U, ed)
    sites = siteinds("Electron", Nsites, conserve_nf=false)

    os = OpSum()
    for j = 1:Nsites
        os += -ed, "Nup", j
        os += -ed, "Ndn", j
    end

    for j = 1:Nsites-1
        os += -t, "Cdagup", j + 1, "Cup", j
        os += -t, "Cdagup", j, "Cup", j + 1

        os += -t, "Cdagdn", j + 1, "Cdn", j
        os += -t, "Cdagdn", j, "Cdn", j + 1
    end

    for j = 1:Nsites
        os += U, "Nupdn", j
    end

    H = MPO(os, sites)

    psi0 = randomMPS(sites; linkdims=16) #Esse comando de Random_MPS so funciona se conserve_qns = false

    #state = [isodd(n) ? "Up" : "Dn" for n=1:Nsites]
    #psi0 = MPS(sites,state)

    nsweeps = 5#100
    maxdim1 = ones(Int, 40) .* 10
    maxdim2 = ones(Int, 50) .* 20
    maxdim3 = [40, 40, 80, 80, 100, 100, 200, 200, 400, 500]
    maxdim = [10, 10, 16, 16, 16]#[maxdim1; maxdim2; maxdim3]
    cutoff = [1E-12]
    noise = [1E-5, 1E-7, 1E-8, 1E-10, 1E-12]

    GS_energy1, GS0 = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise, eigsolve_krylovdim=7, outputlevel=1)

    site_types = [Electron() for i in 1:Nsites]
    sys = System(site_types)#sys = System(Nsites, Electron()) 

    GS = State{Mixed}(sys, GS0)
    print(typeof(GS), "\n")
    rho0 = mix(GS)
    print(typeof(rho0), "\n")
    return rho0
end
rho = Hubbard_DMRG(2, 1., 5., 2.5)

# limits = Limits(
#     cutoff=CUTOFF,
#     maxdim=MAXDIM,
# )

# output(n) = [
#     "density_up.dat" => Nup,
#     "density_dn.dat" => Ndn,
#     "total_number_up.dat" => sum(Nup(i) for i in 1:n),
#     "total_number_dn.dat" => sum(Ndn(i) for i in 1:n),
#     "OSEE.dat" => EE(1 + n ÷ 2, 4),
#     "purity.dat" => Purity,
#     "trace.dat" => Trace,
#     "bond_dim.dat" => Linkdim
# ]

# sim_data(n, step, duration, alg, Gamma, U, mu, t) = SimData(
#     name="F0",
#     description="""
#       Free fermion chain with $n sites and particle source in the center
#       cutoff = $CUTOFF
#       maxdim = $MAXDIM
#       algo #$alg W2o2
#       time_step = $step
#       duration = $duration
#       fermion injection rate at the center Gamma = $Gamma
#       empty initial state: |00...0>
#       model and notations: see P L Krapivsky et al J. Stat. Mech. (2019) 113108 http://doi.org/10.1088/1742-5468/ab4e8e
#   """,
#     phases=[
#         CreateState(
#             name="Initialization - creating an empty (spinless) fermion chain",
#             type=Mixed(),
#             system=System(n, Electron()),
#             state=Hubbard_DMRG(n, t, U, mu),
#             final_measures=output(n),
#         ),
#         Evolve(
#             algo=(
#                 (alg == 0) ? Tdvp() : (
#                     (alg == 1) ? ApproxW(order=4, w=1, n_hermitianize=5) : ApproxW(order=4, w=2, n_hermitianize=5)
#                 )),
#             limits=limits,
#             duration=duration,
#             time_step=step,
#             evolver=-im * (
#                         -t * sum(dag(Cup)(i) * Cup(i + 1) + dag(Cup)(i + 1) * Cup(i) +
#                                  dag(Cdn)(i) * Cdn(i + 1) + dag(Cdn)(i + 1) * Cdn(i) for i in 1:n-1)
#                                  + sum(+ U * Nupdn(i) - mu * Ntot(i) for i in 1:n))
#                     + Dissipator(sqrt(2 * Gamma) * (dag(Cdn) + dag(Cup)))(n) + Dissipator(sqrt(2 * Gamma) * (Cdn + Cup))(1), #dissipative term injecting fermions in the center of the chain
#             measures=output(n),
#             measures_period=2,
#         ),]
# )

# n = 2
# step = 0.1
# duration = 20
# alg = 0 # TDVP
# #alg=1 # W1 order 4
# #alg=2 # W2 order 4
# Gamma = 0.2
# runTMS(sim_data(n, step, duration, alg, Gamma, 5., 2.5, 1.), restart=true)