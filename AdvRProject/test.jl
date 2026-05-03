using TensorTrains
using LinearAlgebra

include("hamiltonian.jl")
include("timeevolution.jl")

N = 3
alpha = complex(1.0)
J = complex(1.0)

println("Building Hamiltonian...")
h_tto = build_hamiltonian_tto(N, alpha, J)

println("Initializing State |000>...")
up_vec = reshape(ComplexF64[1.0, 0.0], (1, 1, 2))
psi_0 = TensorTrain([up_vec for _ in 1:N])

println("Starting Evolution...")
t_total = 0.5 + 0.0im
dt = 0.05 + 0.0im
psi_t = time_evolution_MPS(psi_0, h_tto, t_total, dt, 10)
overlap = dot(psi_0, psi_t)
prob = abs(overlap)^2

println("\n--- Results ---")
println("Final overlap: ", overlap)
println("Survival Probability: ", prob)