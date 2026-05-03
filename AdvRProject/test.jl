using TensorTrains
using LinearAlgebra

include("hamiltonian.jl")

N = 3
alpha = complex(1.0)
J = complex(1.0)

println("Testing Stupid Hamiltonian...")
h_mat = build_hamiltonian_stupid(N, alpha, J)
display(h_mat)

println("\nTesting TTO Hamiltonian...")
h_tto = build_hamiltonian_tto(N, alpha, J)
println("TTO built successfully!")