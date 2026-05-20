include("..\\src\\TensorTrains.jl")
using .TensorTrains
using LinearAlgebra
using BenchmarkTools

include("hamiltonian.jl")
include("timeevolution.jl")

N = 3
alpha = complex(1.0)
J = complex(1.0)
t_total = 0.5
dt_coarse = 0.05
dt_fine = 0.01

println("=== N=$N VALIDATION ===\n")

println("=== STEP 1: Build exact matrix Hamiltonian ===")
H_mat = build_hamiltonian_stupid(N, alpha, J)
println("Hermitian: ", ishermitian(H_mat))

println("\n=== STEP 2: Check |000⟩ is not an eigenstate ===")
psi_0_vec = zeros(ComplexF64, 2^N)
psi_0_vec[1] = 1.0
Hpsi = H_mat * psi_0_vec
println("H|000⟩ = ", Hpsi)
println("(if all zeros, |000⟩ is an eigenstate and survival probability will trivially be 1.0)")

println("\n=== STEP 3: Exact time evolution ===")
psi_t_exact = exp(-im * H_mat * t_total) * psi_0_vec
prob_exact = abs(dot(psi_0_vec, psi_t_exact))^2
println("Exact survival probability: ", prob_exact)

println("\n=== STEP 4: Check TTO Hamiltonian matches exact matrix ===")
H_tto = build_hamiltonian_tto(N, alpha, J)
H_tto_mat = tto_to_matrix(H_tto)
diff = norm(H_tto_mat - H_mat)
println("||H_tto - H_mat|| = ", diff)
println("(should be near zero, < 1e-10)")

println("\n=== STEP 5: TDVP time evolution |000⟩ ===")
rks = ones(Int, N+1)
psi_0_tt = zeros_tt(ComplexF64, ntuple(_ -> 2, N), rks)
for i in 1:N
    psi_0_tt.ttv_vec[i][1, 1, 1] = 1.0
end
psi_t_tdvp = time_evolution_MPS(psi_0_tt, H_tto, t_total, dt_coarse; rmax=100)
overlap = dot(psi_0_tt, psi_t_tdvp)
prob_tdvp = abs(overlap)^2
println("TDVP survival probability: ", prob_tdvp)

println("\n=== STEP 6: Non-trivial initial state |+⟩⊗N ===")
psi_0_plus = zeros_tt(ComplexF64, ntuple(_ -> 2, N), ones(Int, N+1))
for i in 1:N
    psi_0_plus.ttv_vec[i][1, 1, 1] = 1/sqrt(2)
    psi_0_plus.ttv_vec[i][2, 1, 1] = 1/sqrt(2)
end
psi_0_plus_vec = fill((1/sqrt(2))^N, 2^N)

psi_t_plus_exact = exp(-im * H_mat * t_total) * psi_0_plus_vec
prob_plus_exact = abs(dot(psi_0_plus_vec, psi_t_plus_exact))^2

psi_t_plus_coarse = time_evolution_MPS(psi_0_plus, H_tto, t_total, dt_coarse; rmax=100)
prob_plus_coarse = abs(dot(psi_0_plus, psi_t_plus_coarse))^2

psi_t_plus_fine = time_evolution_MPS(psi_0_plus, H_tto, t_total, dt_fine; rmax=100)
prob_plus_fine = abs(dot(psi_0_plus, psi_t_plus_fine))^2

println("Exact survival probability:          ", prob_plus_exact)
println("TDVP survival probability (dt=0.05): ", prob_plus_coarse)
println("Error (dt=0.05):                     ", abs(prob_plus_exact - prob_plus_coarse))
println("TDVP survival probability (dt=0.01): ", prob_plus_fine)
println("Error (dt=0.01):                     ", abs(prob_plus_exact - prob_plus_fine))

println("\n=== STEP 7: Timing (post-compilation) ===")
# warmup run to trigger compilation
time_evolution_MPS(psi_0_plus, H_tto, t_total, dt_fine; rmax=100)
println("Timing for N=$N, dt=$dt_fine:")
@btime time_evolution_MPS($psi_0_plus, $H_tto, $t_total, $dt_fine; rmax=100)