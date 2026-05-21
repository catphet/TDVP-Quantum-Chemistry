include("..\\src\\TensorTrains.jl")
using .TensorTrains
using LinearAlgebra

include("hamiltonian.jl")
include("timeevolution.jl")

# Redirect output to both terminal and file
log_file = open("results_n5.txt", "w")  

function logprint(args...)
    println(args...)
    println(log_file, args...)
    flush(log_file)
end

N = 5
alpha = complex(1.0)
J = complex(1.0)
t_total = 0.5
dt_coarse = 0.05
dt_fine = 0.01

logprint("/// N=$N VALIDATION ///\n")

logprint("/// STEP 1: Build exact matrix Hamiltonian ///")
H_mat = build_hamiltonian_s(N, alpha, J)
logprint("Hermitian: ", ishermitian(H_mat))

logprint("\n/// STEP 2: Check |00000⟩ is not an eigenstate ///")
psi_0_vec = zeros(ComplexF64, 2^N)
psi_0_vec[1] = 1.0
Hpsi = H_mat * psi_0_vec
logprint("H|00000⟩ = ", Hpsi)
logprint("(if all zeros, |00000⟩ is an eigenstate and survival probability will trivially be 1.0)")

logprint("\n/// STEP 3: Exact time evolution ///")
psi_t_exact = exp(-im * H_mat * t_total) * psi_0_vec
prob_exact = abs(dot(psi_0_vec, psi_t_exact))^2
logprint("Exact survival probability: ", prob_exact)

logprint("\n/// STEP 4: Check TTO Hamiltonian matches exact matrix ///")
H_tto = build_hamiltonian_tto(N, alpha, J)
H_tto_mat = tto_to_matrix(H_tto)
diff = norm(H_tto_mat - H_mat)
logprint("||H_tto - H_mat|| = ", diff)
logprint("(should be near zero, < 1e-10)")

logprint("\n/// STEP 5: TDVP time evolution |00000⟩ ///")
rks = ones(Int, N+1)
psi_0_tt = zeros_tt(ComplexF64, ntuple(_ -> 2, N), rks)
for i in 1:N
    psi_0_tt.ttv_vec[i][1, 1, 1] = 1.0
end
psi_t_tdvp = time_evolution_MPS(psi_0_tt, H_tto, t_total, dt_coarse; rmax=100)
overlap = dot(psi_0_tt, psi_t_tdvp)
prob_tdvp = abs(overlap)^2
logprint("TDVP survival probability: ", prob_tdvp)

logprint("\n/// STEP 6: Non-trivial initial state |+⟩⊗N ///")
psi_0_plus = zeros_tt(ComplexF64, ntuple(_ -> 2, N), ones(Int, N+1))
for i in 1:N
    psi_0_plus.ttv_vec[i][1, 1, 1] = 1/sqrt(2)
    psi_0_plus.ttv_vec[i][2, 1, 1] = 1/sqrt(2)
end
psi_0_plus_vec = fill(ComplexF64((1/sqrt(2))^N), 2^N)

psi_t_plus_exact = exp(-im * H_mat * t_total) * psi_0_plus_vec
prob_plus_exact = abs(dot(psi_0_plus_vec, psi_t_plus_exact))^2

psi_t_plus_coarse = time_evolution_MPS(psi_0_plus, H_tto, t_total, dt_coarse; rmax=100)
prob_plus_coarse = abs(dot(psi_0_plus, psi_t_plus_coarse))^2

psi_t_plus_fine = time_evolution_MPS(psi_0_plus, H_tto, t_total, dt_fine; rmax=100)
prob_plus_fine = abs(dot(psi_0_plus, psi_t_plus_fine))^2

logprint("Exact survival probability:          ", prob_plus_exact)
logprint("TDVP survival probability (dt=0.05): ", prob_plus_coarse)
logprint("Error (dt=0.05):                     ", abs(prob_plus_exact - prob_plus_coarse))
logprint("TDVP survival probability (dt=0.01): ", prob_plus_fine)
logprint("Error (dt=0.01):                     ", abs(prob_plus_exact - prob_plus_fine))

logprint("\n/// STEP 7: Timing (post-compilation) ///")
# warmup run to trigger compilation
time_evolution_MPS(psi_0_plus, H_tto, t_total, dt_fine; rmax=100)
logprint("Timing for N=$N, dt=$dt_fine:")
stats = @timed time_evolution_MPS(psi_0_plus, H_tto, t_total, dt_fine; rmax=100)
logprint("Elapsed time: ", stats.time, " seconds")
logprint("Memory: ", stats.bytes, " bytes")
logprint("Allocations: ", stats.gctime, " gc time")

close(log_file)