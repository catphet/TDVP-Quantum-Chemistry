using TensorTrains
using LinearAlgebra
function apply_hamiltonian(H::TensorTrain, psi::TensorTrain)
    L = length(psi)
    new_cores = Vector{Array{ComplexF64, 3}}(undef, L)
    
    for k in 1:L
        W = H.tensors[k]
        A = psi.tensors[k]
        
        rH_l, rH_r, d_in, d_out = size(W)
        rA_l, rA_r, d_phys = size(A)
        
        res = zeros(ComplexF64, rH_l, rA_l, rH_r, rA_r, d_out)
        for i in 1:rH_l, j in 1:rA_l, kr in 1:rH_r, lr in 1:rA_r, do_idx in 1:d_out
            for di in 1:d_in
                res[i, j, kr, lr, do_idx] += W[i, kr, di, do_idx] * A[j, lr, di]
            end
        end
        new_cores[k] = reshape(res, rH_l * rA_l, rH_r * rA_r, d_out)
    end
    return TensorTrain(new_cores)
end

function time_evolution_MPS(state::TensorTrain, hamiltonian::TensorTrain, t::ComplexF64, timestep::ComplexF64, rmax::Int=20)
    remaining_t = real(t)
    dt = timestep
    
    while remaining_t > 0
        h_psi = apply_hamiltonian(hamiltonian, state)
        factor = -im * dt
        for k in 1:length(h_psi)
            h_psi.tensors[k] .*= factor^(1/length(h_psi))
        end
        state = state + h_psi
        compress!(state; svd_trunc = TruncThresh(1e-12))
        n = norm(state)
        for k in 1:length(state)
            state.tensors[k] .*= (1/n)^(1/length(state))
        end
        
        remaining_t -= real(dt)
        println("Time left: ", round(remaining_t, digits=4))
    end
    return state
end