using TensorTrains

# psi_(n+1) = psi_n - timestep * i * H * psi_n
function time_evolution(state::TTvector{ComplexF64, N}, hamiltonian::TToperator{ComplexF64, N}, t::ComplexF64, timestep::ComplexF64)
    while(t > 0)
        t -= timestep
        state = state - timestep * im * hamiltonian * state
    end
    return state
end

# psi_(n+1) = TT-SVD_r(psi_n - timestep * i * H * psi_n)
function time_evolution_MPS(state::TTvector{ComplexF64, N}, hamiltonian::TToperator{ComplexF64, N}, t::ComplexF64, timestep::ComplexF64, rmax=typemax(Int64)::Int64)
    while t > 0
        t -= timestep
        # TODO
    end
    return state
end

# psi_(n+1) = TT-SVD_r(psi_n - timestep * P_T_MPS * i * H * psi_n)
function time_evolution_tangent_proj(state::TTvector{ComplexF64, N}, hamiltonian::TToperator{ComplexF64, N}, t::ComplexF64, timestep::ComplexF64)
    while t > 0
        t -= timestep
        # TODO
    end
    return state
end