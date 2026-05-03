using TensorTrains
function time_evolution(state::TensorTrain{T, N}, hamiltonian::TensorTrain{T, N}, t::ComplexF64, timestep::ComplexF64) where {T, N}
    while t.re > 0
        t -= timestep
        state = state - (timestep * im * (hamiltonian * state))
    end
    return state
end

function time_evolution_MPS(state::TensorTrain{T, N}, hamiltonian::TensorTrain{T, N}, t::ComplexF64, timestep::ComplexF64, rmax::Int=typemax(Int)) where {T, N}
    while t.re > 0
        t -= timestep
        raw_state = state - (timestep * im * (hamiltonian * state))
        state = compress!(raw_state, SVDTrunc(TruncBond(rmax)))
    end
    return state
end

function time_evolution_tangent_proj(state::TensorTrain{T, N}, hamiltonian::TensorTrain{T, N}, t::ComplexF64, timestep::ComplexF64) where {T, N}
    while t.re > 0
        t -= timestep
    end
    return state
end