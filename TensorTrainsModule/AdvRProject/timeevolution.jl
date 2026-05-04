using .TensorTrains

# returns the canonical tt decomposition of rank <= rmax of a tensor 
function ttv_svd_r(tensor::Array{T,d}, index=1, tol=1e-12, rmax=typemax(Int64)) where {T<:Number,d}
    dims = size(tensor)
    ttv_vec = Array{Array{T,3}}(undef, d)

    ttv_ot = -ones(Int64, d)
    ttv_ot[index] = 0
    if index < d
        ttv_ot[index+1:d] .= 1
    end

    rks = ones(Int64, d+1)
    tensor_curr = tensor

    # left : i < index
    for i = 1:(index-1)
        tensor_curr = reshape(tensor_curr, rks[i] * dims[i], :)
        U, S, V = svd(tensor_curr)

        rks[i+1] = max(1, min(rmax, count(>=(tol), S)))

        ttv_vec[i] = zeros(T, dims[i], rks[i], rks[i+1])
        for x = 1:dims[i]
            ttv_vec[i][x, :, :] = U[(rks[i]*(x-1)+1):(rks[i]*x), 1:rks[i+1]]
        end

        tensor_curr = Diagonal(S[1:rks[i+1]]) * V'[1:rks[i+1], :]
    end

    # right : i > index
    if index < d
        for i = d:-1:(index+1)
            tensor_curr = reshape(tensor_curr, :, dims[i] * rks[i+1])
            U, S, V = svd(tensor_curr)

            rks[i] = max(1, min(rmax, count(>=(tol), S)))

            ttv_vec[i] = zeros(T, dims[i], rks[i], rks[i+1])
            for x = 1:dims[i]
                i_vec = dims[i] .* (0:rks[i+1]-1) .+ x
                ttv_vec[i][x, :, :] = V'[1:rks[i], i_vec]
            end

            tensor_curr = U[:, 1:rks[i]] * Diagonal(S[1:rks[i]])
        end
    end

    # center : i = index
    tensor_curr = reshape(tensor_curr, dims[index] * rks[index], :)
    ttv_vec[index] = zeros(T, dims[index], rks[index], rks[index+1])
    for x = 1:dims[index]
        ttv_vec[index][x, :, :] =
            tensor_curr[(rks[index]*(x-1)+1):(rks[index]*x), 1:rks[index+1]]
    end

    return TTvector{T,d}(d, ttv_vec, dims, rks, ttv_ot)
end

#takes the TT representation of a tensor and returns the closest rank <= r TT tensor
function ttv_svd_r(tt::TTvector{T,N}; tol=1e-12, rmax=typemax(Int64)) where {T<:Number,N}
    dims  = tt.ttv_dims
    rks   = copy(tt.ttv_rks)
    cores = deepcopy(tt.ttv_vec)

    for i = 1:(N-1)
        mat = reshape(cores[i], dims[i] * rks[i], rks[i+1])

        F = qr(mat)
        Q = Matrix(F.Q)
        R = Matrix(F.R)

        tmp_rk = size(Q, 2)

        cores[i] = reshape(Q, dims[i], rks[i], tmp_rk)
        old_rkp1 = rks[i+1]
        rks[i+1] = tmp_rk

        mat2 = reshape(permutedims(cores[i+1], (2,1,3)), old_rkp1, dims[i+1] * rks[i+2])
        mat2 = R*mat2

        cores[i+1] = permutedims(reshape(mat2, tmp_rk, dims[i+1], rks[i+2]), (2,1,3))
    end

    for i = N:(-1):2
        mat = reshape(permutedims(cores[i], (2,1,3)), rks[i], dims[i] * rks[i+1])

        F = svd(mat)
        U = F.U
        S = F.S
        V = F.V

        keep = min(rmax, count(>=(tol), S))
        keep = max(1, keep)

        U = U[:, 1:keep]
        S = S[1:keep]
        V = V[:, 1:keep]

        old_rki = rks[i]
        rks[i] = keep

        Vt = V'
        cores[i] = permutedims(reshape(Vt, keep, dims[i], size(cores[i], 3)), (2,1,3))

        US = U * Diagonal(S)

        mat_prev = reshape(cores[i-1], dims[i-1] * rks[i-1], old_rki)

        mat_prev = mat_prev * US

        cores[i-1] = reshape(mat_prev, dims[i-1], rks[i-1], keep)
    end

    ot = ones(Int64, N)
    ot[1] = 0

    return TTvector{T,N}(N, cores, dims, rks, ot)
end

# psi_(n+1) = psi_n - timestep * i * H * psi_n
function time_evolution(state::TTvector{ComplexF64, N}, hamiltonian::TToperator{ComplexF64, N}, t::Float64, timestep::Float64) where {N}
    while(t > 0)
        t -= timestep
        state = state - complex(timestep * im) * hamiltonian * state
    end
    return state
end

# psi_(n+1) = TT-SVD_r(psi_n - timestep * i * H * psi_n)
function time_evolution_MPS(state::TTvector{ComplexF64, N}, hamiltonian::TToperator{ComplexF64, N}, t::Float64, timestep::Float64; rmax::Int64=typemax(Int64)) where {N}
    while(t > 0)
        t -= timestep
        state = ttv_svd_r(state - im * complex(timestep) * hamiltonian * state; rmax=rmax)
    end
    return state
end

# psi_(n+1) = TT-SVD_r(psi_n - timestep * P_T_MPS * i * H * psi_n)
function time_evolution_tangent_proj(state::TTvector{ComplexF64, N}, hamiltonian::TToperator{ComplexF64, N}, t::ComplexF64, timestep::ComplexF64) where {N}
    while t > 0
        t -= timestep
        # TODO
    end
    return state
end