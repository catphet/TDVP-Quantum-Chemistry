using TensorTrains
using LinearAlgebra

pauli_x() = ComplexF64[0 1; 1 0]
pauli_y() = ComplexF64[0 -im; im 0]
id()      = ComplexF64[1 0; 0 1]

function kron_N_sites(mat, N, index)
    res = ComplexF64[1]
    for i in 1:N
        res = i == index ? kron(res, mat) : kron(res, id())
    end
    return res
end

function build_hamiltonian_stupid(N::Int64, alpha::ComplexF64, J::ComplexF64)
    res = zeros(ComplexF64, 2^N, 2^N)
    for i in 1:(N-1), j in (i+1):N
            h_ij = kron_N_sites(pauli_x(), N, i) * kron_N_sites(pauli_x(), N, j) + 
               kron_N_sites(pauli_y(), N, i) * kron_N_sites(pauli_y(), N, j)
        res += (1 / abs(i - j)^alpha) * h_ij
    end
    return (J/2) * res
end

function matrix_to_tto_core(mat::Matrix{ComplexF64})
    return reshape(mat, (1, 1, 2, 2))
end

function kron_tto(mats::Vector{Matrix{ComplexF64}})
    cores = [matrix_to_tto_core(m) for m in mats]
    return TensorTrain(cores)
end

function build_hamiltonian_tto(N::Int64, alpha::ComplexF64, J::ComplexF64)
    res = nothing

    for i = 1:(N-1)
        for j = (i+1):N
            m_x = [id() for _ in 1:N]; m_x[i] = pauli_x(); m_x[j] = pauli_x()
            m_y = [id() for _ in 1:N]; m_y[i] = pauli_y(); m_y[j] = pauli_y()

            coeff = (1 / abs(i - j)^alpha)
            h_ij = kron_tto(m_x) + kron_tto(m_y)

            for k in 1:N
                h_ij.tensors[k] .*= coeff^(1/N)
            end

            if res === nothing
                res = h_ij
            else
                res = res + h_ij
                compress!(res; svd_trunc = TruncThresh(1e-12))
            end
        end
    end

    final_scale = J/2
    for k in 1:N
        res.tensors[k] .*= final_scale^(1/N)
    end
    
    return res
end

function tto_to_matrix(A::TensorTrain)
    L = length(A)
    T = A.tensors[1][1, :, :, :] 

    for k in 2:N
        W = A.tto_vec[k]

        nk = size(W, 1)
        rk_prev = size(W, 3)
        rk = size(W, 4)

        sizeT = size(T)
        left_dims = sizeT[1:end-1]
        prod_left = prod(left_dims)

        Tmat = reshape(T, prod_left, rk_prev)
        Wmat = reshape(permutedims(W, (3,1,2,4)), rk_prev, :)

        Tnew = Tmat * Wmat

        T = reshape(Tnew, left_dims..., nk, nk, rk)
    end

    T = dropdims(T; dims=ndims(T))

    perm = vcat(collect(1:2:2N-1), collect(2:2:2N))
    T = permutedims(T, perm)

    d = prod(dims)
    return reshape(T, d, d)
end