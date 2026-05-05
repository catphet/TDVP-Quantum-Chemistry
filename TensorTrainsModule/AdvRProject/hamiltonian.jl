using .TensorTrains

function pauli_x()
    return ComplexF64[0 1; 1 0]
end

function pauli_y()
    return ComplexF64[0 -im; im 0]
end

function id()
    return ComplexF64[1 0; 0 1]
end

# returns id_2 otimes id_2 otimes ... otimes mat otimes ... otimes id_2
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

    for k in 2:L

        W = A.tensors[k]
        r_left, r_right, d_in, d_out = size(W)
        T_perm = permutedims(T, vcat(2:ndims(T), 1))
        T_mat = reshape(T_perm, :, r_left)
        W_mat = reshape(W, r_left, :)
        res_raw = T_mat * W_mat
        T = reshape(res_raw, size(T_perm)[1:end-1]..., r_right, d_in, d_out)
        n_dims = ndims(T)
        T = permutedims(T, vcat(n_dims-2, 1:n_dims-3, n_dims-1, n_dims))
    end

    T_final = T[1, :, :, :, :, :, :]
    
    perm = vcat(1:2:2*L-1, 2:2:2*L)
    T_permuted = permutedims(dropdims(T, dims=1), perm)
    
    return reshape(T_permuted, 2^L, 2^L)
end