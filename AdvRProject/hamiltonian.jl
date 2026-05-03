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

# computes the long-range hamiltonian that can be found in "Unifying blabla...", p.4 eq.(7)
function build_hamiltonian_stupid(N::Int64, alpha::ComplexF64, J::ComplexF64)
    res = zeros(ComplexF64, 2^N, 2^N)
    for i in 1:(N-1)
        for j in (i+1):N
            sig_i_x = kron_N_sites(pauli_x(), N, i)
            sig_j_x = kron_N_sites(pauli_x(), N, j)
            sig_i_y = kron_N_sites(pauli_y(), N, i)
            sig_j_y = kron_N_sites(pauli_y(), N, j)

            h_ij = sig_i_x * sig_j_x + sig_i_y * sig_j_y

            res = res + (1 / abs(i - j)^alpha) * h_ij
        end
    end
    return J/2 * res
end

####### using TT representation

#takes a 2x2 matrix and returns the corresponding 2x2x1x1 tto core
function matrix_to_tto_core(mat::Matrix{ComplexF64})
    res = zeros(ComplexF64, 2, 2, 1, 1)
    for i in 1:2
        for j in 1:2
            res[i, j, 1, 1] = mat[i, j]
        end
    end
    return res
end

#takes a list of 2x2 matrices and returns the tto operator corresponding to mat_1 otimes mat_2 otimes ... otimes mat_3
function kron_tto(mats::Vector{Matrix{ComplexF64}})
    nsites = length(mats)
    cores = Vector{Array{ComplexF64, 4}}(undef, nsites)

    for i in 1:nsites
        cores[i] = matrix_to_tto_core(mats[i])
    end

    return TToperator{ComplexF64, nsites}(nsites, cores, ntuple(_->2, nsites), ones(Int64, nsites+1), zeros(Int64, nsites))
end

# computes the long-range hamiltonian that can be found in "Unifying blabla...", p.4 eq.(7)
function build_hamiltonian_tto(N::Int64, alpha::ComplexF64, J::ComplexF64)
    res = zeros_tto(ComplexF64, ntuple(_->2, N), ones(Int64, N+1))

    ## that loop is ugly !!!!
    for i = 1:(N-1)
        for j = (i+1):N
            pauli_i_x = Vector{Matrix{ComplexF64}}(undef, N)
            pauli_i_y = Vector{Matrix{ComplexF64}}(undef, N)
            pauli_j_x = Vector{Matrix{ComplexF64}}(undef, N)
            pauli_j_y = Vector{Matrix{ComplexF64}}(undef, N)
            for k in 1:N
                pauli_i_x[k] = id()
                pauli_i_y[k] = id()
                pauli_j_x[k] = id()
                pauli_j_y[k] = id()
            end
            pauli_i_x[i] = pauli_x()
            pauli_i_y[i] = pauli_y()
            pauli_j_x[j] = pauli_x()
            pauli_j_y[j] = pauli_y()

            pauli_i_x_tto = kron_tto(pauli_i_x)
            pauli_i_y_tto = kron_tto(pauli_i_y)
            pauli_j_x_tto = kron_tto(pauli_j_x)
            pauli_j_y_tto = kron_tto(pauli_j_y)

            h_ij = pauli_i_x_tto * pauli_j_x_tto + pauli_i_y_tto * pauli_j_y_tto

            res = res + (1/abs(i - j)^alpha) * h_ij
        end
    end
    return J/2 * res
end

# takes a tto and returns the corresponding matrix, just for debugging
function tto_to_matrix(A::TToperator)
    N = A.N
    dims = A.tto_dims

    T = A.tto_vec[1][:, :, 1, :]

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