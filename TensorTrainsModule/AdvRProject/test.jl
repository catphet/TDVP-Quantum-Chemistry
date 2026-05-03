include("..\\src\\TensorTrains.jl")
using .TensorTrains
include("hamiltonian.jl")
include("timeevolution.jl")
using LinearAlgebra

C = 5 # particles count
rmax = 100

println(typejoin(ComplexF64, ComplexF64))

print("0> starting...\n")

H = build_hamiltonian_tto(C, complex(1.), complex(1.))
print("1> hamiltonian done.\n")
println("type H : " * string(typeof(H)))

t = 1.
timestep = 0.01

rks = typemax(Int64)*ones(Int,C+1)
rks = r_and_d_to_rks(rks,ntuple(_->2, C))
state = rand_tt(ComplexF64, ntuple(_ -> 2, C), rks)
print("1.5> state generation done.\n")
println("type state : " * string(typeof(state)))

#res1 = time_evolution(state, H, t, timestep)
#print("2> time evolution 1 done.\n")

res2 = time_evolution_MPS(state, H, t, timestep; rmax=rmax)
print("3> time evolution 2 done.\n")