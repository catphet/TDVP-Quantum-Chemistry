using TensorTrains
using LinearAlgebra

res = build_hamiltonian_stupid(3, complex(1.), complex(1.))
res_tto = build_hamiltonian_tto(3, complex(1.), complex(1.))

display(res)
display(tto_to_matrix(res_tto))