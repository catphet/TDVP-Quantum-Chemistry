# TDVP-Quantum-Chemistry
Implementing and Benchmarking TDVP for Quantum Chemistry applications

## Setup

1. Check you have the correct Julia version:
   julia --version
   This project was developed with Julia [paste your version here]
   Download the correct version at https://julialang.org/downloads/

2. Navigate to the TensorTrainsModule folder and install dependencies:
   cd TensorTrainsModule
   julia --project=.
   Then in the Julia REPL: import Pkg; Pkg.instantiate()

3. Navigate to the advrproject folder and run:
   cd advrproject
   julia --project=.. test.jl