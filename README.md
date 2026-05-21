# TDVP-Quantum-Chemistry
Implementing and Benchmarking TDVP for Quantum Chemistry applications

## Setup

0. Check you have the correct Julia version:
   julia --version
   This project was developed with Julia 1.12.6
   Download the correct version at https://julialang.org/downloads/

1. Navigate to the TensorTrainsModule folder and install dependencies:
   
   ```
   cd TensorTrainsModule
   julia --project=.
   ```
   
2. Then in the Julia REPL: 
   
   ```import Pkg; Pkg.instantiate()```

3. Navigate to the advrproject folder: 

   ```
   exit()
   cd AdvRProject
   ```
   
4. Call desired validation script (test.jl, validate_n3.jl, or validate_n5.jl):

   ```
   julia --project=.. validate_n3.jl

   kp note

   julia --project=.. compile_results.jl
   .\run_all_validations.ps1
   ```
