# Getting Started

## Installation

To install `SEQUOIA.jl`, use Julia's package manager:
```julia
using Pkg
Pkg.add("Sequoia")

Basic Usage
To configure an optimization problem, create a SEQUOIA_Settings object:
```julia
using Sequoia

settings = SEQUOIA_Settings(
    :SEQUOIA, :LBFGS, false, 1e-8, 1000, 3600.0, 1e-5,
    conv_crit = :GradientNorm,
    max_iter_inner = 500
)