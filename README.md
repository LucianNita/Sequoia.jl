[//]: Logo
<p align="center">
<img
    src="./docs/src/assets/sequoia_logo.svg"
    width=256px
    >
</p>

# Sequoia.jl

[![codecov](https://codecov.io/github/LucianNita/Sequoia.jl/graph/badge.svg?token=WcYswle2ml)](https://codecov.io/github/LucianNita/Sequoia.jl)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://luciannita.github.io/Sequoia.jl/dev)
[![CI](https://github.com/LucianNita/Sequoia.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LucianNita/Sequoia.jl/actions/workflows/CI.yml)

`Sequoia.jl` is a Julia package designed for solving constrained nonlinear optimization problems. It provides flexible and efficient implementations of popular optimization algorithms like the Quadratic Penalty Method (QPM), Augmented Lagrangian Method (ALM), Interior Point Method (IPM), and SEQUOIA (Sequential Optimization Implicit Algorithm). The package is compatible with `CUTEst.jl` for benchmarking and allows users to define custom constraints and objective functions.

---

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [API Reference](#api-reference)
5. [Contributing](#contributing)
6. [License](#license)

---

## Features

- **Multiple Solvers**:
  - Quadratic Penalty Method (QPM)
  - Augmented Lagrangian Method (ALM)
  - Interior Point Method (IPM)
  - SEQUOIA (Sequential Quadratic Approximation)
- **Constraint Support**:
  - Equality constraints
  - Inequality constraints
  - Bound constraints
- **Customizable**: Users can define their own objective functions and constraints as well as use already implemented examples from CUTEst library.
- **CUTEst Compatibility**: Benchmark optimization algorithms with `CUTEst.jl`.

---

## Installation

To install `Sequoia.jl`, use the Github link:

```julia
using Pkg
Pkg.add(url="https://github.com/username/Sequoia.jl.git")
```

Note CUTEst does not work on Windows. To use CUTEst from Windows, follow the instructions in the link "https://jso.dev/CUTEst.jl/v0.13/"

---

## Getting Started 
 
This is a quick example on solving an optimization problem with Sequoia.jl:

Example: Quadratic Objective with Linear Constraints

```julia
using Sequoia

# Define the optimization problem
problem = SEQUOIA_pb(
    2;  # Number of variables
    x0 = [1.5, 1.5],  # Initial guess
    objective = x -> sum(x.^2),  # Quadratic objective
    gradient = x -> 2 .* x,  # Gradient of the objective
    constraints = x -> [x[1] + x[2] - 2, x[2] - 1],  # Constraints
    eqcon = [1],  # Equality constraint index
    ineqcon = [2],  # Inequality constraint index
    jacobian = x -> [1.0 1.0; 0.0 1.0],  # Jacobian of constraints
    solver_settings = SEQUOIA_Settings(
        :QPM,  # Solver method
        :LBFGS,  # Inner solver
        false,  # No warm start
        1e-8,  # Residual tolerance
        100,  # Max iterations
        10.0,  # Max time
        1e-6;  # Cost tolerance
        solver_params = [1.0, 10.0, 10.0, 0.0]
    )
)

# Solve the problem
solve!(problem)

# Display the solution
println("Optimal solution: ", problem.solution_history.iterates[end].x)
println("Objective value: ", problem.objective(problem.solution_history.iterates[end].x))
```

---

## API Reference 

For a detailed API reference, visit the [documentation](https://luciannita.github.io/Sequoia.jl/dev/). 
---

## Contributing 

We welcome contributions to Sequoia.jl! Here is how you can help:

1. Fork the repository.
2. Create a feature branch:
```bash
git checkout -b feature-branch
```
3. Commit your changes:
```bash
git commit -m "Add new feature"
```
4. Push your branch:
```bash
git push origin feature-branch
```
5. Open a pull request.
Before contributing, please review our contributing guidelines. (add link here)

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/LucianNita/Sequoia.jl/blob/main/LICENSE) file for details. (add link)

---