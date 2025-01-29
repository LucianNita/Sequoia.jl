```@raw html
<img class="display-light-only" src="assets/sequoia_logo.svg" alt="Sequoia logo"/>
```
# Introduction to Sequoia.jl

Welcome to the documentation for the `SEQUOIA.jl` package!

Sequoia.jl is a powerful and flexible **optimization library** designed to handle constrained and unconstrained problems efficiently. Built for researchers and practitioners, it provides an intuitive interface for solving problems with equality and inequality constraints, leveraging well established optimization methods like the Quadratic Penalty Method (QPM), Augmented Lagrangian Method (ALM), and Interior Point Methods (IPM) as well as the newly developed custom algorithm **Sequoia**.

The package integrates seamlessly with **Optim.jl** for inner solvers and **CUTEst.jl** for benchmarking, making it an ideal tool for optimization enthusiasts and professionals tackling real-world problems.

---

## **Key Features**

- **Support for Constraints**:
  - Equality constraints $$ h(x) = 0 $$
  - Inequality constraints $$ g(x) \geq 0 $$
  - Variable bounds $$ l \leq x \leq u $$
  
- **Optimization Methods**:
  - **Quadratic Penalty Method (QPM)**: Smoothly handles constraints by adding penalty terms to the objective.
  - **Augmented Lagrangian Method (ALM)**: Combines Lagrange multipliers with penalties for robust convergence.
  - **Interior Point Method (IPM)**: Ensures feasible solutions by iteratively solving subproblems in the interior of the feasible region.
  
- **Seamless Integration**:
  - Compatible with **Optim.jl** inner solvers such as LBFGS, BFGS, and Newton's method.
  - Easy conversion of **CUTEst** problems for benchmarking and testing.

- **Customizability**:
  - Define your own objectives, constraints, and Jacobians.
  - Fine-tune solver settings, such as convergence criteria, tolerance levels, and penalties.

- **Efficient Implementation**:
  - Handles large-scale problems with sparse matrices.
  - Designed for performance and scalability.

---

## **Why Sequoia.jl?**

Sequoia.jl is built with both simplicity and power in mind. Whether you're solving academic problems or tackling real-world challenges, Sequoia provides:

- **Flexibility**: Solve problems with any combination of constraints and bounds.
- **Ease of Use**: Intuitive interfaces and straightforward workflows for setting up optimization problems.
- **Performance**: State-of-the-art solvers with support for sparsity and large-scale optimization.
- **Comprehensive Support**: Tools for benchmarking, debugging, and analyzing solutions.

---

## **How It Works**

To solve a problem with Sequoia.jl:
1. **Define your problem**: Specify the objective function, constraints, and bounds.
2. **Choose a solver**: Select one of the supported outer methods (e.g., QPM, ALM, IPM).
3. **Run the solver**: Use the `solve!` function to find the optimal solution.
4. **Analyze the results**: Access solution history, final objective value, and constraint violations.

---

## **Example Problem**

Here's an example of a simple constrained optimization problem:

### Problem:

$$\text{Minimize } f(x) = x_1^2 + x_2^2 \quad \text{subject to:}$$

$$x_1 + x_2 - 2 = 0, \quad x_2 - 1 \geq 0, \quad x_1 \in [0, 2], \quad x_2 \in [0, 2]$$

### Code in Sequoia.jl:

```julia
using Sequoia

# Define the problem
problem = SEQUOIA_pb(
    2; 
    x0 = [1.0, 2.0],
    constraints = x -> [x[1] + x[2] - 2, x[2] - 1],
    eqcon = [1],  # Equality constraint index
    ineqcon = [2],  # Inequality constraint index
    jacobian = x -> [1.0 1.0; 0.0 1.0],
    objective = x -> sum(x.^2),
    gradient = x -> 2 .* x
)

# Solve using QPM
problem.solver_settings = SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-8, 100, 10.0, 1e-6)
solve!(problem)

# View results
println("Optimal solution: $(problem.solution_history.iterates[end].x)")
println("Objective value: $(problem.objective(problem.solution_history.iterates[end].x))")
```
