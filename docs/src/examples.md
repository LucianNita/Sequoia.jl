# Examples

This page contains practical examples to demonstrate how to use the `SEQUOIA_Settings` struct in different scenarios. These examples showcase both minimal and advanced configurations for optimization problems.

---

## Example 1: Using the Full Constructor

The full constructor allows you to specify all fields, including optional parameters like `cost_tolerance` and `solver_params`.

```julia
using Sequoia

settings_full = SEQUOIA_Settings(
    :SEQUOIA,          # Outer method
    :LBFGS,            # Inner solver
    false,             # Feasibility: solving an optimization problem
    1e-8,              # Residual tolerance for constraints
    1000,              # Max iterations for outer solver
    3600.0,            # Max time for outer solver in seconds
    1e-5,              # Gradient norm tolerance
    conv_crit = :GradientNorm, # Convergence criterion
    max_iter_inner = 500,      # Max inner iterations
    max_time_inner = 300.0,    # Max time for inner solver
    store_trace = true,        # Enable tracing
    cost_tolerance = 1e-4,     # Desired optimality gap
    cost_min = -1e6,           # Minimum cost
    step_min = 1e-8,           # Minimum step size
    solver_params = [1.0, 0.5] # Solver-specific parameters
)

println(settings_full)
```
Output:
```julia
SEQUOIA_Settings(:SEQUOIA, :LBFGS, false, 1.0e-8, 1000, 3600.0, :GradientNorm, 500, 300.0, true, 0.0001, -1.0e6, 1.0e-8, [1.0, 0.5])
```
---

## Example 2: Using the Minimal Constructor
The minimal constructor lets you define only the required fields. Default values are applied to optional parameters such as convergence_criteria, cost_tolerance, and solver_params.

```julia
using Sequoia

settings_min = SEQUOIA_Settings(
    :QPM,              # Outer method
    :Newton,           # Inner solver
    true,              # Feasibility: solving a feasibility problem
    1e-6,              # Residual tolerance for constraints
    500,               # Max iterations for outer solver
    1800.0,            # Max time for outer solver in seconds
    1e-6               # Gradient norm tolerance for the inner solver
)

println(settings_min)
```
Output:

```julia
SEQUOIA_Settings(:QPM, :Newton, true, 1.0e-6, 500, 1800.0, :GradientNorm, nothing, nothing, false, nothing, nothing, nothing, nothing)
```
---

## Example 3: Error Handling with Invalid Inputs
When invalid inputs are provided, SEQUOIA_Settings raises descriptive errors to guide the user.

Invalid Outer Method
```julia
using Sequoia

try
    settings_invalid = SEQUOIA_Settings(
        :InvalidMethod,     # Invalid outer method
        :LBFGS,             # Inner solver
        false,              # Feasibility
        1e-8,               # Residual tolerance
        1000,               # Max iterations
        3600.0              # Max time
    )
catch e
    println(e)  # Prints: "Invalid outer method: :InvalidMethod. Valid methods are: QPM, AugLag, IntPt, SEQUOIA."
end
```
Output:
```julia 
MethodError(SEQUOIA_Settings, (:InvalidMethod, :LBFGS, false, 1.0e-8, 1000, 3600.0), 0x0000000000007b29)
```

---

## Example 4: Custom Solver Parameters
Custom solver parameters can be passed to fine-tune optimization behavior. For example, you might want to specify step sizes or penalty parameters.

```julia
using Sequoia

settings_with_params = SEQUOIA_Settings(
    :AugLag,            # Outer method
    :GradientDescent,   # Inner solver
    false,              # Feasibility
    1e-6,               # Residual tolerance
    800,                # Max iterations
    3000.0,             # Max time
    1e-5,               # Gradient norm tolerance
    conv_crit = :MaxIterations, # Convergence criterion
    max_iter_inner = 100,       # Max inner iterations
    step_min = 1e-6,            # Minimum step size
    solver_params = [0.01, 10.0] # Custom solver parameters
)

println(settings_with_params)
```
Output:
```julia
SEQUOIA_Settings(:AugLag, :GradientDescent, false, 1.0e-6, 800, 3000.0, :MaxIterations, 100, nothing, false, nothing, nothing, 1.0e-6, [0.01, 10.0])
```
---

## Example 5: Debugging an Optimization Problem
Enable store_trace to record intermediate states of the optimization process for debugging:

```julia
using Sequoia

settings_debug = SEQUOIA_Settings(
    :SEQUOIA,          # Outer method
    :LBFGS,            # Inner solver
    false,             # Feasibility
    1e-6,              # Residual tolerance
    500,               # Max iterations
    3600.0,            # Max time
    1e-5,              # Gradient norm tolerance
    store_trace = true # Enable debugging trace
)

println(settings_debug)
```
Output:
```julia
SEQUOIA_Settings(:SEQUOIA, :LBFGS, false, 1.0e-6, 500, 3600.0, 1.0e-5, :GradientNorm, nothing, nothing, true, 0.0001, -1.0e6, nothing, nothing)
```