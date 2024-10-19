using Test
using LinearAlgebra
using Optim

# Include your code here (the penalty update functions, QPM solver, etc.)

# ---------------------- Unit Tests ---------------------- #

# Helper function to run the QPM solver with different strategies
function run_qpm_test(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver, update_fn, penalty_init=1.0)
    x_opt, final_penalty = qpm_solve(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver,
                                     penalty_init=penalty_init, penalty_mult=10.0, tol=1e-6, max_iter=1000, damping_factor=10.0, update_fn=update_fn)
    return x_opt, final_penalty
end

# Test the penalty update functions
@testset "Penalty Update Functions" begin
    # Fixed penalty update test
    @test fixed_penalty_update(1.0, 10.0) == 10.0
    @test fixed_penalty_update(0.5, 2.0) == 1.0

    # Adaptive penalty update test
    @test adaptive_penalty_update(1.0, 0.1, 1e-6, 10.0) ≈ 10.0  # Big violation
    @test adaptive_penalty_update(1.0, 1e-8, 1e-6, 10.0) ≈ 1.0  # Small violation
    @test adaptive_penalty_update(1.0, 1e-8, 1e-6, 2.0) ≈ 1.0   # Damping factor caps the penalty increase
end

# Test QPM solver on simple quadratic problem (already given in the example)
@testset "QPM Solver Simple Problem" begin
    # Define the objective function: f(x) = x₁² + x₂²
    obj_fn = x -> x[1]^2 + x[2]^2
    grad_fn = x -> [2*x[1], 2*x[2]]
    
    # Define a constraint function: g(x) = [x₁ + x₂, x₁]
    cons_fn = x -> [x[1] + x[2], x[1]]
    cons_jac_fn = x -> [1.0 1.0; 1.0 0.0]
    
    # Define bounds and indices for constraints
    lb = [1.0, -Inf]
    ub = [1.0, 0.3]
    eq_indices = [1]
    ineq_indices = [2]
    x0 = [0.25, 0.75]

    # Solve using fixed penalty strategy
    x_opt_fixed, final_penalty_fixed = run_qpm_test(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, Optim.LBFGS(), fixed_penalty_update)
    
    # Check that the result is correct (it should be [0.3, 0.7])
    @test norm(x_opt_fixed .- [0.3, 0.7]) < 1e-6

    # Solve using adaptive penalty strategy
    x_opt_adaptive, final_penalty_adaptive = run_qpm_test(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, Optim.LBFGS(), adaptive_penalty_update)
    
    # Check that the result is correct (it should be [0.3, 0.7])
    @test norm(x_opt_adaptive .- [0.3, 0.7]) < 1e-6
end

# Test for unconstrained problem
@testset "Unconstrained Problem" begin
    # Define an unconstrained quadratic objective function
    obj_fn = x -> x[1]^2 + x[2]^2
    grad_fn = x -> [2*x[1], 2*x[2]]
    
    # No constraints
    cons_fn = x -> Float64[]
    cons_jac_fn = x -> zeros(0, length(x))
    lb = Float64[]
    ub = Float64[]
    eq_indices = Int[]
    ineq_indices = Int[]
    x0 = [1.0, 1.0]

    # Solve with fixed penalty (should converge to [0.0, 0.0])
    x_opt, final_penalty = run_qpm_test(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, Optim.LBFGS(), fixed_penalty_update)
    @test norm(x_opt .- [0.0, 0.0]) < 1e-6
end

# Test for tight constraints
@testset "Tight Constraints" begin
    # Define objective function: f(x) = (x₁ - 1)² + (x₂ - 1)²
    obj_fn = x -> (x[1] - 1)^2 + (x[2] - 1)^2
    grad_fn = x -> [2*(x[1] - 1), 2*(x[2] - 1)]

    # Define tight constraint g(x) = [x₁ + x₂]
    cons_fn = x -> [x[1] + x[2]]
    cons_jac_fn = x -> [1.0 1.0]

    # Bounds and indices
    lb = [1.0]  # Equality constraint x₁ + x₂ = 1
    ub = [1.0]
    eq_indices = [1]
    ineq_indices = Int[]
    x0 = [0.5, 0.5]

    # Solve with adaptive penalty
    x_opt, final_penalty = run_qpm_test(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, Optim.LBFGS(), adaptive_penalty_update)
    
    # Check that the result satisfies x₁ + x₂ = 1 (within tolerance)
    @test norm(sum(x_opt) - 1.0) < 1e-6
end

# Test for a more complex problem with nonlinear constraints
@testset "Nonlinear Constraints" begin
    # Define a nonlinear objective function: f(x) = (x₁ - 1)² + (x₂ - 2)²
    obj_fn = x -> (x[1] - 1)^2 + (x[2] - 2)^2
    grad_fn = x -> [2*(x[1] - 1), 2*(x[2] - 2)]

    # Define nonlinear constraints g(x) = [x₁^2 + x₂]
    cons_fn = x -> [x[1]^2 + x[2]]
    cons_jac_fn = x -> [2*x[1] 1.0]

    # Bounds for constraints
    lb = [2.0]  # x₁² + x₂ >= 2
    ub = [Inf]
    eq_indices = Int[]
    ineq_indices = [1]
    x0 = [0.5, 1.5]

    # Solve with adaptive penalty
    x_opt, final_penalty = run_qpm_test(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, Optim.LBFGS(), adaptive_penalty_update)
    
    # Check that the result satisfies the constraint x₁² + x₂ >= 2
    @test x_opt[1]^2 + x_opt[2] >= 2.0
end
