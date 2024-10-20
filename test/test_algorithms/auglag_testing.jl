using Test
using LinearAlgebra
using Optim

# Import the ALM functions (alm_solve, fixed_penalty_update, adaptive_penalty_update)
# Assuming the alm_solve and related functions are saved in `alm_module.jl`, 
# otherwise just ensure they're in the same file.

# Basic quadratic problem with linear constraints
obj_fn = x -> x[1]^2 + x[2]^2  # Objective: f(x) = x₁² + x₂²
grad_fn = x -> [2*x[1], 2*x[2]]  # Gradient of the objective
cons_fn = x -> [x[1] + x[2], x[1]]  # Constraint function: g(x) = [x₁ + x₂, x₁]
cons_jac_fn = x -> [1.0 1.0; 1.0 0.0]  # Jacobian of the constraint function

lb = [1.0, -Inf]  # Lower bounds for the constraints
ub = [1.0, 0.3]  # Upper bounds for the constraints
eq_indices = [1]  # First constraint is an equality
ineq_indices = [2]  # Second constraint is an inequality
x0 = [0.25, 0.75]  # Initial guess
inner_solver = Optim.LBFGS()  # Using LBFGS as the inner solver

# Begin test suite
@testset "Augmented Lagrangian Method Tests" begin
    
    # Test 1: Basic functionality with fixed penalty update
    @testset "Fixed penalty update strategy" begin
        x_opt_fixed, final_penalty_fixed, final_lambda_fixed = alm_solve(
            obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, 
            x0, inner_solver, penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, update_fn=fixed_penalty_update
        )
        
        # Check if the optimal solution is close to the known solution (approximately [0.5, 0.5])
        @test isapprox(x_opt_fixed, [0.5, 0.5]; atol=1e-3)
        @test final_penalty_fixed > 0  # Ensure penalty parameter is updated
        @test norm(final_lambda_fixed) > 0  # Check Lagrange multipliers were updated
    end

    # Test 2: Basic functionality with adaptive penalty update
    @testset "Adaptive penalty update strategy" begin
        x_opt_adaptive, final_penalty_adaptive, final_lambda_adaptive = alm_solve(
            obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, 
            x0, inner_solver, penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, 
            damping_factor=10.0, update_fn=adaptive_penalty_update
        )

        # Check if the optimal solution is close to the known solution
        @test isapprox(x_opt_adaptive, [0.5, 0.5]; atol=1e-3)
        @test final_penalty_adaptive > 0  # Ensure penalty parameter is updated
        @test norm(final_lambda_adaptive) > 0  # Check Lagrange multipliers were updated
    end

    # Test 3: Warm start with Lagrange multipliers
    @testset "Warm start with Lagrange multipliers" begin
        # Use previously computed Lagrange multipliers for warm start
        x_opt_warm, final_penalty_warm, final_lambda_warm = alm_solve(
            obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, 
            x0, inner_solver, penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, 
            update_fn=fixed_penalty_update, λ_init=final_lambda_fixed
        )

        # Check if the warm-started solution converges correctly
        @test isapprox(x_opt_warm, [0.5, 0.5]; atol=1e-3)
        @test isapprox(final_lambda_warm, final_lambda_fixed; atol=1e-3)  # Check Lagrange multipliers match previous solution
    end

    # Test 4: No inequality constraints (edge case)
    @testset "No inequality constraints" begin
        # Solve without inequality constraints
        ineq_indices_empty = []  # No inequality constraints
        x_opt_no_ineq, _, _ = alm_solve(
            obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices_empty, 
            x0, inner_solver, penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, update_fn=fixed_penalty_update
        )

        # Check if the solution is valid even without inequalities
        @test isapprox(x_opt_no_ineq, [0.5, 0.5]; atol=1e-3)
    end

    # Test 5: Edge case with zero bounds (testing bounds handling)
    @testset "Edge case with tight bounds" begin
        lb_tight = [1.0, 0.0]  # Tighter lower bounds
        ub_tight = [1.0, 0.3]  # Upper bounds remain the same
        
        x_opt_tight, _, _ = alm_solve(
            obj_fn, grad_fn, cons_fn, cons_jac_fn, lb_tight, ub_tight, eq_indices, ineq_indices, 
            x0, inner_solver, penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, update_fn=fixed_penalty_update
        )

        # Check that the solution adheres to the tighter bounds
        @test x_opt_tight[2] >= 0.0  # Ensure the solution adheres to the tight lower bound
    end

    # Test 6: Degenerate problem (multiple identical constraints)
    @testset "Degenerate constraints" begin
        cons_fn_degenerate = x -> [x[1] + x[2], x[1] + x[2]]  # Redundant constraints
        cons_jac_fn_degenerate = x -> [1.0 1.0; 1.0 1.0]  # Identical Jacobian rows

        x_opt_degenerate, _, _ = alm_solve(
            obj_fn, grad_fn, cons_fn_degenerate, cons_jac_fn_degenerate, lb, ub, eq_indices, eq_indices, 
            x0, inner_solver, penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, update_fn=fixed_penalty_update
        )

        # Check if the solution handles degeneracy correctly
        @test isapprox(x_opt_degenerate, [0.5, 0.5]; atol=1e-3)
    end

end
