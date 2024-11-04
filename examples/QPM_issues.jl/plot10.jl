using LinearAlgebra
using Optim
using Plots

### Problem Definition with Non-Convex Cubic Objective

# Define the objective function f(x) = x[1]^3 + x[2]^3
function f(x::Vector)
    return x[1]^3 + x[2]^3
end

# Equality constraint: x + y = 0
function g(x::Vector)
    return x[1] + x[2]
end

# Struct to store the transformed problem
struct TransformedProblem
    f::Function
    g::Function
end

# Define the problem
problem = TransformedProblem(f, g)

### Sequential Feasibility-Based SEQUOIA Solver

function feasibility_objective(x, t, λ_g, μ, problem::TransformedProblem)
    # Augmented objective: constant 1 + penalties for constraints
    f_penalty = max(0, problem.f(x) - t)  # Inequality constraint f(x) - t <= 0
    g_penalty = problem.g(x)              # Equality constraint g(x) = 0
    return 1 + λ_g * g_penalty + 0.5 * μ * g_penalty^2 + 0.5 * μ * f_penalty^2
end

function feasibility_gradient!(grad, x, t, λ_g, μ, problem::TransformedProblem)
    # Compute gradient for x only
    grad_x = zeros(length(x))

    # Gradients with respect to constraints
    if problem.f(x) - t > 0
        grad_x .= 3 * x.^2 * μ * (problem.f(x) - t)
    end

    if problem.g(x) != 0
        grad_x .= grad_x .+ μ * problem.g(x)
    end

    grad .= grad_x
end

# Main SEQUOIA solver function with sequential feasibility
function sequoia_solve(problem::TransformedProblem, x0, inner_solver, options)
    # Initialize variables
    x = x0                # Initial guess for x
    t = problem.f(x)      # Initialize t as the current objective value
    λ_g = 0.0             # Lagrange multiplier for g(x)
    μ = 10.0              # Initial penalty parameter
    tol = 1e-5            # Convergence tolerance
    max_iter = 50         # Maximum outer iterations
    γ, β = 2.0, 0.5       # Parameters for adjusting step size

    solution_history = []
    
    for iter in 1:max_iter
        # Define the feasibility problem with current t
        feasibility_obj = x -> feasibility_objective(x, t, λ_g, μ, problem)
        feasibility_grad! = (g, x) -> feasibility_gradient!(g, x, t, λ_g, μ, problem)

        # Solve the feasibility problem
        result = optimize(feasibility_obj, feasibility_grad!, x, inner_solver, options)
        display(Optim.x_trace(result))
        x = Optim.minimizer(result)
        
        # Check if the solution is feasible
        f_violation = max(0, problem.f(x) - t)
        g_violation = abs(problem.g(x))

        # Store iterate for history and plotting
        push!(solution_history, (x, t))

        # Update t based on feasibility
        if f_violation <= tol && g_violation <= tol
            # Problem is feasible; decrease t to find a tighter upper bound
            t -= β * abs(t)  # Reduce t by a small factor
            println("Feasible at iteration $iter; decreasing t to $t")
        else
            # Problem is infeasible; increase t to relax feasibility requirement
            t += γ * abs(t)  # Increase t by a factor
            println("Infeasible at iteration $iter; increasing t to $t")
        end

        # Update penalty parameter and multiplier
        λ_g += μ * g_violation
        μ *= 1.1  # Mildly increase penalty parameter for robustness

        # Convergence check
        if abs(t) < tol
            println("Converged within tolerance at iteration $iter")
            break
        end
    end

    return solution_history
end

# Define inner solver and options for Optim.jl
inner_solver = Optim.BFGS()
options = Optim.Options(store_trace=true, extended_trace=true,show_trace=false)

# Run SEQUOIA solve with initial guess
x0 = [0.5, 0.5]
solution_history = sequoia_solve(problem, x0, inner_solver, options)

### Visualization

# Set up grid for contour plot
x_vals = -1.0:0.05:1.0
y_vals = -1.0:0.05:1.0
Z_objective = [f([x, y]) for x in x_vals, y in y_vals]

# Plot objective function contour
contour(x_vals, y_vals, Z_objective, levels=20, xlabel="x", ylabel="y", title="Objective and Constraint with SEQUOIA Iterates", color=:blues)

# Overlay the constraint line x + y = 0 (y = -x)
plot!(x_vals, -x_vals, color=:red, linewidth=2, label="x + y = 0 (constraint)")

# Plot SEQUOIA iterates
x_iterates = [iter[1][1] for iter in solution_history]
y_iterates = [iter[1][2] for iter in solution_history]
plot!(x_iterates, y_iterates, marker=:o, markersize=4, label="SEQUOIA Iterates", color=:green, linewidth=2)

# Display the final plot
display(plot!)
