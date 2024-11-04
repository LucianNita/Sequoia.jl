#=
using LinearAlgebra

# Define the barrier parameter (starting value)
μ_initial = 1.0

# Step size parameters for line search
α = 0.01
β = 0.5

# Tolerance for stopping condition
tolerance = 1e-6

# Problem-specific functions
function f(x)
    return x[1]^2 + x[2]^2  # Example: minimize x1^2 + x2^2
end

function ∇f(x)
    return [2*x[1]; 2*x[2]]
end

function g(x)
    return [x[1] - 1, x[2] - 1]  # Example: x1 - 1 ≤ 0, x2 - 1 ≤ 0
end

function ∇g(x)
    return [1 0; 0 1]  # Gradient of constraints x1 - 1 and x2 - 1
end

function h(x)
    return [x[1] + x[2] - 1]  # Example equality constraint: x1 + x2 = 1
end

function ∇h(x)
    return [1 1]  # Gradient of equality constraint
end



# Initial guess
x0 = [2.0, 2.0]  # Initial point for primal variables
s0 = [1.0, 1.0]  # Initial slack variables (real values)
λ0 = [0.5, 0.5]  # Initial Lagrange multipliers for inequality constraints
ν0 = [0.5]       # Initial Lagrange multipliers for equality constraints

# Run the interior point method with quadratic slack variables
x_opt, s_opt, λ_opt, ν_opt = interior_point_quadratic_slack_method(x0, s0, λ0, ν0, μ_initial)

println("Optimal x: ", x_opt)
println("Optimal slack variables: ", s_opt)
println("Optimal Lagrange multipliers for inequality constraints: ", λ_opt)
println("Optimal Lagrange multipliers for equality constraints: ", ν_opt)
=#
#=
using Optim, Sequoia
using LinearAlgebra

# Example 1: Minimizing a quadratic function with an equality constraint
#function example_simple_equality()
    # Objective: Minimize f(x) = (x1 - 2)^2 + (x2 - 3)^2
    objective_fn = x -> (x[1] - 2.0)^2 + (x[2] - 3.0)^2

    # Constraint: x1 + x2 - 5 = 0
    constraints_fn = x -> [x[1] - x[2], x[1] + x[2] - 4.0]

    # Gradient of the objective
    gradient_fn = x -> [2.0 * (x[1] - 2.0), 2.0 * (x[2] - 3.0)]

    # Jacobian of the constraint
    jacobian_fn = x -> [1.0 -1.0; 1.0 1.0] 

    # Initialize SEQUOIA_pb problem
    pb = SEQUOIA_pb(
        2,
        x0 = [0.0, 0.0],                     # Initial guess
        is_minimization = true,               # Minimization problem
        objective = objective_fn,             # Objective function
        gradient = gradient_fn,               # Gradient of the objective
        constraints = constraints_fn,         # Constraints function
        jacobian = jacobian_fn,               # Jacobian of the constraints
        eqcon = [1],                          # Equality constraint index
        ineqcon = [2],                         # No inequality constraints
        solver_settings = SEQUOIA_Settings(:IntPt, :LBFGS, false, 1e-6, 1000, 300),
    )

    # Solve the problem using QPM
    solve!(pb)

    # Display the results
    println("Solution History:")
    for step in pb.solution_history.iterates
        println("Iteration: ", step.outer_iteration_number)
        println("Solution: ", step.x)
        println("Objective Value: ", step.fval)
        println("Constraint Violation: ", step.convergence_metric)
    end
    =#
    using LinearAlgebra

    using LinearAlgebra

    function kkt_solver(f, ∇f, g, ∇g, h, ∇h, x0, s0, λ0, μ0; tol=1e-6, max_iters=100, α=0.01, β=0.5)
        """
        Solve the KKT system for a nonlinear optimization problem with equality and inequality constraints.
    
        Arguments:
        - f: Objective function, f(x).
        - ∇f: Gradient of the objective function, ∇f(x).
        - g: Inequality constraint function, g(x).
        - ∇g: Gradient of the inequality constraint function, ∇g(x).
        - h: Equality constraint function, h(x).
        - ∇h: Gradient of the equality constraint function, ∇h(x).
        - x0: Initial guess for the decision variables.
        - s0: Initial guess for the slack variables.
        - λ0: Initial guess for the Lagrange multipliers of inequality constraints.
        - μ0: Initial guess for the Lagrange multipliers of equality constraints.
        
        Keyword Arguments:
        - tol: Convergence tolerance.
        - max_iters: Maximum number of iterations.
        - α: Step size scaling factor for backtracking line search.
        - β: Reduction factor for backtracking line search.
    
        Returns:
        - x, s, λ, μ: The optimized variables.
        """
    
        # Initial values
        x, s, λ, μ = x0, s0, λ0, μ0
        n, m, p = length(x), length(s), length(μ)
        
        for iter in 1:max_iters
            # Evaluate functions and their gradients
            fx = f(x)
            ∇fx = ∇f(x)
            gx = g(x)
            hx = h(x)
            
            # Compute gradient matrices
            ∇gx = ∇g(x)   # Gradient of inequality constraints
            ∇hx = ∇h(x)   # Gradient of equality constraints
            
            # Residuals for KKT conditions
            # Stationarity residual
            r_stationarity = ∇fx + ∇gx * λ + ∇hx * μ
            
            # Complementary slackness residual (element-wise multiplication)
            r_complementary = λ .* s
    
            # Primal feasibility residuals
            r_primal_feasibility = [gxi + si^2 for (gxi, si) in zip(gx, s)]
            
            # Equality constraint residuals
            r_equality = hx
    
            # Concatenate residuals into a single vector for the RHS
            rhs = -vcat(r_stationarity, r_primal_feasibility, r_complementary, r_equality)
    
            # Construct KKT matrix
            # Upper-left block: Hessian (Lagrangian) w.r.t. x, currently only using ∇²f(x) here.
            H = I * 0.0   # Example Hessian placeholder, modify if second derivatives are used
            for i in 1:m
                H += λ[i] * ∇gx[:, i] * ∇gx[:, i]'
            end

            # Upper-right blocks (∇g and ∇h for constraints)
            Jg = hcat([∇gx[:, i] for i in 1:m]...)  # Jacobian of g (inequality)
            Jh = hcat([∇hx[:, j] for j in 1:p]...)  # Jacobian of h (equality)
    
            # Construct diagonal matrices S and Λ for slack variables and Lagrange multipliers
            S = Diagonal(2 * s)       # Diagonal matrix for 2 * s
            Λ = Diagonal(λ)           # Diagonal matrix for λ
    
            # Form the full KKT matrix based on the structured blocks
            KKT_matrix = [
                H         Jg        Jh      zeros(n, m);
                Jg'       S         zeros(m, p) Λ;
                zeros(m, n) Λ      zeros(m, m) S;
                Jh'       zeros(p, m) zeros(p, m) zeros(p, p)
            ]
    
            # Solve for the Newton step
            Δ = KKT_matrix \ rhs
    
            # Extract direction for each variable
            Δx = Δ[1:n]
            Δs = Δ[n+1:n+m]
            Δλ = Δ[n+m+1:n+m+m]
            Δμ = Δ[end-p+1:end]
    
            # Backtracking line search
            t = 1.0
            while minimum(s + t * Δs) <= 0 || minimum(λ + t * Δλ) <= 0
                t *= β
            end
            while norm([g(x + t * Δx) + (s + t * Δs).^2; h(x + t * Δx)]) > (1 - α * t) * norm(rhs)
                t *= β
            end
    
            # Update variables
            x += t * Δx
            s += t * Δs
            λ += t * Δλ
            μ += t * Δμ
            
            # Convergence check based on the residual norm
            r_norm = norm(rhs)
            if r_norm < tol
                println("Converged in $iter iterations with residual norm $r_norm")
                return x, s, λ, μ
            end
        end
        
        error("Max iterations reached without convergence.")
    end
    
# Define the functions
f(x) = (x[1] - 1)^2 + (x[2] - 2)^2
∇f(x) = [2 * (x[1] - 1); 2 * (x[2] - 2)]

g(x) = [x[1]^2 + x[2]^2 - 1]
∇g(x) = [2 * x[1] 2 * x[2]]'   # Gradient of g1

h(x) = [x[1] + x[2] - 1]
∇h(x) = [1 1]'                 # Gradient of h1

# Initial guesses
x0 = [0.5, 0.5]
s0 = [0.5]
λ0 = [1.0]
μ0 = [1.0]

# Call the KKT solver
x_opt, s_opt, λ_opt, μ_opt = kkt_solver(f, ∇f, g, ∇g, h, ∇h, x0, s0, λ0, μ0)

println("Optimal x: ", x_opt)
println("Optimal s: ", s_opt)
println("Optimal λ: ", λ_opt)
println("Optimal μ: ", μ_opt)

