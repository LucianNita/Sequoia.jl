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

# KKT System Solver with equality constraints and quadratic slack variables
function solve_kkt_quadratic_slack(x, s, λ, ν, μ)
    # Assemble the KKT system
    n = length(x)
    m = length(s)
    p = length(h(x))  # Number of equality constraints

    # Gradient of Lagrangian (Stationarity)
    Lx = ∇f(x) + ∇g(x)' * λ + ∇h(x)' * ν

    # Primal feasibility: g_i(x) + s_i^2 = 0 and h_j(x) = 0
    primal_feas_g = g(x) + s.^2
    primal_feas_h = h(x)

    # Build the system of equations (KKT system)
    KKT_matrix = [zeros(n, n) ∇g(x)' ∇h(x)' zeros(n, m);
                  ∇g(x) zeros(m, m) zeros(m, p) diagm(2 .* s);
                  ∇h(x) zeros(p, m) zeros(p, p) zeros(p, m);
                  zeros(m, n) diagm(2 .* s) zeros(m, p) I(m)]

    KKT_rhs = [-Lx;
               -primal_feas_g;
               -primal_feas_h;
               zeros(m)]

    # Solve the KKT system for search directions Δx, Δλ, Δν, Δs
    Δ = KKT_matrix \ KKT_rhs

    Δx = Δ[1:n]
    Δλ = Δ[n+1:n+m]
    Δν = Δ[n+m+1:n+m+p]
    Δs = Δ[n+m+p+1:end]

    return Δx, Δλ, Δν, Δs
end

# Line search function
function line_search(x, s, λ, ν, Δx, Δλ, Δν, Δs)
    t = 1.0

    # Backtracking line search to ensure objective decreases
    while f(x + t * Δx) > f(x) + α * t * dot(∇f(x), Δx)
        t *= β
    end

    return t
end

# Interior Point Method with Quadratic Slack and Equality Constraints
function interior_point_quadratic_slack_method(x0, s0, λ0, ν0, μ_initial)
    x = x0
    s = s0
    λ = λ0
    ν = ν0
    μ_param = μ_initial

    # Iterate until convergence
    while μ_param > tolerance
        # Solve the KKT system using Newton's method with quadratic slack
        Δx, Δλ, Δν, Δs = solve_kkt_quadratic_slack(x, s, λ, ν, μ_param)

        # Line search to find the step size
        t = line_search(x, s, λ, ν, Δx, Δλ, Δν, Δs)

        # Update primal and dual variables
        x += t * Δx
        λ += t * Δλ
        ν += t * Δν
        s += t * Δs

        # Update the barrier parameter
        μ_param *= 0.9  # Reduce the barrier parameter
    end

    return x, s, λ, ν
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
