using CUTEst, NLPModels
using LinearAlgebra
using SparseArrays

# Function to extract equality and inequality constraints
function extract_constraints(prob)
    cl = prob.meta.lcon
    cu = prob.meta.ucon
    ncon = prob.meta.ncon

    eq_idx = findall(cl .== cu) # Equality constraints where c_L == c_U
    ineq_idx = findall(cl .!= cu) # Inequality constraints where c_L != c_U

    return eq_idx, ineq_idx
end

# Function to compute the equality constraints residuals
function ce(x, prob, eq_idx)
    c = cons(prob,x)
    return c[eq_idx] .- prob.meta.cl[eq_idx]
end

# Function to compute the inequality constraints residuals
function ci(x, prob, ineq_idx)
    c = cons(prob,x)
    ci_list = []

    for idx in ineq_idx
        cl = prob.meta.lcon[idx]
        cu = prob.meta.ucon[idx]

        if isfinite(cu) && isfinite(cl) && cl != cu
            # Transform c_L ≤ c(x) ≤ c_U into two inequalities
            push!(ci_list, c[idx] - cu)   # c(x) - c_U ≤ 0
            push!(ci_list, cl - c[idx])   # c_L - c(x) ≤ 0
        elseif isfinite(cu) && !isfinite(cl)
            push!(ci_list, c[idx] - cu)   # c(x) - c_U ≤ 0
        elseif !isfinite(cu) && isfinite(cl)
            push!(ci_list, cl - c[idx])   # c_L - c(x) ≤ 0
        else
            # Ignore constraints with infinite bounds
            println("Constraint $idx has infinite bounds and is ignored.")
        end
    end
    return vcat(ci_list)
end

# Function to compute the KKT residuals
function KKT_residuals(z, prob, eq_idx, ineq_idx, μ_barrier)
    n = prob.meta.nvar
    m = length(ineq_idx) * 2  # Adjusted for splitting inequalities
    p = length(eq_idx)

    x = z[1:n]
    s = z[n+1:n+m]
    λ = z[n+m+1:n+m+p]
    μ = z[n+m+p+1:n+m+p+m]
    ν = z[n+m+p+m+1:end]

    # Compute gradients and Jacobians
    grad_f = grad(prob,x)
    c = cons(prob,x)
    J = jac(prob,x)

    # Equality constraints
    ce_res = c[eq_idx] .- prob.meta.lcon[eq_idx]
    Jce = J[eq_idx, :]

    # Inequality constraints
    ci_res = ci(x, prob, ineq_idx)
    Jg_list = []

    for idx in ineq_idx
        cl = prob.meta.lcon[idx]
        cu = prob.meta.ucon[idx]
        Jc = J[idx, :]

        if isfinite(cu) && isfinite(cl) && cl != cu
            # c_L ≤ c(x) ≤ c_U
            push!(Jg_list, Jc)   # For c(x) - c_U ≤ 0
            push!(Jg_list, -Jc)  # For c_L - c(x) ≤ 0
        elseif isfinite(cu) && !isfinite(cl)
            push!(Jg_list, Jc)   # For c(x) - c_U ≤ 0
        elseif !isfinite(cu) && isfinite(cl)
            push!(Jg_list, -Jc)  # For c_L - c(x) ≤ 0
        end
    end
    Jg = vcat(Jg_list)
    display(size(Jg))
    # Residuals
    res1 = grad_f - Jce' * λ - Jg' * μ  # Stationarity
    res2 = -2 * s .* μ + ν              # Complementarity
    res3 = ce_res                       # Equality constraints
    res4 = ci_res + s.^2                # Inequality constraints with slack
    res5 = s .* ν - μ_barrier * ones(m) # Barrier term for slacks

    # Concatenate all residuals
    return vcat(res1, res2, res3, res4, res5)
end

# Function to solve the KKT system using Newton's method
function solve_KKT(prob_name; tol=1e-6, max_iter=50, μ_barrier=1e-2)
    # Load problem
    prob = CUTEstModel(prob_name)
    n = prob.meta.nvar

    # Extract constraints
    eq_idx, ineq_idx = extract_constraints(prob)
    p = length(eq_idx)
    m = length(ineq_idx) * 2  # Adjusted for splitting inequalities

    # Initial guess
    x0 = prob.meta.x0
    s0 = ones(m) * 0.1        # Initial slacks
    λ0 = zeros(p)
    μ0 = ones(m)
    ν0 = ones(m)

    # Concatenate all variables
    z = vcat(x0, s0, λ0, μ0, ν0)

    # Newton's method
    for iter = 1:max_iter
        # Compute residuals
        res = KKT_residuals(z, prob, eq_idx, ineq_idx, μ_barrier)
        norm_res = norm(res)

        println("Iteration $iter, Residual Norm: $norm_res")

        if norm_res < tol
            println("Converged!")
            break
        end

        # Compute Jacobian (system matrix)
        nvar = n + m + p + m + m
        J_sys = zeros(nvar, nvar)

        # Extract variables
        x = z[1:n]
        s = z[n+1:n+m]
        λ = z[n+m+1:n+m+p]
        μ = z[n+m+p+1:n+m+p+m]
        ν = z[n+m+p+m+1:end]

        # Gradients and Jacobians
        grad_f = prob.grad(x)
        J = prob.jac(x)

        # Equality constraints
        Jce = J[eq_idx, :]

        # Inequality constraints
        Jg_list = []
        for idx in ineq_idx
            cl = prob.meta.lcon[idx]
            cu = prob.meta.ucon[idx]
            Jc = J[idx, :]

            if isfinite(cu) && isfinite(cl) && cl != cu
                push!(Jg_list, Jc)   # For c(x) - c_U ≤ 0
                push!(Jg_list, -Jc)  # For c_L - c(x) ≤ 0
            elseif isfinite(cu) && !isfinite(cl)
                push!(Jg_list, Jc)   # For c(x) - c_U ≤ 0
            elseif !isfinite(cu) && isfinite(cl)
                push!(Jg_list, -Jc)  # For c_L - c(x) ≤ 0
            end
        end
        Jg = vcat(Jg_list)

        # Fill the Jacobian matrix
        # Stationarity w.r.t x
        J_sys[1:n, 1:n] = prob.hess(x)  # Hessian of Lagrangian (assuming exact Hessian)
        J_sys[1:n, n+m+1:n+m+p] = -Jce'
        J_sys[1:n, n+m+p+1:n+m+p+m] = -Jg'

        # Complementarity w.r.t s
        J_sys[n+1:n+m, n+1:n+m] = -2 * Diagonal(μ)
        J_sys[n+1:n+m, n+m+p+1:n+m+p+m] += -2 * Diagonal(s)
        J_sys[n+1:n+m, n+m+p+m+1:end] = Diagonal(ones(m))

        # Equality constraints w.r.t x
        J_sys[n+m+1:n+m+p, 1:n] = Jce

        # Inequality constraints with slack
        J_sys[n+m+p+1:n+m+p+m, 1:n] = Jg
        J_sys[n+m+p+1:n+m+p+m, n+1:n+m] = Diagonal(2s)

        # Barrier term for slacks
        J_sys[n+m+p+m+1:end, n+1:n+m] = Diagonal(ν)
        J_sys[n+m+p+m+1:end, n+m+p+m+1:end] = Diagonal(s)

        # Solve for Newton step
        dz = -J_sys \ res

        # Line search (simple backtracking)
        alpha = 1.0
        while true
            z_new = z + alpha * dz
            # Ensure s > 0 and ν > 0
            if minimum(z_new[n+1:n+m]) > 0 && minimum(z_new[n+m+p+m+1:end]) > 0
                break
            else
                alpha *= 0.5
                if alpha < 1e-8
                    error("Line search failed.")
                end
            end
        end

        # Update variables
        z = z_new
    end

    # Extract solution
    x_sol = z[1:n]
    println("Optimal solution x: ", x_sol)
    println("Optimal objective value: ", prob.obj(x_sol))

    finalize(prob)

    return x_sol
end

# Example usage with a specific problem from CUTEst
solve_KKT("HS118")
