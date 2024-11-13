using LinearAlgebra
using Optim
using Plots

# Define the objective function
function objective(x::Vector)
    return (x[1] - 2)^4 + (x[1] - 2 * x[2])^2
end

# Define the inequality constraint: (x - 1)^2 + (y - 1)^2 <= 0.25
function inequality_constraint(x::Vector)
    return (x[1] - 1)^2 + (x[2] - 1)^2 - 0.25
end

# Define the equality constraint: x + y - 3 = 0
function equality_constraint(x::Vector)
    return x[1] + x[2] - 3
end

# Define the quadratic penalized objective function with both constraints
function penalized_objective(x::Vector, penalty_param)
    # Original objective + penalty for equality constraint + penalty for inequality constraint
    return objective(x) + penalty_param * equality_constraint(x)^2 +
           penalty_param * max(inequality_constraint(x), 0)^2
end

# Penalty parameters for experiments
penalty_params = [0.1, 1, 10, 100, 1000]  # Adjust these as needed
num_runs = 5  # Averaging multiple runs for stable results

# Arrays to store average computation time and condition numbers
average_computation_times = []
average_constraint_violations = []

# Solve the penalized problem for each penalty parameter
for penalty_param in penalty_params
    # Store the computation times and constraint violations for averaging
    times_for_param = []
    constraint_violations_for_param = []
    
    for _ in 1:num_runs
        # Define penalized function with the current penalty parameter
        penalized_func(x) = penalized_objective(x, penalty_param)
        
        # Solve the unconstrained optimization problem
        result = optimize(penalized_func, [0.0, 0.0], BFGS())
        
        # Extract computation time
        time_taken = result.time_run
        push!(times_for_param, time_taken)
        
        # Calculate and store constraint violation for equality and inequality constraints
        optimal_x = Optim.minimizer(result)
        equality_violation = abs(equality_constraint(optimal_x))
        inequality_violation = max(inequality_constraint(optimal_x), 0)
        total_violation = equality_violation + inequality_violation
        push!(constraint_violations_for_param, total_violation)
    end
    
    # Store average time and constraint violation
    avg_time = mean(times_for_param)
    avg_violation = mean(constraint_violations_for_param)
    push!(average_computation_times, avg_time)
    push!(average_constraint_violations, avg_violation)
end

# Plot Average Computation Time vs Penalty Parameter
plot(penalty_params, average_computation_times, 
     xlabel="Penalty Parameter (log scale)", 
     ylabel="Average Computation Time (seconds, log scale)", 
     title="Average Computation Time vs Penalty Parameter", 
     xscale=:log10, 
     yscale=:log10, 
     lw=2, marker=:o)

#= Plot Average Constraint Violation vs Penalty Parameter
plot(penalty_params, average_constraint_violations, 
     xlabel="Penalty Parameter (log scale)", 
     ylabel="Average Constraint Violation (log scale)", 
     title="Average Constraint Violation vs Penalty Parameter", 
     xscale=:log10, 
     yscale=:log10, 
     lw=2, marker=:o)
=#