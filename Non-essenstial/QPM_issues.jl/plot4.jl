using LinearAlgebra
using Optim
using Statistics  # For computing the mean
using Plots

# Original objective function
function objective(x::Vector)
    return x[1]^2 + x[2]^2
end

# Constraint function (x + y - 1 = 0)
function constraint(x::Vector)
    return x[1] + x[2] - 1
end

# Quadratic penalty function
function penalized_objective(x::Vector, penalty_param)
    return objective(x) + penalty_param * constraint(x)^2
end

# Parameters
penalty_params = 10.0 .^[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]  # Adjust as needed
num_runs = 10  # Number of runs to average for each penalty parameter

# Array to store average computation times
average_computation_times = []

for penalty_param in penalty_params
    # Store the computation times for each run
    times_for_param = []
    
    for _ in 1:num_runs
        # Define the penalized objective function for the current penalty parameter
        penalized_func(x) = penalized_objective(x, penalty_param)
        
        # Solve the optimization problem and extract computation time
        result = optimize(penalized_func, [0.0, 0.0], BFGS())
        time_taken = result.time_run  # Extract the time directly from the result
        
        # Append the time for this run
        push!(times_for_param, time_taken)
    end
    
    # Compute the average time for this penalty parameter
    avg_time = mean(times_for_param)
    push!(average_computation_times, avg_time)
end

# Plot Average Computational Time vs Penalty Parameter
plot(penalty_params, average_computation_times, 
     xlabel="Penalty Parameter (log scale)", 
     ylabel="Average Computation Time (seconds, log scale)", 
     title="Average Computation Time vs Penalty Parameter", 
     xscale=:log10, 
     yscale=:log10, 
     lw=2, marker=:o)
