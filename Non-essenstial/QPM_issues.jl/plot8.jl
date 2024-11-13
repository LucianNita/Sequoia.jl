using LinearAlgebra
using Optim
using Statistics  # For computing the mean
using Plots

### Problem Definition with Non-Convex Cubic Objective

# Define the objective function
function objective(x::Vector)
    return x[1]^3 + x[2]^3
end

# Equality constraint: x + y = 0
function equality_constraint(x::Vector)
    return x[1] + x[2]
end

# Struct to store the problem functions
struct OptimizationProblem
    objective::Function
    equality_constraint::Function
end

# Function to create the penalized objective for a given penalty parameter
function penalized_objective(problem::OptimizationProblem, penalty_param::Float64)
    return x -> problem.objective(x) + penalty_param * problem.equality_constraint(x)^2
end

# Define the problem
problem = OptimizationProblem(objective, equality_constraint)

### Experiment: Solving for Multiple Penalty Parameters and Measuring Performance

# Penalty parameters
penalty_params = [0.1, 1, 10, 100, 1000]
num_runs = 5

# Arrays to store average results
average_computation_times = []
average_constraint_violations = []
average_objective_values = []

# Solve the problem for each penalty parameter
for penalty_param in penalty_params
    # Track computation times, constraint violations, and objective values for averaging
    times_for_param = []
    violations_for_param = []
    objective_values_for_param = []
    
    for _ in 1:num_runs
        # Define the penalized objective function with current penalty
        penalized_func = penalized_objective(problem, penalty_param)
        
        # Solve using the quadratic penalty method starting from a small initial guess
        result = optimize(penalized_func, [0.01, 0.01], BFGS())
        optimal_x = Optim.minimizer(result)
        
        # Measure computation time
        push!(times_for_param, result.time_run)
        
        # Calculate total constraint violation
        equality_violation = abs(equality_constraint(optimal_x))
        push!(violations_for_param, equality_violation)
        
        # Calculate objective function value at optimal_x
        push!(objective_values_for_param, objective(optimal_x))
    end
    
    # Store average values for this penalty parameter
    push!(average_computation_times, mean(times_for_param))
    push!(average_constraint_violations, mean(violations_for_param))
    push!(average_objective_values, mean(objective_values_for_param))
end

### Plotting Results

# Plot Average Constraint Violation and Objective Function Value vs Penalty Parameter
p = plot(penalty_params, average_constraint_violations, 
         xlabel="Penalty Parameter (log scale)", 
         ylabel="Average Constraint Violation (log scale)", 
         title="Constraint Violation and Objective Value vs Penalty Parameter", 
         xscale=:log10, 
         yscale=:log10, 
         lw=2, marker=:o, label="Constraint Violation")

# Plot Objective Function Value on the same plot with a secondary y-axis
plot!(p, penalty_params, average_objective_values, 
      ylabel="Objective Function Value (log scale)", 
      yaxis=:right, 
      xscale=:log10, 
      yscale=:log10, 
      lw=2, marker=:s, label="Objective Value")

# Display the plot
display(p)
