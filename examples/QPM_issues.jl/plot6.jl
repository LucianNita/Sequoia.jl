using LinearAlgebra
using Optim
using Statistics  # For computing the mean
using Plots

### PART 1: Problem Definition

# Define a problem struct to store objective and constraint functions
struct OptimizationProblem
    objective::Function
    equality_constraint::Function
    inequality_constraint::Function
end

# Example problem setup (non-convex, tight constraints)
function example_problem()
    # Objective function
    objective(x::Vector) = (x[1] - 2)^4 + (x[1] - 2 * x[2])^2
    
    # Equality constraint: x + y - 3 = 0
    equality_constraint(x::Vector) = x[1] + x[2] - 3
    
    # Inequality constraint: (x - 1)^2 + (y - 1)^2 <= 0.25
    inequality_constraint(x::Vector) = (x[1] - 1)^2 + (x[2] - 1)^2 - 0.25
    
    return OptimizationProblem(objective, equality_constraint, inequality_constraint)
end

# Function to create the penalized objective for a given penalty parameter
function penalized_objective(problem::OptimizationProblem, penalty_param::Float64)
    return x -> problem.objective(x) + 
                penalty_param * problem.equality_constraint(x)^2 + 
                penalty_param * max(problem.inequality_constraint(x), 0)^2
end

### PART 2: Experiment Functions

# Function to measure computation time for each penalty parameter
function compute_average_time(problem::OptimizationProblem, penalty_params::Vector{Float64}, num_runs::Int)
    average_times = []
    
    for penalty_param in penalty_params
        times = []
        for _ in 1:num_runs
            penalized_func = penalized_objective(problem, penalty_param)
            result = optimize(penalized_func, [0.0, 0.0], BFGS())
            push!(times, result.time_run)
        end
        push!(average_times, mean(times))
    end
    return average_times
end

# Function to measure constraint violations for each penalty parameter
function compute_average_violations(problem::OptimizationProblem, penalty_params::Vector{Float64}, num_runs::Int)
    average_violations = []
    
    for penalty_param in penalty_params
        violations = []
        for _ in 1:num_runs
            penalized_func = penalized_objective(problem, penalty_param)
            result = optimize(penalized_func, [0.0, 0.0], BFGS())
            optimal_x = Optim.minimizer(result)
            
            # Calculate total constraint violation
            equality_violation = abs(problem.equality_constraint(optimal_x))
            inequality_violation = max(problem.inequality_constraint(optimal_x), 0)
            total_violation = equality_violation + inequality_violation
            push!(violations, total_violation)
        end
        push!(average_violations, mean(violations))
    end
    return average_violations
end

# Function to compute condition number for each penalty parameter
function compute_condition_numbers(problem::OptimizationProblem, penalty_params::Vector{Float64})
    condition_numbers = []
    
    for penalty_param in penalty_params
        penalized_func = penalized_objective(problem, penalty_param)
        result = optimize(penalized_func, [0.0, 0.0], BFGS())
        optimal_x = Optim.minimizer(result)
        
        # Compute Hessian and condition number
        H_obj = [2.0 0.0; 0.0 2.0]  # For illustrative purposes
        grad_constr = [1.0, 1.0]
        H_penalty = 2 * penalty_param * (grad_constr * grad_constr')
        H = H_obj + H_penalty
        push!(condition_numbers, cond(H))
    end
    return condition_numbers
end

### PART 3: Plotting Functions

# Function to plot computation time vs penalty parameter
function plot_computation_time(penalty_params, average_times)
    plot(penalty_params, average_times,
         xlabel="Penalty Parameter (log scale)", 
         ylabel="Average Computation Time (seconds, log scale)", 
         title="Average Computation Time vs Penalty Parameter", 
         xscale=:log10, 
         yscale=:log10, 
         lw=2, marker=:o)
end

# Function to plot constraint violation vs penalty parameter
function plot_constraint_violation(penalty_params, average_violations)
    plot(penalty_params, average_violations,
         xlabel="Penalty Parameter (log scale)", 
         ylabel="Average Constraint Violation (log scale)", 
         title="Average Constraint Violation vs Penalty Parameter", 
         xscale=:log10, 
         yscale=:log10, 
         lw=2, marker=:o)
end

# Function to plot condition number vs penalty parameter
function plot_condition_number(penalty_params, condition_numbers)
    plot(penalty_params, condition_numbers,
         xlabel="Penalty Parameter (log scale)", 
         ylabel="Condition Number (log scale)", 
         title="Condition Number vs Penalty Parameter", 
         xscale=:log10, 
         yscale=:log10, 
         lw=2, marker=:o)
end

### PART 4: Run Experiments

# Define problem and parameters
problem = example_problem()
penalty_params = 10.0 .^[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
num_runs = 20

# Run each experiment
average_times = compute_average_time(problem, penalty_params, num_runs)
average_violations = compute_average_violations(problem, penalty_params, num_runs)
condition_numbers = compute_condition_numbers(problem, penalty_params)

# Plot results
plot_computation_time(penalty_params, average_times)
#plot_constraint_violation(penalty_params, average_violations)
#plot_condition_number(penalty_params, condition_numbers)
