using Optim
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

# Solve the quadratic penalty problem for a series of penalty parameters
penalty_params = [0.1, 1, 10, 100, 1000, 10000, 100000]  # Adjust these as needed
solutions = []
constraint_violations = []
objective_values = []

for penalty_param in penalty_params
    # Define the penalized objective function for the current penalty parameter
    penalized_func(x) = penalized_objective(x, penalty_param)
    
    # Solve the unconstrained optimization problem
    result = optimize(penalized_func, [0.0, 0.0], Newton())
    optimal_x = Optim.minimizer(result)
    
    # Store the solution, objective value, and constraint violation
    push!(solutions, optimal_x)
    push!(objective_values, objective(optimal_x))
    push!(constraint_violations, abs(constraint(optimal_x)))
end

# Define tick positions and labels
x_ticks = [10. ^i for i in -1:4]         # Adjusted based on the penalty parameter range
y_ticks = [10. ^i for i in -5:2]         # Expanded to include smaller constraint violation values

# Helper function to create formatted tick labels using LaTeX notation
x_labels = ["10^{$(i)}" for i in -1:4]  # X-axis ticks with formatted labels
y_labels = ["10^{$(i)}" for i in -5:2]  # Y-axis ticks with formatted labels

# Plot with customized ticks, grid lines, and improved axis labels
plot(penalty_params, constraint_violations, 
     xlabel="Penalty Parameter (log scale)", 
     ylabel="Constraint Violation (log scale)", 
     title="Effect of Penalty Parameter on Constraint Violation and Objective Value", 
     lw=2, marker=:o, 
     xscale=:log10, 
     yscale=:log10, 
     xticks=(x_ticks, x_labels), # Set custom x-axis ticks and labels
     yticks=(y_ticks, y_labels), # Set custom y-axis ticks and labels
     grid=:true, 
     minorgrid=true, 
     legend=:right, # Place legend on the east side of the plot
     label="Constraint Violation")

# Overlay plot for objective value vs penalty parameter
plot!(penalty_params, objective_values, 
      ylabel="Value (log scale)", 
      lw=2, marker=:o, 
      label="Objective Value")


