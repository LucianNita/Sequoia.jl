using LinearAlgebra
using Optim
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
function penalized_objective(problem::OptimizationProblem, penalty_param::Real)
    return x -> problem.objective(x) + penalty_param * problem.equality_constraint(x)^2
end

# Define the problem
problem = OptimizationProblem(objective, equality_constraint)

### Setting up the Contour Plot and Iterates

# Penalty parameter for visualization (e.g., 1000 for noticeable constraint enforcement)
penalty_param = 1000

# Set up grid for contour plot
x_vals = -1.0:0.05:1.0
y_vals = -1.0:0.05:1.0
Z_objective = [objective([x, y]) for x in x_vals, y in y_vals]

# Plot objective function contour
contour(x_vals, y_vals, Z_objective, levels=20, xlabel="x", ylabel="y", title="Objective and Constraint with Iterates", color=:blues)

# Overlay the constraint line x + y = 0 (y = -x)
plot!(x_vals, -x_vals, color=:red, linewidth=2, label="x + y = 0 (constraint)")

# Solve the problem using quadratic penalty with trace storage
initial_guess = [0.5, 0.5]
penalized_func = penalized_objective(problem, penalty_param)
result = optimize(penalized_func, initial_guess, GradientDescent(), Optim.Options(store_trace=true, extended_trace=true))

# Retrieve iterates from optimization history
iterates = Optim.x_trace(result)

# Plot iterates on the contour plot
x_iterates = [iter[1] for iter in iterates]
y_iterates = [iter[2] for iter in iterates]
plot!(x_iterates, y_iterates, marker=:o, markersize=4, label="Iterates", color=:green, linewidth=2)

# Display the final plot
display(plot!)
