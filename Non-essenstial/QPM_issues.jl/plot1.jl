using Plots

# Define the objective function
function objective(x::Vector)
    return x[1]^2 + x[2]^2
end

# Define the equality constraint
function constraint(x::Vector)
    return x[1] + x[2] - 1
end

# Plotting function for visualization
function plot_problem()
    # Define grid for x and y
    x_vals = -1.5:0.1:1.5
    y_vals = -1.5:0.1:1.5
    
    # Compute objective function values on the grid
    Z = [objective([x, y]) for x in x_vals, y in y_vals]
    
    # Plot contour of the objective function
    contour(x_vals, y_vals, Z, levels=20, color=:blues, xlabel="x", ylabel="y", title="Objective Function and Constraint", label="Objective Contours")
    
    # Add the constraint line (x + y = 1)
    plot!(x_vals, 1 .- x_vals, color=:red, linewidth=2, label="x + y = 1 (constraint)")
    
    # Set plot limits for clarity
    xlims!(-1.5, 1.5)
    ylims!(-1.5, 1.5)
end

# Run the plot function to visualize the problem setup
plot_problem()
