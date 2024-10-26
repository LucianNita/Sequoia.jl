using CUTEst, Sequoia

# Load a CUTEst problem, e.g., HS25
problem = CUTEstModel("HS21")

sequoia_problem = cutest_to_sequoia(problem)

solution_history = solve!(sequoia_problem);

# Access the solution history
println("Solution history: ", solution_history)

finalize(problem)
