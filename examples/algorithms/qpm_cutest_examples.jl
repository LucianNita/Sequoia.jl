using CUTEst, Sequoia, Optim

problem=CUTEstModel("STANCMIN")#HS37");#BT1

sequoia_problem = cutest_to_sequoia(problem)
set_solver_settings!(sequoia_problem, SEQUOIA_Settings(:QPM_lite,:LBFGS,false,10^-6,1000,300, 10^-6))
Sequoia.solve!(sequoia_problem);
sol_hist = sequoia_problem.solution_history.iterates
println("Solution history: ", sol_hist[end].x)# Access the solution history


#finalize(problem)