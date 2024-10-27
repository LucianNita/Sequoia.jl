using CUTEst, Sequoia, Optim

problem=CUTEstModel("HS21");#BT1

sequoia_problem = cutest_to_sequoia(problem)
#sol_hist=qpm_solve_cutest(problem,Optim.LBFGS(),Optim.Options(g_tol=1e-6, iterations=100, store_trace=true, extended_trace=true, show_trace=false))
sol_hist = solve!(sequoia_problem);
println("Solution history: ", sol_hist)# Access the solution history

finalize(problem)