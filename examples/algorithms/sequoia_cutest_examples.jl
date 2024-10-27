
using Sequoia,CUTEst,Optim,NLPModels,LinearAlgebra

problem=CUTEstModel("HS21");
sequoia_problem = cutest_to_sequoia(problem)
set_solver_settings!(sequoia_problem, SEQUOIA_Settings(:SEQUOIA,:LBFGS,false,10^-6,1000,300))
sol_hist = solve!(sequoia_problem);
#sol_hist=sequoia_solve_cutest(problem,Optim.LBFGS())
println("Solution history: ", sol_hist.iterates[end].x)# Access the solution history
finalize(problem)