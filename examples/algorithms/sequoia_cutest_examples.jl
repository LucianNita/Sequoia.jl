
using Sequoia,CUTEst,Optim,NLPModels,LinearAlgebra

problem=CUTEstModel("HS21");#S365")#
sequoia_problem = cutest_to_sequoia(problem)
set_solver_settings!(sequoia_problem, SEQUOIA_Settings(:SEQUOIA,:LBFGS,false,10^-6,1000,300))
#sequoia_problem.x0=ones(7)
sol_hist = Sequoia.solve!(sequoia_problem);
println("Solution history: ", sol_hist.iterates[end].x)# Access the solution history
#finalize(problem)