
using Sequoia,CUTEst,Optim,NLPModels,LinearAlgebra

problem=CUTEstModel("HS21");
sequoia_problem = cutest_to_sequoia(problem)
set_solver_settings!(sequoia_problem, SEQUOIA_Settings(:IntPt,:LBFGS,false,10^-6,1000,300,10^-6))
Sequoia.solve!(sequoia_problem);

#println("Solution history: ", sequoia_problem.solution_history.iterates[end].x)# Access the solution history
#finalize(problem)