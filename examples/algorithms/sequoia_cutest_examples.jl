
using Sequoia,CUTEst,Optim,NLPModels,LinearAlgebra

problem=CUTEstModel("HS21");
sol_hist=sequoia_solve_cutest(problem,Optim.LBFGS())
finalize(problem)