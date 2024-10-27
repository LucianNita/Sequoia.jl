
using Sequoia,CUTEst,Optim,NLPModels,LinearAlgebra

problem=CUTEstModel("BT1");
sol_hist=sequoia_feasibility_cutest(problem,Optim.LBFGS())
finalize(problem)