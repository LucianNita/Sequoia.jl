using CUTEst
using Optim
using NLPModels
using SolverBenchmark
using SolverCore

using DataFrames, JLD2 # to store and save results

# List of algorithms to test
algorithms = [
    #(:NelderMead, NelderMead()),
    #(:BFGS, BFGS()),
    #(:LBFGS, LBFGS()),
    #(:ConjugateGradient, ConjugateGradient()),
    #(:GradientDescent, GradientDescent()),
    #(:Newton, Newton())
]

TOL  = 1e-4; # tolerance 1e-4, 1e-6, 1e-8
NLOG = 0; # size 0, 1, 2, 3
NMIN = (NLOG > 0 ? 10^NLOG + 1 : 0);
NMAX = (NLOG > 2 ? Inf : 10^(NLOG + 1));
MAX_TIME = 60.0; # seconds
MAX_ITER = Int(1e8);

# List of CUTEst problems to test
cutest_problems = CUTEst.select(max_con=0, min_var=NMIN, max_var=NMAX)
problems = (CUTEstModel(p) for p in cutest_problems)


function test_algo(nlp::AbstractNLPModel,algo)
    
    #x0 = nlp.meta.x0

    #g=zeros(nlp.meta.nvar)
    #h=zeros(nlp.meta.nvar,nlp.meta.nvar)
    
    res=optimize(x->obj(nlp,x), (x)->grad(nlp,x), (x)->hess(nlp,x), nlp.meta.x0, algo, inplace=false,Optim.Options(g_tol = TOL, iterations = MAX_ITER, store_trace = true, show_trace = false, show_warnings = true, time_limit=MAX_TIME) )
    

    if Optim.converged(res)
        status=:first_order
    elseif Optim.iteration_limit_reached(res)
        status=:max_iter
    elseif res.time_run>MAX_TIME
        status=:max_time
    elseif !res.ls_success
        status=:small_step
    else
        status=:unknown
    end

    return GenericExecutionStats(nlp, status=status, solution = res.minimizer, iter=res.iterations, objective = res.minimum, elapsed_time = res.time_run) 
end

#=
df = Vector{DataFrame}(undef,0)
for (algo_name, algo) in algorithms
    push!(df,solve_problems(mdl->test_algo(mdl,algo), algo_name, problems))
end

#a6=vcat(df[1],df[2],df[3],df[4],df[5],df[6]);

d=df[1];
@save "Unconstrained_testing" * ".jld2" d
=#