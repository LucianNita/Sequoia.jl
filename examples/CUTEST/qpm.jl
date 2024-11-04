using Sequoia,CUTEst,Optim,NLPModels,LinearAlgebra

using SolverBenchmark, SolverCore # benchmarking tools
using DataFrames, JLD2 # to store and save results

TOL  = 1e-4 # tolerance 1e-4, 1e-6
NLOG = 0 # size 0, 1, 2
NMIN = (NLOG > 0 ? 10^NLOG + 1 : 0)
NMAX = 10^(NLOG + 1)
MAX_TIME = 60.0 # seconds
MAX_ITER = 1000000000;

attrname = "cons_O$(string(NLOG))"
if TOL == 1e-4
    attrname = attrname * "_tol4"
elseif TOL == 1e-6
    attrname = attrname * "_tol6"
else
    @error "check filename"
end

cutest_problems =
    CUTEst.select(min_var = NMIN, min_con = 1, max_var = NMAX, max_con = NMAX)
problems = (CUTEstModel(p) for p in cutest_problems)
length(problems) # number of problems

function sequoia(
    sqm::CUTEstModel; # Problem to be solved by this method

    # termination
    tol::Real = eltype(x0)(1e-4), # tolerance
    max_time::Float64 = 60.0, # max time (s)
)   

    sequoia_problem = cutest_to_sequoia(sqm)
    set_solver_settings!(sequoia_problem, SEQUOIA_Settings(:QPM_lite,:LBFGS,false,tol,1000,max_time,10^-6))
    
    start_time = time()
    Sequoia.solve!(sequoia_problem);

    el_time = time() - start_time

    sol_hist = sequoia_problem.solution_history;

    return GenericExecutionStats(
        sqm,
        status = sol_hist.iterates[end].solver_status,
        solution = sol_hist.iterates[end].x,
        iter = sol_hist.iterates[end].outer_iteration_number,
        primal_feas = sol_hist.iterates[end].cval[1],
        objective = sol_hist.iterates[end].fval,
        elapsed_time = el_time
    )

end

seq_solver =
    nlp -> sequoia(
        nlp,
        tol = TOL,
        max_time = MAX_TIME)


sequoia_stats = solve_problems(seq_solver, :sequoia, problems)
@save "qpm_" * attrname * ".jld2" sequoia_stats

