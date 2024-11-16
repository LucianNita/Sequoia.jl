using Sequoia,CUTEst,Optim,NLPModels,LinearAlgebra

using SolverBenchmark, SolverCore # benchmarking tools
using DataFrames, JLD2 # to store and save results

TOL  = 1e-4 # tolerance 1e-4, 1e-6
NLOG = 0 # size 0, 1, 2
NMIN = (NLOG > 0 ? 10^NLOG + 1 : 0)
NMAX = 10^(NLOG + 1)
MAX_TIME = 5.0 # seconds
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
    max_time::Float64 = 5.0, # max time (s)
)   

    sequoia_problem = cutest_to_sequoia(sqm)
    set_solver_settings!(sequoia_problem, SEQUOIA_Settings(:SEQUOIA,:LBFGS,true,tol,1000,max_time,10^-6))
    sequoia_problem.solver_settings.cost_tolerance=10^-6
    sequoia_problem.solver_settings.cost_min=-10^12
    start_time = time()
    Sequoia.solve!(sequoia_problem);

    el_time = time() - start_time

    return GenericExecutionStats(
        sqm,
        status = sequoia_problem.solution_history.iterates[end].solver_status,
        solution = sequoia_problem.solution_history.iterates[end].x,
        iter = sequoia_problem.solution_history.iterates[end].outer_iteration_number,
        primal_feas = sequoia_problem.solution_history.iterates[end].cval[1],
        objective = sequoia_problem.solution_history.iterates[end].fval,
        elapsed_time = el_time
    )

end

seq_solver =
    nlp -> sequoia(
        nlp,
        tol = TOL,
        max_time = MAX_TIME)


sequoia_stats5 = solve_problems(seq_solver, :sequoia, problems)
@save "sequoia1_" * attrname * ".jld2" sequoia_stats5

