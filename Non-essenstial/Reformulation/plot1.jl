using LinearAlgebra
using SparseArrays

using NLPModels, CUTEst, ADNLPModels
using SolverCore
using NLPModelsIpopt, Percival

export SequoiaModel2

export obj,
    grad!,
    hess_coord!,
    hess_structure!,
    hprod!,
    cons!,
    jac_coord!,
    jac_structure!,
    jprod!, jprod_nln!
    jtprod!
export objgrad!, objgrad, grad, cons, jac, jprod, jtprod, jac_coord, hess, hess_coord, hprod

import NLPModels: increment!

function validate_vector(v::Vector{Float64})
    # Check if there are any NaN values
    if any(isnan.(v))
        throw(ArgumentError("Vector contains NaN values"))
    end
    # Check if there are any Inf values
    if any(isinf.(v))
        throw(ArgumentError("Vector contains Inf values"))
    end
end


"""
SequoiaModel object.

Given the `nlp` problem

min_{x}     f(x)								      
subject to  lvar <= x    <= uvar, 
            lcon <= c(x) <= ucon,

and some t, the associated SequoiaModel is the problem

min_{x,t}    t
subject to   lvar ≤ x    ≤ uvar,
             lcon ≤ c(x) ≤ ucon,
                    f(x) ≤ t.
"""
mutable struct SequoiaModel2{T,S,M<:AbstractNLPModel{T,S}} <: AbstractNLPModel{T,S}
    nlp::M # original problem
    nx::Integer # nlp's nvar
    ny::Integer # nlp's ncon

    meta::NLPModelMeta{T,S}
    counters::Counters

    t::AbstractFloat # cost bound parameter
end


"""
Constructor for SequoiaModel
"""
function SequoiaModel2(
    nlp::AbstractNLPModel;
    x = nlp.meta.x0,
    y = nlp.meta.y0,
    t::AbstractFloat = obj(nlp,x),
)
    nx = nlp.meta.nvar
    ny = nlp.meta.ncon
    if length(x) != nx
        error("wrong length of passed argument x")
    end
    nvar = nx+1;
    meta = NLPModelMeta(
        nvar;
        lvar = vcat(nlp.meta.lvar, -Inf),
        uvar = vcat(nlp.meta.uvar, Inf),
        x0 = vcat(x, t),
        y0 = vcat(y, 1.0),
        name = nlp.meta.name * "-Sequoia2",
        nnzj = nlp.meta.nnzj + nx + 1,
        nnzh = nlp.meta.nnzh,
        ncon = nlp.meta.ncon + 1,
        lcon = vcat(nlp.meta.lcon, -Inf),
        ucon = vcat(nlp.meta.ucon, 0),
    )

    return SequoiaModel2(nlp, nx, ny, meta, Counters(), t)
end

"""
Methods for SequoiaModel
"""
# objective function
# f = obj(nlp, x)
function NLPModels.obj(sqm::SequoiaModel2, xl::AbstractVector{<:Float64})
    increment!(sqm, :neval_obj)
    if length(xl) != sqm.meta.nvar
        error("wrong length of argument xl")
    end
    
    #display("ATTEMPT obj")
    x = xl[1:sqm.nx]
    t = xl[sqm.nx+1]
    #display("END obj")
    #validate_vector(xl)
    return t
end

# objective gradient
# g = grad!(nlp, x, g)
function NLPModels.grad!(sqm::SequoiaModel2, xl::AbstractVector{<:Float64}, g::AbstractVector{<:Float64})
    increment!(sqm, :neval_grad)
    if length(xl) != sqm.meta.nvar
        error("wrong length of argument xl")
    end
    if length(g) != sqm.meta.nvar
        error("wrong length of argument g")
    end
    #display("ATTEMPT grad")
    x = Vector(xl[1:sqm.nx])  # Convert SubArray to Vector
    t = xl[sqm.nx+1]
    dfx = similar(x)
    grad!(sqm.nlp, x, dfx)
    g[1:sqm.nx] .= 0.0#.*dfx
    g[sqm.nx+1] = 1.0;
    #display("END grad")
    return g
end
#

# constraints function
# c = cons!(nlp, x, c)
function NLPModels.cons!(sqm::SequoiaModel2, xl::AbstractVector{<:Float64}, cx::AbstractVector{<:Float64})
    increment!(sqm, :neval_cons)
    if length(xl) != sqm.meta.nvar
        error("wrong length of argument xl")
    end
    if length(cx) != sqm.meta.ncon
        error("wrong length of argument cx")
    end
    #display("ATTEMPT cons")
    x = Vector(xl[1:sqm.nx])
    t = xl[sqm.nx+1]
    nlp_ncon = sqm.nlp.meta.ncon
    c_nlp = zeros(eltype(cx), nlp_ncon)
    cons!(sqm.nlp, x, c_nlp)
    cx[1:end-1] = c_nlp
    cx[end] = obj(sqm.nlp, x) - t
    #display("END cons")
    return cx
end

function NLPModels.cons_nln!(sqm::SequoiaModel2, xl::AbstractVector{<:Float64}, cx::AbstractVector{<:Float64})
    increment!(sqm, :neval_cons)
    if length(xl) != sqm.meta.nvar
        error("wrong length of argument xl")
    end
    if length(cx) != sqm.meta.ncon
        error("wrong length of argument cx")
    end
    #display("ATTEMPT cons")
    x = Vector(xl[1:sqm.nx])
    t = xl[sqm.nx+1]
    nlp_ncon = sqm.nlp.meta.ncon
    c_nlp = zeros(eltype(cx), nlp_ncon)
    cons!(sqm.nlp, x, c_nlp)
    cx[1:end-1] = c_nlp
    cx[end] = obj(sqm.nlp, x) - t
    #display("END cons")
    return cx
end
# constraints Jacobian coordinates
# vals = jac_coord!(nlp, x, vals)
function NLPModels.jac_coord!(sqm::SequoiaModel2, xl::AbstractVector{<:Float64}, vals::AbstractVector{<:Float64})
    increment!(sqm, :neval_jac)
    if length(xl) != sqm.meta.nvar
        error("wrong length of argument xl")
    end
    if length(vals) != sqm.meta.nnzj
        error("wrong length of argument vals")
    end
    #display("ATTEMPT jc")
    x = Vector(xl[1:sqm.nx])  # Convert SubArray to Vector
    t = xl[sqm.nx+1]
    nlp_nnzj = sqm.nlp.meta.nnzj
    nlp_vals = zeros(eltype(xl), nlp_nnzj)
    dfx = similar(x)
    jac_coord!(sqm.nlp, x, nlp_vals)
    grad!(sqm.nlp, x, dfx)
    vals[1:nlp_nnzj] .= nlp_vals
    vals[nlp_nnzj+1:sqm.meta.nnzj-1] .= dfx
    vals[sqm.meta.nnzj] = -1.0
    #display("End jc")
    return vals
end
#
# constraints Jacobian structure
# jac_structure!(nlp, rows, cols)
function NLPModels.jac_structure!(sqm::SequoiaModel2, rows::Vector{<:Integer}, cols::Vector{<:Integer})
    increment!(sqm, :neval_jac)
    if length(rows) != sqm.meta.nnzj
        error("wrong length of argument rows")
    end
    if length(cols) != sqm.meta.nnzj
        error("wrong length of argument cols")
    end
    #display("ATTEMPT js")
    nlp_nnzj = sqm.nlp.meta.nnzj
    nlp_rows = zeros(Int32, nlp_nnzj)
    nlp_cols = zeros(Int32, nlp_nnzj)
    jac_structure!(sqm.nlp, nlp_rows, nlp_cols)
    rows[1:nlp_nnzj] .= nlp_rows
    cols[1:nlp_nnzj] .= nlp_cols
    rows[nlp_nnzj+1:sqm.meta.nnzj] .= [sqm.nlp.meta.ncon+1 for i = 1:sqm.meta.nvar]
    cols[nlp_nnzj+1:sqm.meta.nnzj] .= [i for i = 1:sqm.meta.nvar]
    #display("END js")
    return rows, cols
end
#
# constraints transposed Jacobian-vector product
# Jtv = jtprod!(nlp, x, v, Jtv)
function NLPModels.jtprod!(sqm::SequoiaModel2, xl::AbstractVector{<:Float64}, v::AbstractVector{<:Float64}, Jtv::AbstractVector{<:Float64})
    increment!(sqm, :neval_jtprod)
    if length(xl) != sqm.meta.nvar
        error("wrong length of argument xl")
    end
    if length(v) != sqm.meta.ncon
        error("wrong length of argument v")
    end
    if length(Jtv) != sqm.meta.nvar
        error("wrong length of argument Jtv")
    end
    ##display("ATTEMPT EVALJT")
    x = Vector(xl[1:sqm.nx])  # Convert SubArray to Vector
    dfx = similar(x)
    grad!(sqm.nlp, x, dfx)
    nlp_Jtv = similar(x)
    v_nlp = v[1:end-1];
    v_e   = v[end];
    jtprod!(sqm.nlp, x, v_nlp, nlp_Jtv)
    Jtv[1:sqm.nx] .= nlp_Jtv + dfx * v_e
    #Jtv[1:sqm.nx] += dfx * v[end] 
    Jtv[sqm.nx+1] = -v_e
    ##display("END EVALJT")
    return Jtv
end

function NLPModels.jprod_nln!(sqm::SequoiaModel2, xl::AbstractVector{<:Float64}, v::AbstractVector{<:Float64}, Jv::AbstractVector{<:Float64})
    # Ensure x is a full Vector (not a SubArray) and allocate similar vectors for other quantities
    x = Vector(xl[1:sqm.nx])
    dfx = similar(x)
    grad!(sqm.nlp, x, dfx)

    # Check that Jv has enough space for nx + 1 elements
    #if length(Jv) < sqm.nx + 1
    #    error("Jv must have at least $(sqm.nx + 1) elements")
    #end
    #display(length(xl))
    #display(length(v))
    #display(length(Jv))

    # Compute the Jacobian transpose product
    nlp_Jv = similar(Jv, length(Jv)-1)
    v_nlp = v[1:end-1]  # First part of v (related to original constraints)
    v_e = v[end]        # Last element of v (related to the slack constraint)
    jprod!(sqm.nlp, x, v_nlp, nlp_Jv)
    #display(nlp_Jv)
    #display(Jv)
    # Set the Jacobian-vector product in the result vector Jv
    Jv[1:end-1] = nlp_Jv
    #display(dfx' * v_nlp - v_e)
    #display(typeof(Jv[end]))
    # Compute the additional slack term and assign to the last element
    Jv[end] = dfx' * v_nlp - v_e
    return Jv
end




#
# Lagrangian Hessian structure
# hess_structure!(nlp, rows, cols)
function NLPModels.hess_structure!(sqm::SequoiaModel2, rows::Vector{<:Integer}, cols::Vector{<:Integer})
    increment!(sqm, :neval_hess)
    if length(rows) != sqm.meta.nnzh
        error("wrong length of argument rows")
    end
    if length(cols) != sqm.meta.nnzh
        error("wrong length of argument cols")
    end
    ##display("ATT HESS")
    nlp_nnzh = sqm.nlp.meta.nnzh
    nlp_rows = zeros(Int64, nlp_nnzh)
    nlp_cols = zeros(Int64, nlp_nnzh)
    hess_structure!(sqm.nlp, nlp_rows, nlp_cols)
    rows[1:nlp_nnzh] .= nlp_rows
    cols[1:nlp_nnzh] .= nlp_cols
    ##display("end HESS")
    return (rows, cols)
end

# Lagrangian Hessian coordinates
# vals = hess_coord!(nlp, x, y, vals; obj_weight=1.0)
function NLPModels.hess_coord!(sqm::SequoiaModel2, xl::Vector{<:Float64}, y::Vector{<:Float64}, vals::Vector{<:Float64}; obj_weight = 0.0)
    increment!(sqm, :neval_hess)
    if length(xl) != sqm.meta.nvar
        error("wrong length of argument xl")
    end
    if length(y) != sqm.meta.ncon
        error("wrong length of argument y")
    end
    if length(vals) != sqm.meta.nnzh
        error("wrong length of argument vals")
    end
    ##display("ATT HC")
    x = Vector(xl[1:sqm.nx])
    nlp_nnzh = sqm.nlp.meta.nnzh
    nlp_vals = zeros(eltype(xl), nlp_nnzh)

    y_nlp = y[1:end-1];
    y_e   = y[end];
    hess_coord!(sqm.nlp, x, y_nlp, nlp_vals, obj_weight = y_e)
    vals[1:nlp_nnzh] .= nlp_vals
    ##display("END HC")
    return vals
end

# Lagrangian Hessian-vector product
# Hv = hprod!(nlp, x, y, v, Hv; obj_weight=1.0)
function NLPModels.hprod!(sqm::SequoiaModel2, xl::AbstractVector{<:Float64}, y::AbstractVector{<:Float64}, v::AbstractVector{<:Float64}, Hv::AbstractVector{<:Float64}; obj_weight::Float64 = 0.0)
    increment!(sqm, :neval_hprod)
    if length(xl) != sqm.meta.nvar
        error("wrong length of argument xl")
    end
    if length(y) != sqm.meta.ncon
        error("wrong length of argument y")
    end
    if length(v) != sqm.meta.nvar
        error("wrong length of argument v")
    end
    if length(Hv) != sqm.meta.nvar
        error("wrong length of argument Hv")
    end
    #display("ATT Hp")
    x = Vector(xl[1:sqm.nx])
    nlp_Hv = similar(x)

    v_nlp = v[1:end-1];
    v_e   = v[end];
    y_nlp = y[1:end-1];
    y_e   = y[end];

    Hv[1:sqm.nx] .= hprod!(sqm.nlp, x, y_nlp, v_nlp, nlp_Hv, obj_weight = y_e)
    Hv[1:sqm.nx] .= nlp_Hv 
    Hv[sqm.nx + 1]=0.0;
    #Hv[end] = 0.0
    #display("END Hp")
    return Hv
end

# regip called with an NLPModel
function sequoia2(nlp::AbstractNLPModel; kwargs...)
    return sequoia2(SequoiaModel2(nlp); kwargs...)
end

"""
Main function for RegIP method.
"""
function sequoia2(
    sqm::SequoiaModel2; # Problem to be solved by this method
    
    # initial primal-dual pair
    x0::Vector{<:AbstractFloat} = sqm.nlp.meta.x0,
    y0::Vector{<:AbstractFloat} = sqm.nlp.meta.y0,

    # termination
    tol::Real = eltype(x0)(1e-4), # tolerance
    max_time::Float64 = 60.0, # max time (s)
)

    start_time = time()
    # allocations
    x_seq2 = similar(sqm.meta.x0)
    y_seq2 = similar(sqm.meta.y0)
    #fx = similar(sqm.t)
    #fx = obj(sqm.nlp, x0)
    x_seq2[1:sqm.nx] .= x0
    x_seq2[sqm.nx+1] = obj(sqm.nlp, x0)
    y_seq2[1:sqm.ny] .= y0
    y_seq2[sqm.ny+1] = 1.0

    #display(cons(sqm,x_seq2))
    #display(cons(sqm.nlp,x0))

    # solve subproblem
    #sub_output = ipopt(
    #    sqm,
    #    tol = tol,
    #    dual_inf_tol = Inf,
    #    constr_viol_tol = Inf,
    #    compl_inf_tol = Inf,
    #    acceptable_iter = 0,
    #    print_level = 0,
    #   max_cpu_time = max_time,
    #    max_iter = Int(1e8),
    #    x0 = x_seq2,
    #    y0 = y_seq2,
    #)      
    #display(typeof(x_seq2)  )
        sub_output =  percival(
            sqm,
            x =x_seq2,
            inity =true,
            atol = tol,
            rtol = tol,
            ctol = tol,
            max_time = max_time,
            max_iter = Int(1e8),
        )

    el_time = time() - start_time

    return GenericExecutionStats(
        sqm,
        status = sub_output.status,
        solution = sub_output.solution,
        iter = sub_output.iter,
        primal_feas = sub_output.primal_feas,
        objective = sub_output.solution[end],
        elapsed_time = el_time
    )

end


using NLPModelsIpopt, Percival, ADNLPModels, NLPModels
using CUTEst # test set
using SolverBenchmark # benchmarking tools
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

seq_solver =
    nlp -> sequoia2(
        nlp,
        x0 = nlp.meta.x0,
        y0 = nlp.meta.y0,
        tol = TOL,
        max_time = MAX_TIME)

ipopt_solver =
        nlp -> ipopt(
            nlp,
            x0 = nlp.meta.x0,
            y0 = nlp.meta.y0,
            tol = TOL,
            max_cpu_time = MAX_TIME,
            max_iter = MAX_ITER,
            print_level = 0,
            dual_inf_tol = Inf,
            constr_viol_tol = Inf,
            compl_inf_tol = Inf,
            acceptable_iter = 0,
        )

perci_solver =
        nlp -> percival(
            nlp,
            inity = true,
            #inity = nlp.meta.y0,
            atol = TOL,
            rtol = TOL,
            ctol = TOL,
            max_time = MAX_TIME,
            max_iter = MAX_ITER,
        )

    seqp_stats = solve_problems(seq_solver, :seqp, problems)
    @save "seq_" * attrname * "_seqp22.jld2" seqp_stats

    #ipopt_stats = solve_problems(ipopt_solver, :ipo, problems)
    #@save "ipo_" * attrname * "_ipo22.jld2" ipopt_stats

    #perci_stats = solve_problems(perci_solver, :per, problems)
    #@save "perci_" * attrname * "_perci22.jld2" perci_stats
