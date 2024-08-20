module Sequoia

import LinearAlgebra
import Optim

include("structs.jl")

"""
SEQUOIA is the main data strucutre of the package and stores the problem definition for SEQUOIA problems.

The `SEQUOIA` struct stores information related to the optimization problem including:
- Problem dimension (`nvar`).
- Objective function and its sense (feasibility, minimization, maximization, etc.).
- Constraints functions along with indices associated to equality and inequality constraints.
- Initial guess for the variables and penalty parameter.
- It stores solutions, settings, problem data and iteration dependent parameters (t & guess) 

This module also provides functions to modify the optimization problem by setting or updating the objective, sense, constraints, and initial guess.
"""
mutable struct SEQUOIA
    nvar::Integer     # Problem dimension (number of variables)

    objective::Function     # Cost function to be minimized or maximized
    objSense::String        # Optimization sense: MIN, MAX, FEAS, etc.

    constraints::Vector{Function}   # Vector of constraint functions
    eqcon::Vector{Int}      # Indices of equality constraints. Assumes c_i(x)=0
    ineqcon::Vector{Int}    # Indices of inequality constraints. Assumes c_i(x)≤0
    
    x0::Vector{Float64}  # Initial guess for the variables
    t::Float64   # "Penalty parameter used in the SEQUOIA method

    #settings::SEQUOIA_Settings #General problem settings
    #solutionHist::Vector{Optim.MultivariateOptimizationResults} #Vector of past solutions in "Optim.jl" format
    #History::SEQUOIA_Hist  #This stores past data about past parameters and guesses
    #exitCode::Integer      #Did the algorithm terminat? 0:No; >0:Yes. -1: optimize not called yet; 1:optimality tolerance reached successfully solved; 2:Infeasibility catch; 3:Maximum number of iterations reached; 4:Unbounded catch; 
    
    """
    Constructs a new SEQUOIA object.

    Arguments:
    - `nvar`: Number of variables in the problem.
    - `obj`: Objective function (default is zero function).
    - `sense`: Optimization sense ("FEAS+MIN" by default).
    - `cons`: Vector of constraint functions (default is an empty vector).
    - `eqcon`: Indices of equality constraints (default is an empty vector).
    - `ineqcon`: Indices of inequality constraints (default is an empty vector).
    - `x0`: Initial guess for the variables (default is a zero vector).
    - `t`: Penalty parameter (default is the objective function evaluated at `x0`).

    Different constructor variants are provided to accommodate various initial conditions.
    """
    SEQUOIA(nvar::Integer; obj::Function=x->0.0, sense="FEAS+MIN", cons::Vector{Function}=Vector{Function}(undef, 0), eqcon=Vector{Int}(undef, 0), ineqcon=Vector{Int}(undef, 0), x0::Vector{Float64}=zeros(nvar), t=obj(x0) )= new(nvar,obj,sense,cons,eqcon,ineqcon,x0,t)
    
    SEQUOIA(nvar::Integer, obj::Function; sense="FEAS+MIN", cons::Vector{Function}=Vector{Function}(undef, 0), eqcon=Vector{Int}(undef, 0), ineqcon=Vector{Int}(undef, 0), x0::Vector{Float64}=zeros(nvar), t=obj(x0) )= new(nvar,obj,sense,cons,eqcon,ineqcon,x0,t)
    
    SEQUOIA(nvar::Integer, obj::Function, sense::String, cons::Function, eqcon=Vector{Int}, ineqcon=Vector{Int}; x0::Vector{Float64}=zeros(nvar), t=obj(x0) )= new(nvar,obj,sense,[cons],eqcon,ineqcon,x0,t)
    
    SEQUOIA(obj::Function, sense::String, cons::Function, eqcon=Vector{Int}, ineqcon=Vector{Int}, x0::Vector{Float64}; t=obj(x0) )= new(length(x0),obj,sense,[cons],eqcon,ineqcon,x0,t)
    
    SEQUOIA(obj::Function, sense::String, cons::Function, eqcon=Vector{Int}, ineqcon=Vector{Int}, x0::Vector{Float64}, t::Float64 )= new(length(x0),obj,sense,[cons],eqcon,ineqcon,x0,t)
end

"""
Sets the objective function of the SEQUOIA problem.

# Arguments
- `pb`: The SEQUOIA problem instance.
- `obj`: The new objective function to be set.
"""
function set_objective!(pb::SEQUOIA, obj::Function)
    pb.objective = obj
end

"""
Sets the optimization sense of the SEQUOIA problem.

Valid values for `sense` are: "FEAS", "MIN", "MAX", "FEAS+MIN", "FEAS+MAX".

# Arguments
- `pb`: The SEQUOIA problem instance.
- `sense`: The new optimization sense to be set.
"""
function set_sense!(pb::SEQUOIA, sense::String)
    if sense in ["FEAS", "MIN", "MAX", "FEAS+MIN", "FEAS+MAX"]
        pb.objSense = sense
    else
        error("Sense has to be either \"FEAS\", \"MIN\", \"MAX\", \"FEAS+MIN\", or \"FEAS+MAX\".")
    end
end

"""
Sets or updates the constraints of the SEQUOIA problem.

This function replaces all existing constraints.

# Arguments
- `pb`: The SEQUOIA problem instance.
- `con`: The new constraints function.
- `eqcon`: Vector of indices for new equality constraints (default is an empty vector).
- `ineqcon`: Vector of indices for new inequality constraints (default is an empty vector).
"""
function clean_set_const!(pb::SEQUOIA, con::Function, eqcon=Vector{Int}, ineqcon=Vector{Int})
    pb.constraints = [con]
    pb.eqcon = eqcon
    pb.ineqcon = ineqcon
end

"""
Adds an equality constraint to the SEQUOIA problem.

# Arguments
- `pb`: The SEQUOIA problem instance.
- `con`: The equality constraint function to be added.
"""
function add_eq_const!(pb::SEQUOIA, con::Function)
    push!(pb.constraints, con)
    l = length(pb.eqcon) + length(pb.ineqcon)
    ncon = length(con(ones(pb.nvar)))
    append!(pb.eqcon, l + 1:l + ncon)
end

"""
Adds an inequality constraint to the SEQUOIA problem.

# Arguments
- `pb`: The SEQUOIA problem instance.
- `con`: The inequality constraint function to be added.
"""
function add_ineq_const!(pb::SEQUOIA, con::Function)
    push!(pb.constraints, con)
    l = length(pb.eqcon) + length(pb.ineqcon)
    ncon = length(con(ones(pb.nvar)))
    append!(pb.ineqcon, l + 1:l + ncon)
end

"""
Sets the initial guess for the SEQUOIA problem.

# Arguments
- `pb`: The
"""

function set_guess!(pb::SEQUOIA,guess::Vector{Float64}) #Setter function. Define the initial guess
    if length(guess)==pb.nvar
        pb.x0=guess;
    else
        error("Problem dimension has to match the length of the initial guess"); #Make sure dimensions and initial guess match
    end
end

end # module Sequoia

