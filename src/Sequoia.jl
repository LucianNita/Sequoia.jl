module Sequoia

import LinearAlgebra
import Optim

include("structs.jl")

"""
This is the main struct for SEQUOIA problems. It stores solutions, settings, problem data and iteration dependent parameters (t & guess) 
"""
mutable struct SEQUOIA
    nvar::Integer     #Problem dimension

    objective::Function     #Cost function
    objSense::String        #MIN, MAX, FEAS 

    constraints::Function   #Function handle storing constraints 
    eqcon::Vector{Int}      #Indices of equality constraints. Assumes c_i(x)=0
    ineqcon::Vector{Int}    #Indices of inequality constraints. Assumes c_i(x)≤0
    

    x0::Vector{Float64}  #This is the starting point for the "current" iteration
    t::Vector{Float64}   #This is the "penalty parameter" of SEQUOIA parametrizing the sequence produced

    settings::SEQUOIA_Settings #General problem settings
    solutionHist::Vector{Optim.MultivariateOptimizationResults} #Vector of past solutions in "Optim.jl" format
    History::SEQUOIA_Hist  #This stores past data about past parameters and guesses
    exitCode::Integer      #Did the algorithm terminat? 0:No; >0:Yes. -1: optimize not called yet; 1:optimality tolerance reached successfully solved; 2:Infeasibility catch; 3:Maximum number of iterations reached; 4:Unbounded catch; 

    SEQUOIA(nvar::Integer; obj::Function, sense="FEAS+MIN", cons::Function, eqcon=[], ineqcon=[], x0::Vector{Float64}, t=obj(x0))= new(nvar; obj,sense,cons,eqcon,ineqcon,x0,t); #The most basic constructor
    #with obj&cons
    #with guess
    #no size, just guess
end

end # module Sequoia

