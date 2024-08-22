"""
This struct stores the main settings required by SEQUOIA. Apart from feasibility which is just a mode, no other settings would dynamically change during run.
"""
mutable struct SEQUOIA_Settings 
    optimizer                       #The inner solver used for solving the unconstrained problem. Can be "LBFGS()", "BFGS()", "Newton()", or "GradientDescent()". Others are possible but not tested. 
    max_iter::Integer               #Maximum number of outer iterations allowed
    resid_tolerance::Float64        #Residual tolerance (ie when I consider my constraints satisfied)
    cost_tolerance::Float64         #Desired optimality gap
    cost_min::Float64               #Minimum cost - useful for spotting possible unbounded problems

    method::String                  #Only SEQUOIA, QPM, AugLag and IntPt supported for now
    feasibility::Bool               #Do we solve the feasibility problem or is an objective accounted? 
    
    SEQUOIA_Settings(;optimizer=LBFGS(),max_iter=3000,resid_tolerance=10^-6,cost_tolerance=10^-2, cost_min=-10^10, method="SEQUOIA", feasibility=true) = new(optimizer,max_iter,resid_tolerance,cost_tolerance, cost_min, method, feasibility) #Basic constructor function with defaults
    
    SEQUOIA_Settings(optimizer; max_iter=3000,resid_tolerance=10^-6,cost_tolerance=10^-2, cost_min=-10^10, method="SEQUOIA", feasibility=true) = new(optimizer,max_iter,resid_tolerance,cost_tolerance, cost_min, method, feasibility) #
end #Throw error if wrong type of optimizer




"""
This struct stores a history of past SEQUOIA iterations. Useful for warm-starting. 
"""
mutable struct SEQUOIA_Hist
    guess::Vector{Vector{Float64}} #All the past initial guesses, for info only
    tHist::Vector{Float64}         #History of past parameters (note for QPM t is used as μ)
    rHist::Vector{Vector{Float64}} #History of all constraint residuals (computed with respect to the penalty function)
    tu::Vector{Integer}            #Indices of iterations where a cost upper bound was produced (useful in SEQUOIA only) 
    tl::Vector{Integer}            #Indices of iterations where a cost lower bound was produced (useful in SEQUOIA only) 

    SEQUOIA_Hist(guess::Vector{Float64}, t::Float64, res::Vector{Float64}) = new([guess], [t], [res], Vector{Int}(undef, 0), Vector{Int}(undef, 0) ); #Basic constructor function
end


function add_History!(pastHist::SEQUOIA_Hist, guess::Vector{Float64}, t::Float64, res::Vector{Float64}, tu=nothing, tl=nothing)
    push!(pastHist.guess,guess);
    push!(pastHist.tHist,t);
    push!(pastHist.rHist,res);
    if !isnothing(tu)
        push!(pastHist.tu,tu);
    end
    if !isnothing(tl)
        push!(pastHist.tl,tl);
    end
end

#function set_History!(pb::SEQUOIA) 
#    pb.History=SEQUOIA_Hist(pb.guess,pb.t);
#    if pb.settings.method=="SEQUOIA"
#        tu,tl,r=check_residual(pb);
#        pb.History.rHist=[r];
#        pb.History.tu=[tu];
#        pb.History.tl=[tl];
#    end
#end


#set_solutionHist!(pb::SEQUOIA,sol::SEQUOIA_Hist) = pb.solutionHist=[sol]; #Setting the solution history for the first time from undef state

#add_solutionHist!(pb::SEQUOIA,sol::SEQUOIA_Hist) = push!(pb.solutionHist,sol); #Adding to the solution history vector (vector already exists, we just push iterates)

