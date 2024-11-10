using NLPModels
export cutest_to_sequoia

"""
    cutest_to_sequoia(cutest_problem::CUTEstModel)

Converts a CUTEst problem into a `SEQUOIA_pb` instance.

# Arguments
- `cutest_problem::CUTEstModel`: The CUTEst problem instance to be converted.

# Returns
- A `SEQUOIA_pb` instance initialized with data from the CUTEst problem.
"""
function cutest_to_sequoia(cutest_problem::CUTEstModel)::SEQUOIA_pb
    meta = cutest_problem.meta #Extract metadata

    nvar = meta.nvar # Retrieve the number of variables from the CUTEst problem
    minimization = meta.minimize # Is the CUTEst problem a minimization problem?
    x0 = cutest_problem.meta.x0 # Retrieve the initial guess from the CUTEst problem
    
    objective_fn = x -> obj(cutest_problem, x) # Define the objective function using CUTEst's interface  
    gradient_fn = x -> grad(cutest_problem, x) # Define the gradient function using NLPModels.gradient 

    # Initialize empty fields for constraints
    constraints_fn = nothing
    jacobian_fn = nothing
    eqcon = Int[]
    ineqcon = Int[]
    
    if cutest_problem.meta.ncon > 0
        
        constraints_fn = x -> cons(cutest_problem, x) # Define the constraints function using CUTEst's interface
        jacobian_fn = x -> jac(cutest_problem, x) # Define the Jacobian using CUTEst's interface
        
        # Determine equality and inequality constraint indices 
        eqcon = cutest_problem.meta.jfix # Equality constraint indices
        ineqcon = sort(vcat(cutest_problem.meta.jlow, cutest_problem.meta.jupp, cutest_problem.meta.jrng)) # Inequality constraint indices
    end
    
    # Create a SEQUOIA_pb instance
    pb = SEQUOIA_pb(
        nvar;
        x0 = x0,
        is_minimization = minimization,          
        objective = objective_fn,
        gradient = gradient_fn,
        constraints = constraints_fn,
        jacobian = jacobian_fn,
        eqcon = eqcon,
        ineqcon = ineqcon,
        cutest_nlp=cutest_problem
    )
    
    try
        validate_pb!(pb)
    catch e
        println("Validation failed for the converted SEQUOIA_pb instance. Error: ", e)
        rethrow(e)
    end

    return pb
end
