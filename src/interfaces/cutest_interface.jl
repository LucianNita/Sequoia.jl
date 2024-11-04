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
    # Retrieve the number of variables from the CUTEst problem
    nvar = cutest_problem.meta.nvar

    # Is the CUTEst problem a minimization problem?
    minimization = cutest_problem.meta.minimize
    
    # Retrieve the initial guess from the CUTEst problem
    x0 = cutest_problem.meta.x0
    
    # Define the objective function using CUTEst's interface
    objective_fn = x -> obj(cutest_problem, x)
    
    # Define the gradient function using NLPModels.gradient (optional: automatic differentiation if not provided)
    gradient_fn = x -> grad(cutest_problem, x)
    
    # Check if the problem has constraints
    has_constraints = cutest_problem.meta.ncon > 0 #+length(cutest_problem.meta.ilow)+length(cutest_problem.meta.iupp)+length(cutest_problem.meta.ifix) > 0
    
    # Initialize empty fields for constraints
    constraints_fn = nothing
    jacobian_fn = nothing
    eqcon = Int[]
    ineqcon = Int[]
    
    if has_constraints
        # Define the constraints function using CUTEst's interface
        constraints_fn = x -> cons(cutest_problem, x)
        
        # Define the Jacobian using CUTEst's interface
        jacobian_fn = x -> jac(cutest_problem, x)
        
        # Determine equality and inequality constraint indices 
        eqcon = cutest_problem.meta.jfix # Equality constraint indices
        ineqcon = sort(vcat(cutest_problem.meta.jlow, cutest_problem.meta.jupp, cutest_problem.meta.jrng)) # Inequality constraint indices
    end
    
    # Create a SEQUOIA_pb instance
    pb = SEQUOIA_pb(
        nvar,
        x0 = x0,
        is_minimization = minimization,          # Assume minimization problem by default
        objective = objective_fn,
        gradient = gradient_fn,
        constraints = constraints_fn,
        jacobian = jacobian_fn,
        eqcon = eqcon,
        ineqcon = ineqcon,
        cutest_nlp=cutest_problem
    )
    
    #validate_cutest_to_sequoia(pb,cutest_problem);

    return pb
end
