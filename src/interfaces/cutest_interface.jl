using NLPModels, CUTEst

# Interface function to convert a CUTEst model into a SEQUOIA_pb problem
function cutest_to_sequoia(cutest_model::CUTEst.Model, solver_settings::SEQUOIA_Settings)
    # Create an NLPModel from the CUTEst model
    nlp = CUTEstNLPModel(cutest_model)

    # Extract objective function
    obj_fn = x -> obj(nlp, x)
    
    # Extract gradient using NLPModels
    grad_fn = x -> grad(nlp, x)

    # Extract constraints function if present, else return an empty array
    if hess_structure(nlp)[1] > 0 || cons_length(nlp) > 0
        cons_fn = x -> cons(nlp, x)
        jac_fn = x -> jac(nlp, x)
    else
        cons_fn = x -> []
        jac_fn = nothing  # No constraints, no Jacobian
    end

    # Get bounds (lower and upper) and other problem information
    lb, ub = nlp.meta.lvar, nlp.meta.uvar

    # Get equality and inequality constraint indices
    eq_indices = [i for i in 1:cons_length(nlp) if nlp.meta.eqtol[i] > 0]
    ineq_indices = [i for i in 1:cons_length(nlp) if nlp.meta.eqtol[i] == 0]

    # Get initial guess for the solution
    x0 = nlp.meta.x0

    # Prepare and return SEQUOIA_pb problem structure
    sequoia_problem = SEQUOIA_pb(
        objective = obj_fn,
        gradient = grad_fn,
        constraints = cons_fn,
        jacobian = jac_fn,
        l_bounds = lb,
        u_bounds = ub,
        eqcon = eq_indices,
        ineqcon = ineq_indices,
        x0 = x0,
        nvar = length(x0),  # Number of variables
        solver_settings = solver_settings,
        solution = nothing,  # Will be filled in after solving
        solution_history = SEQUOIA_Iterates()  # Empty solution history
    )
    
    return sequoia_problem
end
