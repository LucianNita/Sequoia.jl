# Verification function to ensure input consistency
"""
    verify_input_consistency(cons_fn, x0, lb, ub, eq_indices, ineq_indices)

Verifies that the constraints are consistent:
- The number of constraints returned by `cons_fn(x)` must match the total number of constraints given by `eq_indices` and `ineq_indices`.
- Indices in `eq_indices` and `ineq_indices` must cover exactly the set of all constraints (i.e., all indices from `1` to `length(cons_fn(x))` must appear exactly once).
- The lower bounds and upper bounds must have consistent sizes, and the lower bound must be <= the upper bound.

# Arguments:
- `cons_fn`: A function that returns the constraint values ( g(x) ), which is a vector of constraints.
- `x0`: The initial guess for the decision variables ( x ), used to check the number of constraints.
- `lb`: A vector of lower bounds for the constraints.
- `ub`: A vector of upper bounds for the constraints.
- `eq_indices`: A vector containing the indices of the constraints that are equality constraints. These indices refer to entries in ( g(x) ) where ( g_i(x) = lb_i = ub_i ).
- `ineq_indices`: A vector containing the indices of the constraints that are inequality constraints. These indices refer to entries in ( g(x) ) where ( lb_i ≤ g_i(x) ≤ ub_i ).

# Throws:
- `error`: If the number of constraints returned by `cons_fn(x0)` does not match the combined size of `eq_indices` and `ineq_indices`.
- `error`: If the indices in `eq_indices` and `ineq_indices` do not cover all constraints exactly once.
- `error`: If the size of `lb` and `ub` are not equal.
- `error`: If any element in `lb` is greater than its corresponding element in `ub`.
"""
function verify_input_consistency(cons_fn, x0, lb, ub, eq_indices, ineq_indices)
    # Check if lower and upper bounds have the same length
    if length(lb) != length(ub)
        error("Lower bounds and upper bounds must be of equal length.")
    end

    # Ensure each lower bound is less than or equal to its corresponding upper bound
    if any(lb .> ub)
        error("Each lower bound must be less than or equal to the corresponding upper bound.")
    end

    # Ensure the number of constraints matches the number of specified indices
    num_constraints = length(cons_fn(x0))
    total_specified_constraints = length(eq_indices) + length(ineq_indices)
    if num_constraints != total_specified_constraints
        error("The number of constraints returned by the constraint function does not match the total number of specified constraints (equality + inequality).")
    end

    # Ensure all indices are covered exactly once by eq_indices and ineq_indices
    all_indices = sort(vcat(eq_indices, ineq_indices))
    if all_indices != collect(1:num_constraints)
        error("Indices for equality and inequality constraints must cover all constraint indices exactly once.")
    end
end