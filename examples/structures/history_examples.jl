"""
# SEQUOIA_History Examples

This file contains example use cases for the `SEQUOIA_History` struct, showcasing its usage:
1. Adding iterates to a history.
2. Retrieving all field values.
3. Clearing the history.
4. Handling invalid inputs for adding iterates.
5. Handling invalid inputs for retrieving fields.
"""

using Sequoia  # Assuming Sequoia module contains SEQUOIA_History and SEQUOIA_Solution_step

# Example 1: Adding Iterates to a History
"""
This example demonstrates how to create a `SEQUOIA_History` instance and add iterates to it.

# Usage:
    example_add_iterates()

# Expected Output:
    SEQUOIA_History with 3 iterates
"""
function example_add_iterates()
    history = SEQUOIA_History()  # Create an empty history

    # Create and add some iterates
    iterate1 = SEQUOIA_Solution_step(1, 0.01, :first_order, 0.1, 5, [0.5, 1.0], -10.0, [0.0, 0.0])
    iterate2 = SEQUOIA_Solution_step(2, 0.02, :max_iter, 0.2, 10, [0.6, 1.1], -9.5, [0.01, -0.01])
    iterate3 = SEQUOIA_Solution_step(3, 0.03, :acceptable, 0.3, 15, [0.7, 1.2], -9.0, [0.02, -0.02])

    add_iterate!(history, iterate1)
    add_iterate!(history, iterate2)
    add_iterate!(history, iterate3)

    println("SEQUOIA_History with $(length(history.iterates)) iterates")
end

# Example 2: Retrieving All Field Values
"""
This example shows how to extract all values for a specific field across the history.

# Usage:
    example_get_all_fields()

# Expected Output:
    Convergence metrics: [0.01, 0.02, 0.03]
"""
function example_get_all_fields()
    history = SEQUOIA_History([
        SEQUOIA_Solution_step(1, 0.01, :first_order, 0.1, 5, [0.5, 1.0], -10.0, [0.0, 0.0]),
        SEQUOIA_Solution_step(2, 0.02, :max_iter, 0.2, 10, [0.6, 1.1], -9.5, [0.01, -0.01]),
        SEQUOIA_Solution_step(3, 0.03, :acceptable, 0.3, 15, [0.7, 1.2], -9.0, [0.02, -0.02])
    ])

    # Retrieve convergence metrics from all iterates
    convergence_metrics = get_all(history, :convergence_metric)
    println("Convergence metrics: $convergence_metrics")
end

# Example 3: Clearing the History
"""
This example demonstrates how to clear all iterates from a `SEQUOIA_History`.

# Usage:
    example_clear_history()

# Expected Output:
    SEQUOIA_History cleared. Number of iterates: 0
"""
function example_clear_history()
    history = SEQUOIA_History([
        SEQUOIA_Solution_step(1, 0.01, :first_order, 0.1, 5, [0.5, 1.0], -10.0, [0.0, 0.0])
    ])
    
    clear_history!(history)
    println("SEQUOIA_History cleared. Number of iterates: $(length(history.iterates))")
end

# Example 4: Handling Invalid Inputs for Adding Iterates
"""
This example demonstrates how the fallback method for `add_iterate!` handles invalid inputs.

# Usage:
    example_invalid_iterates()

# Expected Output:
    Error: Expected `SEQUOIA_Solution_step`, but got `String`.
    Error: Expected `SEQUOIA_History`, but got `String`.
"""
function example_invalid_iterates()
    history = SEQUOIA_History()
    
    try
        add_iterate!(history, "invalid_iterate")  # Invalid input
    catch e
        println("Error: ", e)
    end

    try
        add_iterate!("invalid_history", SEQUOIA_Solution_step(1, 0.01, :first_order, 0.1, 5, [0.5, 1.0], -10.0, [0.0, 0.0]))
    catch e
        println("Error: ", e)
    end
end

# Example 5: Handling Invalid Inputs for Retrieving Fields
"""
This example demonstrates how the fallback method for `get_all` handles invalid inputs.

# Usage:
    example_invalid_fields()

# Expected Output:
    Error: Expected `SEQUOIA_History`, but got `String`.
    Error: Invalid field: `invalid_field`. Valid fields are: (:outer_iteration_number, :convergence_metric, :solver_status, :inner_comp_time, :num_inner_iterations, :x, :fval, :gval, :cval, :solver_params, :x_iterates)
    Error: Invalid input: `field` must be of type `Symbol`.
    """
function example_invalid_fields()
    history = SEQUOIA_History([
        SEQUOIA_Solution_step(1, 0.01, :first_order, 0.1, 5, [0.5, 1.0], -10.0, [0.0, 0.0])
    ])

    try
        get_all("invalid_history", :x)  # Invalid history input
    catch e
        println("Error: ", e)
    end

    try
        get_all(history, :invalid_field)  # Invalid field
    catch e
        println("Error: ", e)
    end

    try
        get_all(history, "invalid_field")  # Invalid field type
    catch e
        println("Error: ", e)
    end
end

# Call All Examples
example_add_iterates()
example_get_all_fields()
example_clear_history()
example_invalid_iterates()
example_invalid_fields()
