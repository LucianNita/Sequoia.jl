"""
# Example 1: Using the Full Constructor

This example shows how to use the full constructor for `SEQUOIA_History`, adding multiple solution steps and retrieving values using the `get_all` function.

```julia
example_full_history()

history = SEQUOIA_History()
step1 = SEQUOIA_Solution_step(
    1, 1e-6, :success, 0.02, 2, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1]]
)
step2 = SEQUOIA_Solution_step(
    2, 1e-6, :success, 0.03, 2, [1.1, 2.1], 0.45, [0.05, 0.15], nothing, [0.1, 0.2], [[1.1, 2.1], [1.2, 2.2]]
)
add_iterate!(history, step1)
add_iterate!(history, step2)

# Retrieve all `fval` from the history
all_fvals = get_all(history, :fval)
println(all_fvals)

Example output:

[0.5, 0.45]
"""
function example_full_history()
    history = SEQUOIA_History()
    step1 = SEQUOIA_Solution_step(
        1, 1e-6, :success, 0.02, 2, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1]]
    )
    step2 = SEQUOIA_Solution_step(
        2, 1e-6, :success, 0.03, 2, [1.1, 2.1], 0.45, [0.05, 0.15], nothing, [0.1, 0.2], [[1.1, 2.1], [1.2, 2.2]]
    )
    add_iterate!(history, step1)
    add_iterate!(history, step2)

    all_fvals = get_all(history, :fval)
    println(all_fvals)
end


"""
# Example 2: Testing Validation for `add_iterate!`

This example demonstrates how to use `add_iterate!` for adding valid solution steps to the `SEQUOIA_History` and shows how the function validates inputs. It catches errors when invalid inputs are provided for both `history` and `iterate`.

```julia
example_add_iterate_validation()

history = SEQUOIA_History()
step = SEQUOIA_Solution_step(
    1, 1e-6, :success, 0.02, 2, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1]]
)
add_iterate!(history, step)

# This will raise an error because the `history` argument is not of type `SEQUOIA_History`
try
    add_iterate!([1, 2, 3], step)
catch e
    println(e)
end

# This will raise an error because the `iterate` argument is not of type `SEQUOIA_Solution_step`
try
    add_iterate!(history, [1.0, 2.0])
catch e
    println(e)
end

Example output:

ArgumentError: Invalid input: `history` must be of type `SEQUOIA_History`.
ArgumentError: Invalid input: `iterate` must be of type `SEQUOIA_Solution_step`.
"""
function example_add_iterate_validation()
    history = SEQUOIA_History()
    step = SEQUOIA_Solution_step(
        1, 1e-6, :success, 0.02, 2, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1]]
    )
    add_iterate!(history, step)

    # Test with an invalid history input
    try
        add_iterate!([1, 2, 3], step)
    catch e
        println(e)
    end

    # Test with an invalid iterate input
    try
        add_iterate!(history, [1.0, 2.0])
    catch e
        println(e)
    end
end

"""
# Example 3: Retrieving All Field Values with `get_all`

This example demonstrates how to add multiple solution steps to `SEQUOIA_History` and use the `get_all` function to retrieve specific fields from the stored steps, such as the solution vectors `x` and the gradient values `gval`.

```julia
example_get_all_fields()

history = SEQUOIA_History()
step1 = SEQUOIA_Solution_step(
    1, 1e-6, :success, 0.02, 2, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1]]
)
step2 = SEQUOIA_Solution_step(
    2, 1e-6, :success, 0.03, 2, [1.1, 2.1], 0.45, [0.05, 0.15], nothing, [0.1, 0.2], [[1.1, 2.1], [1.2, 2.2]]
)
add_iterate!(history, step1)
add_iterate!(history, step2)

# Retrieve all solution vectors `x` from the history
all_x_values = get_all(history, :x)
println(all_x_values)

# Retrieve all gradient values `gval` from the history
all_gvals = get_all(history, :gval)
println(all_gvals)

Example output:

[[1.0, 2.0], [1.1, 2.1]]
[[0.1, 0.2], [0.05, 0.15]]
"""
function example_get_all_fields()
    history = SEQUOIA_History()
    step1 = SEQUOIA_Solution_step(
        1, 1e-6, :success, 0.02, 2, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1]]
    )
    step2 = SEQUOIA_Solution_step(
        2, 1e-6, :success, 0.03, 2, [1.1, 2.1], 0.45, [0.05, 0.15], nothing, [0.1, 0.2], [[1.1, 2.1], [1.2, 2.2]]
    )
    add_iterate!(history, step1)
    add_iterate!(history, step2)

    # Retrieve all `x` values from the history
    all_x_values = get_all(history, :x)
    println(all_x_values)

    # Retrieve all `gval` values from the history
    all_gvals = get_all(history, :gval)
    println(all_gvals)
end

"""
# Example 4: Handling Invalid Field Access with `get_all`

This example shows how to handle an invalid field access in `SEQUOIA_History` using the `get_all` function. When an invalid field is specified, an error is raised and caught in the `try-catch` block.

```julia
example_invalid_field()

history = SEQUOIA_History()
step = SEQUOIA_Solution_step(
    1, 1e-6, :success, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]]
)
add_iterate!(history, step)

# Attempt to retrieve values for an invalid field
try
    get_all(history, :invalid_field)
catch e
    println(e)
end

Example output:

ArgumentError: Invalid field: `invalid_field` is not a valid field of `SEQUOIA_Solution_step`.
"""
function example_invalid_field()
    history = SEQUOIA_History()
    step = SEQUOIA_Solution_step(
        1, 1e-6, :success, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]]
    )
    add_iterate!(history, step)

    # Attempt to retrieve values for an invalid field
    try
        get_all(history, :invalid_field)
    catch e
        println(e)  # Prints the error message
    end
end

function example_get_convergence_metrics()
    history = SEQUOIA_History()
    step1 = SEQUOIA_Solution_step(
        1, 1e-6, :success, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1]]
    )
    step2 = SEQUOIA_Solution_step(
        2, 1e-5, :success, 0.03, 6, [1.1, 2.1], 0.45, [0.05, 0.15], nothing, [0.1, 0.2], [[1.1, 2.1], [1.2, 2.2]]
    )
    add_iterate!(history, step1)
    add_iterate!(history, step2)

    all_convergence_metrics = get_all(history, :convergence_metric)
    println(all_convergence_metrics)
end
