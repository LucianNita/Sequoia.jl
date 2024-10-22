export SEQUOIA_History
export add_iterate!, get_all

"""
    SEQUOIA_History

This struct stores a collection of solution steps (`SEQUOIA_Solution_step`) obtained during the iterative optimization process in the SEQUOIA solver.

# Fields

- `iterates::Vector{SEQUOIA_Solution_step}`: A vector that holds multiple iteration steps, each represented by a `SEQUOIA_Solution_step`.
"""
struct SEQUOIA_History
    iterates::Vector{SEQUOIA_Solution_step}  # A vector to hold multiple iteration steps

    # Constructor to initialize an empty SEQUOIA_History
    SEQUOIA_History() = new(Vector{SEQUOIA_Solution_step}())
end

"""
    add_iterate!(history::SEQUOIA_History, iterate::SEQUOIA_Solution_step)

Add a new `SEQUOIA_Solution_step` (an iteration step) to the `SEQUOIA_History` collection.

# Arguments

- `history::SEQUOIA_History`: An instance of `SEQUOIA_History` that holds the collection of solution steps.
- `iterate::SEQUOIA_Solution_step`: A `SEQUOIA_Solution_step` to be added to the `history`.
"""
function add_iterate!(history::SEQUOIA_History, iterate::SEQUOIA_Solution_step)
    push!(history.iterates, iterate)
end

"""
    add_iterate!(history, iterate)

Just a fallback method. This function validates the inputs and ensures that the history and iterate are of the correct types. It only dispatches on this if the types are incorrect. Otherwise, Julia's multiple dispatch, dispatches on the most specific type (which is the function actually performing the task)

# Arguments

- `history`: The instance of `SEQUOIA_History` to validate.
- `iterate`: The instance of `SEQUOIA_Solution_step` to validate.

# Throws

- `ArgumentError` if `history` is not of type `SEQUOIA_History` or if `iterate` is not of type `SEQUOIA_Solution_step`.
"""
function add_iterate!(history::Any, iterate::Any)
    validate_history(history)
    validate_iterate(iterate)
end

# Function to extract specific information
"""
    get_all(history::SEQUOIA_History, field::Symbol) -> Vector{Any}

Retrieve all values for a specified field from each iterate in `SEQUOIA_History`.

# Arguments

- `history::SEQUOIA_History`: An instance of `SEQUOIA_History` holding the collection of solution iterates.
- `field::Symbol`: The name of the field to retrieve values from (e.g., `:x`, `:fval`, `:gval`).

# Returns
A vector of values for the specified field from each iterate.
"""
function get_all(history::SEQUOIA_History, field::Symbol)
    validate_field(field)
    [getfield(iterate, field) for iterate in history.iterates]
end

"""
    get_all(history, field)

Just a fallback method. This function first validates the `history` and `field` inputs, ensuring that the history is of the correct type and the field is a valid field of `SEQUOIA_Solution_step`. It only dispatches on this if the types are incorrect. Otherwise, Julia's multiple dispatch, dispatches on the most specific type (which is the function actually performing the task)

# Arguments

- `history`: The instance of `SEQUOIA_History` to validate.
- `field`: The field to extract from the iterates, validated to be a valid field of `SEQUOIA_Solution_step`.

# Throws

- `ArgumentError` if `history` is not of type `SEQUOIA_History` or `field` is not a valid field of `SEQUOIA_Solution_step`.
"""
function get_all(history::Any, field::Any)
    validate_history(history)
    validate_field(field)
end