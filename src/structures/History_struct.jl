export SEQUOIA_History, add_iterate!, get_all, clear_history!

"""
    SEQUOIA_History

This struct stores a collection of solution steps (`SEQUOIA_Solution_step`) obtained during the iterative optimization process in the SEQUOIA solver.

# Fields

- `iterates::Vector{SEQUOIA_Solution_step}`: A vector that holds multiple iteration steps, each represented by a `SEQUOIA_Solution_step`.

# Constructor

## Default Constructor
    SEQUOIA_History()

Creates an empty instance of `SEQUOIA_History`.

## With Initial Iterates
    SEQUOIA_History(iterates::Vector{SEQUOIA_Solution_step})

Creates a `SEQUOIA_History` with pre-populated iterates.
"""
struct SEQUOIA_History
    iterates::Vector{SEQUOIA_Solution_step}  # A vector to hold multiple iteration steps

    # Default constructor for an empty history
    SEQUOIA_History() = new(Vector{SEQUOIA_Solution_step}())

    # Constructor for initializing with pre-existing iterates
    SEQUOIA_History(iterates::Vector{SEQUOIA_Solution_step}) = new(iterates)
end


# Methods for SEQUOIA_History
"""
    add_iterate!(history::SEQUOIA_History, iterate::SEQUOIA_Solution_step)

Adds a new `SEQUOIA_Solution_step` to the `SEQUOIA_History` collection.

# Arguments
- `history::SEQUOIA_History`: The history to add the iterate to.
- `iterate::SEQUOIA_Solution_step`: The solution step to add.

# Returns
- The updated `SEQUOIA_History` instance.

# Throws
- `ArgumentError` if inputs are invalid.
"""
function add_iterate!(history::SEQUOIA_History, iterate::SEQUOIA_Solution_step)
    push!(history.iterates, iterate)
end

"""
    get_all(history::SEQUOIA_History, field::Symbol) -> Vector{Any}

Retrieves all values for a specified field from each iterate in the `SEQUOIA_History`.

# Arguments
- `history::SEQUOIA_History`: The history to extract values from.
- `field::Symbol`: The name of the field to retrieve.

# Returns
- A vector of values for the specified field from each iterate.

# Throws
- `ArgumentError` if inputs are invalid.
"""
function get_all(history::SEQUOIA_History, field::Symbol)
    validate_field(field)
    [getfield(iterate, field) for iterate in history.iterates]
end

"""
    clear_history!(history::SEQUOIA_History)

Clears all iterates in the `SEQUOIA_History`.

# Arguments
- `history::SEQUOIA_History`: The history to clear.

# Returns
- The cleared `SEQUOIA_History` instance.
"""
function clear_history!(history::SEQUOIA_History)
    empty!(history.iterates)
end

# Fallback Methods
"""
    add_iterate!(history::Any, iterate::Any)

Fallback method to validate inputs before adding an iterate to a history.

# Arguments
- `history`: Input to validate as `SEQUOIA_History`.
- `iterate`: Input to validate as `SEQUOIA_Solution_step`.

# Throws
- `ArgumentError` if inputs are invalid.
"""
function add_iterate!(history::Any, iterate::Any)
    validate_history(history)
    validate_iterate(iterate)
end

"""
    get_all(history::Any, field::Any)

Fallback method to validate inputs before extracting field values from a history.

# Arguments
- `history`: Input to validate as `SEQUOIA_History`.
- `field`: Input to validate as a valid field.

# Throws
- `ArgumentError` if inputs are invalid.
"""
function get_all(history::Any, field::Any)
    validate_history(history)
    validate_field(field)
end