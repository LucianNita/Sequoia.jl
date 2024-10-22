"""
    validate_history(history)

Validates whether the input `history` is of type `SEQUOIA_History`.

# Arguments
- `history`: The input to validate.

# Throws
- `ArgumentError` if the input is not of type `SEQUOIA_History`.

# Example

```julia
validate_history(history)  # Throws an error if history is not of type `SEQUOIA_History`
"""
function validate_history(history::Any)
    if !(history isa SEQUOIA_History)
        throw(ArgumentError("Invalid input: `history` must be of type `SEQUOIA_History`."))
    end
end

"""
    validate_iterate(iterate)

Validates whether the input `iterate` is of type `SEQUOIA_Solution_step`.

# Arguments
- `iterate`: The input to validate, which should be a `SEQUOIA_Solution_step` instance.

# Throws
- `ArgumentError` if the input is not of type `SEQUOIA_Solution_step`.

# Example

```julia
iterate = SEQUOIA_Solution_step(1, 1e-6, :success, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2])
validate_iterate(iterate)  # This will pass without error

invalid_iterate = [1.0, 2.0]  # Invalid type
validate_iterate(invalid_iterate)  # Throws ArgumentError
"""
function validate_iterate(iterate::Any)
    if !(iterate isa SEQUOIA_Solution_step)
        throw(ArgumentError("Invalid input: `iterate` must be of type `SEQUOIA_Solution_step`."))
    end
end

"""
    validate_field(field::Any)

Just a fallback method. Validates that the input `field` is of type `Symbol`.

# Arguments
- `field::Any`: The input to validate.

# Throws
- `ArgumentError`: If `field` is not of type `Symbol`.

# Example

```julia
validate_field(:x)      # Passes
validate_field("x")     # Throws an ArgumentError: `field` must be of type `Symbol`.
"""

function validate_field(field::Any)
    if !(field isa Symbol)
        throw(ArgumentError("Invalid input: `field` must be of type `Symbol`."))
    end
end

"""
    validate_field(field::Symbol)

Validates whether the input `field` is a valid field of `SEQUOIA_Solution_step`.

# Arguments
- `field::Symbol`: The field name to validate, which should be one of the valid fields in `SEQUOIA_Solution_step`.

# Valid Fields
The following fields are considered valid:
- `:outer_iteration_number`
- `:convergence_metric`
- `:solver_status`
- `:inner_comp_time`
- `:num_inner_iterations`
- `:x`
- `:fval`
- `:gval`
- `:cval`
- `:solver_params`
- `:x_iterates`

# Throws
- `ArgumentError` if the input `field` is not one of the valid fields.

# Example

```julia
validate_field(:x)  # This will pass without error

validate_field(:invalid_field)  # Throws ArgumentError
"""
function validate_field(field::Symbol)
    valid_fields = (:outer_iteration_number, :convergence_metric, :solver_status, :inner_comp_time, 
                    :num_inner_iterations, :x, :fval, :gval, :cval, :solver_params, :x_iterates)
    if !(field in valid_fields)
        throw(ArgumentError("Invalid field: `$field` is not a valid field of `SEQUOIA_Solution_step`. Valid fields are: $valid_fields."))
    end
end
