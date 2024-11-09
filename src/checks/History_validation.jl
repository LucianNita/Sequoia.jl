"""
    validate_history(history::Any)

Validates that the input is a valid `SEQUOIA_History` instance.

# Arguments
- `history`: Input to validate.

# Throws
- `ArgumentError` if the input is not of type `SEQUOIA_History`.
"""
function validate_history(history::Any)
    if !(history isa SEQUOIA_History)
        throw(ArgumentError("Expected `SEQUOIA_History`, but got `$(typeof(history))`."))
    end
end

"""
    validate_iterate(iterate::Any)

Validates that the input is a valid `SEQUOIA_Solution_step` instance.

# Arguments
- `iterate`: Input to validate.

# Throws
- `ArgumentError` if the input is not of type `SEQUOIA_Solution_step`.
"""
function validate_iterate(iterate::Any)
    if !(iterate isa SEQUOIA_Solution_step)
        throw(ArgumentError("Expected `SEQUOIA_Solution_step`, but got `$(typeof(iterate))`."))
    end
end

"""
    validate_field(field::Any)

Validates that the input `field` is of type `Symbol`.

# Arguments
- `field::Any`: The input to validate.

# Throws
- `ArgumentError`: If `field` is not of type `Symbol`.
"""
function validate_field(field::Any)
    if !(field isa Symbol)
        throw(ArgumentError("Invalid input: `field` must be of type `Symbol`."))
    end
end

"""
    validate_field(field::Symbol)

Validates that the field exists in the `SEQUOIA_Solution_step` struct.

# Arguments
- `field::Symbol`: The field to validate.

# Throws
- `ArgumentError` if the field is not a valid field of `SEQUOIA_Solution_step`.
"""
function validate_field(field::Symbol)
    if !(field in fieldnames(SEQUOIA_Solution_step))
        throw(ArgumentError("Invalid field: `$field`. Valid fields are: $(fieldnames(SEQUOIA_Solution_step))"))
    end
end