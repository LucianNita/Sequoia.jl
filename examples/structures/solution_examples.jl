"""
# Example 1: Using the Full Constructor

This example shows how to use the full constructor for `SEQUOIA_Solution_step`, where all fields are specified, including `x_iterates` and `solver_params`.

    ```julia
    example_full_constructor()
    
    solution_full = SEQUOIA_Solution_step(
        10,                       # outer_iteration_number
        1e-6,                     # convergence_metric
        :success,                 # solver_status
        0.02,                     # inner_comp_time
        5,                        # num_inner_iterations
        [1.0, 2.0],               # x
        0.5,                      # fval
        [0.1, 0.2],               # gval
        nothing,                  # cval
        [0.5, 0.3],               # solver_params
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]]  # x_iterates
    )
    
Example output:

SEQUOIA_Solution_step(10, 1.0e-6, :success, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]])

"""
function example_full_constructor()
    solution_full = SEQUOIA_Solution_step(
        10,                       # outer_iteration_number
        1e-6,                     # convergence_metric
        :success,                 # solver_status
        0.02,                     # inner_comp_time
        5,                        # num_inner_iterations
        [1.0, 2.0],               # x
        0.5,                      # fval
        [0.1, 0.2],               # gval
        nothing,                  # cval
        [0.5, 0.3],               # solver_params
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]]  # x_iterates
    )
    println(solution_full)
end

"""
# Example 2: Using the Intermediate Constructor

This example demonstrates the intermediate constructor for `SEQUOIA_Solution_step`, specifying `cval` and `solver_params`, while `x_iterates` is omitted.

    ```julia
    example_intermediate_constructor()
    
    solution_intermediate = SEQUOIA_Solution_step(
        10,                       # outer_iteration_number
        1e-6,                     # convergence_metric
        :failed,                  # solver_status
        0.03,                     # inner_comp_time
        6,                        # num_inner_iterations
        [1.1, 2.1],               # x
        0.45,                     # fval
        [0.05, 0.15],             # gval
        [0.1, 0.2],               # cval
        [0.01, 0.02]              # solver_params
    )

Example output:

SEQUOIA_Solution_step(10, 1.0e-6, :failed, 0.03, 6, [1.1, 2.1], 0.45, [0.05, 0.15], [0.1, 0.2], [0.01, 0.02], nothing)
    
"""
function example_intermediate_constructor()
    solution_intermediate = SEQUOIA_Solution_step(
        10,                       # outer_iteration_number
        1e-6,                     # convergence_metric
        :failed,                  # solver_status
        0.03,                     # inner_comp_time
        6,                        # num_inner_iterations
        [1.1, 2.1],               # x
        0.45,                     # fval
        [0.05, 0.15],             # gval
        [0.1, 0.2],               # cval
        [0.01, 0.02]              # solver_params
    )
    println(solution_intermediate)
end

"""
# Example 3: Using the Minimal Constructor

This example shows the minimal constructor for `SEQUOIA_Solution_step`, which only includes the mandatory fields (`x`, `fval`, `gval`), and omits optional parameters like `cval`, `solver_params`, and `x_iterates`.

```julia
example_minimal_constructor()

solution_minimal = SEQUOIA_Solution_step(
    0,                        # outer_iteration_number
    0.0,                      # convergence_metric
    :success,                 # solver_status
    0.0,                      # inner_comp_time
    0,                        # num_inner_iterations
    [1.0, 2.0],               # x
    0.5,                      # fval
    [0.1, 0.2]                # gval
)

Example output:

SEQUOIA_Solution_step(0, 0.0, :success, 0.0, 0, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, nothing, nothing)

"""
function example_minimal_constructor()
    solution_minimal = SEQUOIA_Solution_step(
        0,                        # outer_iteration_number
        0.0,                      # convergence_metric
        :success,                 # solver_status
        0.0,                      # inner_comp_time
        0,                        # num_inner_iterations
        [1.0, 2.0],               # x
        0.5,                      # fval
        [0.1, 0.2]                # gval
    )
    println(solution_minimal)
end

"""
# Example 4: Error Handling with Invalid Inputs

This example shows how the validation mechanism in `SEQUOIA_Solution_step` works when invalid inputs are provided. An error is raised when the solution vector `x` is empty.

```julia
example_invalid_inputs()

try
    SEQUOIA_Solution_step(
        10,                       # outer_iteration_number
        1e-6,                     # convergence_metric
        :success,                 # solver_status
        0.02,                     # inner_comp_time
        5,                        # num_inner_iterations
        Float64[],                # Invalid: empty solution vector `x`
        0.5,                      # fval
        [0.1, 0.2],               # gval
        nothing,                  # cval
        [0.5, 0.3],               # solver_params
        [[1.0, 2.0]]              # x_iterates
    )
catch e
    println(e)  # Prints the error message
end

Example output:

ArgumentError("Solution vector `x` cannot be empty.")

"""
function example_invalid_inputs()
    try
        SEQUOIA_Solution_step(
            10,                       # outer_iteration_number
            1e-6,                     # convergence_metric
            :success,                 # solver_status
            0.02,                     # inner_comp_time
            5,                        # num_inner_iterations
            Float64[],                # Invalid: empty solution vector `x`
            0.5,                      # fval
            [0.1, 0.2],               # gval
            nothing,                  # cval
            [0.5, 0.3],               # solver_params
            [[1.0, 2.0]]              # x_iterates
        )
    catch e
        println(e)  # Prints the error message
    end
end

"""
# Example 5: Handling Mismatched Dimensions in `x_iterates`

This example demonstrates what happens when the number of `x_iterates` vectors doesn't match the number of inner iterations.

```julia
example_invalid_x_iterates()

try
    SEQUOIA_Solution_step(
        10,                       # outer_iteration_number
        1e-6,                     # convergence_metric
        :success,                 # solver_status
        0.02,                     # inner_comp_time
        5,                        # num_inner_iterations
        [1.0, 2.0],               # x
        0.5,                      # fval
        [0.1, 0.2],               # gval
        nothing,                  # cval
        [0.5, 0.3],               # solver_params
        [[1.0, 2.0], [1.1, 2.1]]  # Invalid: fewer `x_iterates` than `num_inner_iterations`
    )
catch e
    println(e)  # Prints the error message
end

Example output:

ArgumentError("`x_iterates` must contain exactly `num_inner_iterations` vectors.")

"""
function example_invalid_x_iterates()
    try
        SEQUOIA_Solution_step(
            10,                       # outer_iteration_number
            1e-6,                     # convergence_metric
            :success,                 # solver_status
            0.02,                     # inner_comp_time
            5,                        # num_inner_iterations
            [1.0, 2.0],               # x
            0.5,                      # fval
            [0.1, 0.2],               # gval
            nothing,                  # cval
            [0.5, 0.3],               # solver_params
            [[1.0, 2.0], [1.1, 2.1]]  # Invalid: fewer `x_iterates` than `num_inner_iterations`
        )
    catch e
        println(e)  # Prints the error message
    end
end
