@testset "SEQUOIA_History Tests" begin

    @testset "SEQUOIA_History - Valid Calls" begin
        # Create a mock SEQUOIA_Solution_step for testing with correct x_iterates length
        solution_step1 = SEQUOIA_Solution_step(
            1, 1e-6, :success, 0.02, 2, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1]]
        )  # Now, num_inner_iterations = 2, matching x_iterates length

        solution_step2 = SEQUOIA_Solution_step(
            2, 1e-6, :success, 0.03, 2, [1.1, 2.1], 0.45, [0.05, 0.15], [0.1, 0.2], nothing, [[1.0, 2.0], [1.1, 2.1]]
        )  # Same adjustment here, num_inner_iterations = 2

        # Test SEQUOIA_History initialization
        history = SEQUOIA_History()
        @test history.iterates == []  # Initially empty

        # Test add_iterate! with valid SEQUOIA_Solution_step
        add_iterate!(history, solution_step1)
        @test length(history.iterates) == 1
        @test history.iterates[1] == solution_step1

        add_iterate!(history, solution_step2)
        @test length(history.iterates) == 2
        @test history.iterates[2] == solution_step2

        # Test get_all with valid field names
        all_x = get_all(history, :x)
        @test all_x == [[1.0, 2.0], [1.1, 2.1]]

        all_fvals = get_all(history, :fval)
        @test all_fvals == [0.5, 0.45]

        all_convergence_metrics = get_all(history, :convergence_metric)
        @test all_convergence_metrics == [1e-6, 1e-6]
    end

    @testset "SEQUOIA_History functions - Invalid Calls" begin
        # Create a mock SEQUOIA_Solution_step for testing with correct x_iterates length
        solution_step1 = SEQUOIA_Solution_step(
            1, 1e-6, :success, 0.02, 2, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1]]
        )  # Now, num_inner_iterations = 2, matching x_iterates length

        solution_step2 = SEQUOIA_Solution_step(
            2, 1e-6, :success, 0.03, 2, [1.1, 2.1], 0.45, [0.05, 0.15], [0.1, 0.2], nothing, [[1.0, 2.0], [1.1, 2.1]]
        )  # Same adjustment here, num_inner_iterations = 2

        # Test SEQUOIA_History initialization
        history = SEQUOIA_History()

        # Test add_iterate! validation
        @test_throws ArgumentError add_iterate!(history, [1.0, 2.0])  # Invalid iterate type
        @test_throws ArgumentError add_iterate!(0.5, solution_step1)  # Invalid history type

        # Test get_all validation
        @test_throws ArgumentError get_all(history, :invalid_field)  # Invalid field name
        @test_throws ArgumentError get_all([1, 2, 3], :x)            # Invalid history type

        # Test fallback methods
        @test_throws ArgumentError add_iterate!(0.5, [1.0, 2.0])  # Invalid both history and iterate
        @test_throws ArgumentError get_all(0.5, "invalid_field")  # Invalid both history and field

        # Attempt to retrieve values with a non-Symbol field (String in this case)
        @test_throws ArgumentError get_all(history, "fval")  # Invalid field type (should be a Symbol)
    end

    # Example 1: Using the Full Constructor
    @testset "SEQUOIA_History Full Constructor" begin
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
        @test all_fvals == [0.5, 0.45]
    end

    # Example 2: Testing Validation for `add_iterate!`
    @testset "Validation for `add_iterate!`" begin
        history = SEQUOIA_History()
        step = SEQUOIA_Solution_step(
            1, 1e-6, :success, 0.02, 2, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1]]
        )
        add_iterate!(history, step)

        @test_throws ArgumentError add_iterate!([1, 2, 3], step)  # Invalid history type
        @test_throws ArgumentError add_iterate!(history, [1.0, 2.0])  # Invalid iterate type
    end

    # Example 3: Retrieving All Field Values with `get_all`
    @testset "Retrieve All Field Values" begin
        history = SEQUOIA_History()
        step1 = SEQUOIA_Solution_step(
            1, 1e-6, :success, 0.02, 2, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1]]
        )
        step2 = SEQUOIA_Solution_step(
            2, 1e-6, :success, 0.03, 2, [1.1, 2.1], 0.45, [0.05, 0.15], nothing, [0.1, 0.2], [[1.1, 2.1], [1.2, 2.2]]
        )
        add_iterate!(history, step1)
        add_iterate!(history, step2)

        all_x_values = get_all(history, :x)
        @test all_x_values == [[1.0, 2.0], [1.1, 2.1]]

        all_gvals = get_all(history, :gval)
        @test all_gvals == [[0.1, 0.2], [0.05, 0.15]]
    end

    # Example 4: Handling Invalid Field Access
    @testset "Invalid Field Access" begin
        history = SEQUOIA_History()
        step = SEQUOIA_Solution_step(
            1, 1e-6, :success, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]]
        )
        add_iterate!(history, step)

        @test_throws ArgumentError get_all(history, :invalid_field)  # Invalid field name
    end

    # Example 5: Retrieving Convergence Metrics
    @testset "Retrieve Convergence Metrics" begin
        history = SEQUOIA_History()
        step1 = SEQUOIA_Solution_step(
            1, 1e-6, :success, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, [0.5, 0.3], [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]]
        )
        step2 = SEQUOIA_Solution_step(
            2, 1e-5, :success, 0.03, 6, [1.1, 2.1], 0.45, [0.05, 0.15], nothing, [0.1, 0.2], [[1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4], [1.5, 2.5], [1.6, 2.6]]
        )
        add_iterate!(history, step1)
        add_iterate!(history, step2)

        all_convergence_metrics = get_all(history, :convergence_metric)
        @test all_convergence_metrics == [1.0e-6, 1.0e-5]
    end

end
