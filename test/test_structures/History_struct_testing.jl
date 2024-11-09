@testset "SEQUOIA_History Unit Tests" begin

    # Test Example 1: Adding Iterates to a History
    @testset "Add Iterates" begin
        history = SEQUOIA_History()  # Create an empty history

        # Create and add iterates
        iterate1 = SEQUOIA_Solution_step(1, 0.01, :first_order, 0.1, 5, [0.5, 1.0], -10.0, [0.0, 0.0])
        iterate2 = SEQUOIA_Solution_step(2, 0.02, :max_iter, 0.2, 10, [0.6, 1.1], -9.5, [0.01, -0.01])
        iterate3 = SEQUOIA_Solution_step(3, 0.03, :acceptable, 0.3, 15, [0.7, 1.2], -9.0, [0.02, -0.02])

        add_iterate!(history, iterate1)
        add_iterate!(history, iterate2)
        add_iterate!(history, iterate3)

        @test length(history.iterates) == 3
        @test history.iterates[1] == iterate1
        @test history.iterates[2] == iterate2
        @test history.iterates[3] == iterate3
    end

    # Test Example 2: Retrieving All Field Values
    @testset "Retrieve Field Values" begin
        history = SEQUOIA_History([
            SEQUOIA_Solution_step(1, 0.01, :first_order, 0.1, 5, [0.5, 1.0], -10.0, [0.0, 0.0]),
            SEQUOIA_Solution_step(2, 0.02, :max_iter, 0.2, 10, [0.6, 1.1], -9.5, [0.01, -0.01]),
            SEQUOIA_Solution_step(3, 0.03, :acceptable, 0.3, 15, [0.7, 1.2], -9.0, [0.02, -0.02])
        ])

        convergence_metrics = get_all(history, :convergence_metric)
        @test convergence_metrics == [0.01, 0.02, 0.03]

        solver_statuses = get_all(history, :solver_status)
        @test solver_statuses == [:first_order, :max_iter, :acceptable]
    end

    # Test Example 3: Clearing the History
    @testset "Clear History" begin
        history = SEQUOIA_History([
            SEQUOIA_Solution_step(1, 0.01, :first_order, 0.1, 5, [0.5, 1.0], -10.0, [0.0, 0.0])
        ])
        
        clear_history!(history)
        @test length(history.iterates) == 0
    end

    # Test Example 4: Handling Invalid Inputs for Adding Iterates
    @testset "Invalid Iterates" begin
        history = SEQUOIA_History()

        # Invalid iterate type
        @test_throws ArgumentError add_iterate!(history, "invalid_iterate")

        # Invalid history type
        @test_throws ArgumentError add_iterate!("invalid_history", SEQUOIA_Solution_step(1, 0.01, :first_order, 0.1, 5, [0.5, 1.0], -10.0, [0.0, 0.0]))
    end

    # Test Example 5: Handling Invalid Inputs for Retrieving Fields
    @testset "Invalid Fields" begin
        history = SEQUOIA_History([
            SEQUOIA_Solution_step(1, 0.01, :first_order, 0.1, 5, [0.5, 1.0], -10.0, [0.0, 0.0])
        ])

        # Invalid history type
        @test_throws ArgumentError get_all("invalid_history", :x)

        # Invalid field name
        @test_throws ArgumentError get_all(history, :invalid_field)

        # Invalid field type
        @test_throws ArgumentError get_all(history, "invalid_field")
    end
end
