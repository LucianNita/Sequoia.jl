q=2.5*10^-5
problem = SEQUOIA_pb(
        2;
        x0 = [1.0, -1.0],
        constraints = x -> [-x[1] , -x[2], x[1]+x[2]-1/q],
        eqcon = [3],
        ineqcon = [1,2],
        jacobian = x -> [-1.0 0.0; 0.0 -1.0; 1.0 1.0],
        objective = x -> (q*x[1]-1)^2+(q*x[2]+1)^2,
        gradient = x -> [2*q*(q*x[1]-1); 2*q*(q*x[2]+1)],
        solver_settings = SEQUOIA_Settings(
            :SEQUOIA, 
            :GradientDescent, 
            true, 
            1e-10, 
            200, 
            10.0, 
            1e-10;
            solver_params = [1.0, 4.0, 0.7]
        )
    )
    solve!(problem)

    # Print results
    println("SEQUOIA Solve Completed")
    println("Final Solution: $(problem.solution_history.iterates[end].x)")
    println("Exit Status: $(problem.solution_history.iterates[end].solver_status)")
    println("Objective Value: $(problem.objective(problem.solution_history.iterates[end].x))")