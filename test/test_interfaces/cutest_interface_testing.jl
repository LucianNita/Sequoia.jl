# Unit test set for CUTEst to SEQUOIA_pb conversion using actual CUTEst problems
@testset "CUTEst to SEQUOIA_pb Conversion with Real Problems" begin
    # Load a list of real CUTEst problems
    problems = ["ROSENBR","HS21","INVALID"]
    
    # Iterate over each loaded problem
    for i in eachindex(problems)
        if i==length(problems)
            @test_throws ErrorException cutest_problem=CUTEstModel(problems[i])
            continue
        end
        cutest_problem=CUTEstModel(problems[i])

        @testset "Testing conversion for problem: $(cutest_problem.meta.name)" begin
            # Convert the CUTEst problem to a SEQUOIA_pb instance
            sequoia_pb = cutest_to_sequoia(cutest_problem)
            
            # Basic checks
            @test sequoia_pb.nvar == cutest_problem.meta.nvar
            @test sequoia_pb.x0 == cutest_problem.meta.x0
            @test sequoia_pb.is_minimization == cutest_problem.meta.minimize

            # Objective function check: validate evaluation at the initial point
            x0 = cutest_problem.meta.x0
            @test sequoia_pb.objective(x0) ≈ obj(cutest_problem, x0)

            # Gradient check: validate evaluation at the initial point
            @test sequoia_pb.gradient(x0) ≈ grad(cutest_problem, x0)

            # Check if the problem has constraints and validate constraints if present
            if cutest_problem.meta.ncon > 0
                # Constraint check: validate evaluation at the initial point
                @test sequoia_pb.constraints(x0) ≈ cons(cutest_problem, x0)
                
                # Jacobian check: validate evaluation at the initial point
                @test sequoia_pb.jacobian(x0) ≈ jac(cutest_problem, x0)

                # Validate equality and inequality indices
                @test sequoia_pb.eqcon == cutest_problem.meta.jfix
                @test sequoia_pb.ineqcon == sort(vcat(cutest_problem.meta.jlow, cutest_problem.meta.jupp, cutest_problem.meta.jrng))
            end

            # Ensure the original CUTEst problem reference is preserved in the SEQUOIA_pb instance
            @test sequoia_pb.cutest_nlp === cutest_problem
        end

        finalize(cutest_problem)
    end
end
