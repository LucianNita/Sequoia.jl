@testset "Testing SEQUOIA struct initialization" begin
    
    # Test case 1: Basic initialization
    nvar = 2
    objective = x -> sum(x.^2)
    sense = "MIN"
    constraints = x -> [x[1] + x[2] - 1]
    eqcon = [1]
    ineqcon = Int[]
    x0 = [0.5, 0.5]
    t = objective(x0)

    seq_obj = SEQUOIA(nvar; obj=objective, sense=sense, cons=constraints, eqcon=eqcon, ineqcon=ineqcon, x0=x0, t=t)

    @test seq_obj.nvar == nvar
    @test seq_obj.objective == objective
    @test seq_obj.objSense == sense
    @test seq_obj.constraints == constraints
    @test seq_obj.eqcon == eqcon
    @test seq_obj.ineqcon == ineqcon
    @test seq_obj.x0 == x0
    @test seq_obj.t == t

    # Test case 2: Default initialization
    seq_obj_default = SEQUOIA(nvar)

    @test seq_obj_default.nvar == nvar
    @test seq_obj_default.objSense == "FEAS+MIN"
    @test seq_obj_default.x0 == zeros(nvar)
    @test seq_obj_default.t == 0.0

    # Test case 3: Initialization with only mandatory fields
    seq_obj_mandatory = SEQUOIA(nvar, obj=objective)

    @test seq_obj_mandatory.nvar == nvar
    @test seq_obj_mandatory.objective == objective
    @test seq_obj_mandatory.t == objective(zeros(nvar))

    # Test case 4: Initialization with custom penalty parameter
    custom_t = 10.0
    seq_obj_custom_t = SEQUOIA(nvar, obj=objective, x0=x0, t=custom_t)

    @test seq_obj_custom_t.t == custom_t

    # Test case 5: Checking constraints handling
    constraints_2 = x -> [x[1]^2 + x[2]^2 - 1, x[1] - 0.5]
    eqcon_2 = [1]
    ineqcon_2 = [2]

    seq_obj_constraints = SEQUOIA(nvar, cons=constraints_2, eqcon=eqcon_2, ineqcon=ineqcon_2)

    @test seq_obj_constraints.constraints == constraints_2
    @test seq_obj_constraints.eqcon == eqcon_2
    @test seq_obj_constraints.ineqcon == ineqcon_2

    # Test case 6: Edge case with zero variables
    seq_obj_zero_vars = SEQUOIA(0)

    @test seq_obj_zero_vars.nvar == 0
    @test length(seq_obj_zero_vars.x0) == 0
    @test seq_obj_zero_vars.t == 0.0

end