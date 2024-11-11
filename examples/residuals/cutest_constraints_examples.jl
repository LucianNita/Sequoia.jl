# Example Use Cases for `res` and `dresdx` functions 
using CUTEst
using Sequoia


# Example 1: Basic constraint computation
"""
This example demonstrates how to compute the constraint vector `res` for a CUTEst problem.

# Usage:
    example_res()

# Expected Output:
    Constraint vector:
    [399.99208149095404, 399.99208149095404, 799.992081490954, -0.48, -0.48, -0.48, -0.48, 0.0, 0.0, -0.48, -0.48, -1200.0, -1200.0]
"""
function example_res()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS75")  # Example with constraints
    x = problem.meta.x0  # Use the initial guess for x

    # Compute the constraint vector
    violation = res(x, problem)

    println("Constraint vector:")
    println(violation)

    finalize(problem)  # Finalize CUTEst environment
end

# Example 2: Jacobian of the constraint vector
"""
This example demonstrates how to compute the Jacobian of the constraint vector `dresdx` for a CUTEst problem.

# Usage:
    example_dresdx()

# Expected Output:
    Jacobian of the constraint vector:
    sparse([1, 2, 3, 4, 5, 6, 10, 1, 2, 3, 4, 5, 7, 11, 2, 8, 12, 1, 9, 13], [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4], [-968.9124217106447, -968.9124217106447, 1937.8248434212894, -1.0, 1.0, -1.0, 1.0, -968.9124217106447, 1937.8248434212894, -968.9124217106447, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0], 13, 4)
"""
function example_dresdx()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS75")  # Example with constraints
    x = problem.meta.x0  # Use the initial guess for x

    # Compute the Jacobian of the constraint vector
    jacobian = dresdx(x, problem)

    println("Jacobian of the constraint vector:")
    println(jacobian)

    finalize(problem)  # Finalize CUTEst environment
end

# Example 3: Combined use of `res` and `dresdx` for analysis
"""
This example demonstrates how to compute both the constraint vector and its Jacobian for a CUTEst problem.

# Usage:
    example_combined()

# Expected Output:
    Constraint vector:
    [-0.2599999999999998, -0.0017999999999999989, 21000.010000000013]
    Jacobian of the constraint vector:
    sparse([1, 2, 3, 1, 2, 3, 1, 2, 3], [1, 1, 1, 2, 2, 2, 3, 3, 3], [2.6, 0.018000000000000002, -220000.00000000003, 0.4, 0.002, 0.2, -1.0, -1.0, -1.0], 3, 3)
"""
function example_combined()
    # Initialize a CUTEst problem
    problem = CUTEstModel("POLAK4")  # Example with constraints
    x = problem.meta.x0  # Use the initial guess for x

    # Compute the constraint vector
    violation = res(x, problem)
    println("Constraint vector:")
    println(violation)

    # Compute the Jacobian of the constraint vector
    jacobian = dresdx(x, problem)
    println("Jacobian of the constraint vector:")
    println(jacobian)

    finalize(problem)  # Finalize CUTEst environment
end

# Example 4: Testing with another CUTEst problem
"""
This example demonstrates how to use `res` and `dresdx` with another CUTEst problem.

# Usage:
    example_hs85_problem()

# Expected Output:
    Constraint vector:
    [-5.0, -258.6706235292108, -0.5573003795008624, -24.853941620120224, -23.5, -1.4860514541387033, -2.3392912070690084, -44.44353293431118, -420.17198439862256, -199.12319145999732, -1.2741792466029973, -0.03379726428285129, -32.22231262621746, -327.67916256637, -503.1447560137914, -5.112565930634773, -2142.349636440344, -7404.559334699463, -0.20487342889801852, -64304.036640653285, -8.740314011661464e6, -168.63, -1034.6756485458613, -21.415708792930992, -406.91346706568885, -156.83301560137744, -65.83180854000267, -4.159820753397003, -0.04220273571714871, -133.15368737378253, -35.73283743362998, -14.069243986208676, -513.2624340693652, -32.52636355965615, -10478.078665300536, -0.11812657110198149, -4611.633359346713, -603080.9883385357, -195.5852, -11.400000000000006, -115.0, -74.0, -2.0, -6.385499999999979, -208.88, -19.75, -20.096600000000024, -57.198800000000006]
    Jacobian of the constraint vector:
    sparse([2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 44, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 40, 45, 1, 4, 5, 11, 13, 18, 19, 20, 22, 28, 30, 35, 36, 37, 41, 46, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 42, 47, 4, 18, 19, 20, 35, 36, 37, 43, 48], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5], [-0.17205792737623615, -0.00975386206747548, -0.1514481869144151, -0.00905565717565526, -0.17205748633744994, -0.20738987409082477, -0.61149698239607, -0.0015957317143961678, -9.940761548871336e-5, 0.06806540327655945, -1.550378550219206, -1.6374822973373526, -0.022159451280910215, -1.5496033609440965, -0.356180272677649, -9.543251464218627e-5, 13.637889897126865, 2847.5385295046144, 0.00905565717565526, 0.17205748633744994, 0.20738987409082477, 0.61149698239607, 0.0015957317143961678, 9.940761548871336e-5, -0.06806540327655945, 1.550378550219206, 1.6374822973373526, 0.022159451280910215, 1.5496033609440965, 0.356180272677649, 9.543251464218627e-5, -13.637889897126865, -2847.5385295046144, -1.0, 1.0, -1.5, -8.491243313084913e-6, 0.010550430920771368, 0.851409358681709, -1.0, -3.9928413024838045, 3.9928413024838045, -0.004592974849401471, -0.001046044522372666, 0.06678113571717786, 1.6769933470431981, -0.004729360428121805, 1.6761548503696766, -64.74945981342124, 0.001297447757829252, -644.2082469742447, -2139.575689573954, 1.0, 3.9928413024838045, -3.9928413024838045, 0.004592974849401471, 0.001046044522372666, -0.06678113571717786, -1.6769933470431981, 0.004729360428121805, -1.6761548503696766, 64.74945981342124, -0.001297447757829252, 644.2082469742447, 2139.575689573954, -1.0, 1.0, 1.0, 2.445065393646984, -1.0, 0.01219855979122146, -0.6494561657259439, -55.00145296927799, -0.00013763342309371657, 72.544550085629, 1.0, -0.01219855979122146, 0.6494561657259439, 55.00145296927799, 0.00013763342309371657, -72.544550085629, -1.0, 1.0, 0.5304841524579755, -0.10136005183410524, 0.05174318407553205, 0.0938396168340765, 0.027920263415825406, 0.5304850049006827, -0.40084453169574275, -0.15756073662076542, 0.0005451913391332007, 3.396320980487501e-5, -0.023254954467725727, -0.055230766121717956, -0.22358546943392987, 0.045435889940080715, -0.21942248019829097, 0.12169113271489818, 0.0010547349826098013, -561.1267485821348, -641.2170982147003, -0.0938396168340765, -0.027920263415825406, -0.5304850049006827, 0.40084453169574275, 0.15756073662076542, -0.0005451913391332007, -3.396320980487501e-5, 0.023254954467725727, 0.055230766121717956, 0.22358546943392987, -0.045435889940080715, 0.21942248019829097, -0.12169113271489818, -0.0010547349826098013, 561.1267485821348, 641.2170982147003, -1.0, 1.0, 2.2250391479950467, 5.232912104119, 1.309462871900374e-5, -6.901985924680899, -5.232912104119, -1.309462871900374e-5, 6.901985924680899, -1.0, 1.0], 48, 5)
"""
function example_hs85_problem()
    # Initialize another CUTEst problem
    problem = CUTEstModel("HS85")  # Another example with constraints
    x = problem.meta.x0  # Use the initial guess for x

    # Compute the constraint vector
    violation = res(x, problem)
    println("Constraint vector:")
    println(violation)

    # Compute the Jacobian of the constraint vector
    jacobian = dresdx(x, problem)
    println("Jacobian of the constraint vector:")
    println(jacobian)

    finalize(problem)  # Finalize CUTEst environment
end

# Example 5: Evaluate `res` and `dresdx` at a custom point
"""
This example demonstrates how to evaluate `res` and `dresdx` at a custom point for a CUTEst problem.

# Usage:
    example_custom_point()

# Expected Output:
    Constraint vector at custom point:
    [4.0, 2.0, -1.0, -1.0000000003e10, 2.0]
    Jacobian of the constraint vector at custom point:
    sparse([1, 1, 3, 4, 5, 2], [1, 2, 2, 3, 3, 4], [2.0, 4.0, -1.0, -1.0, 1.0, 1.0], 5, 4)
"""
function example_custom_point()
    # Initialize a CUTEst problem
    problem = CUTEstModel("ALLINITC")  # Example with constraints
    x = [1.0,2.0,3.0,4.0]  # Use a different point for testing 

    # Compute the constraint vector
    violation = res(x, problem)
    println("Constraint vector at custom point:")
    println(violation)

    # Compute the Jacobian of the constraint vector
    jacobian = dresdx(x, problem)
    println("Jacobian of the constraint vector at custom point:")
    println(jacobian)

    finalize(problem)  # Finalize CUTEst environment
end

# Example 6: Testing with another CUTEst problem and custom point
"""
This example demonstrates how to use `res` and `dresdx` with another CUTEst problem.

# Usage:
    example_hs15_problem()

# Expected Output:
    Constraint vector:
    [-3.0, -6.0, 1.5]
    Jacobian of the constraint vector:
    sparse([1, 2, 3, 1, 2], [1, 1, 1, 2, 2], [-2.0, -1.0, 1.0, -2.0, -4.0], 3, 2)
"""
function example_hs15_problem()
    # Initialize another CUTEst problem
    problem = CUTEstModel("HS15")  # Another example with constraints
    x = 2*ones(problem.meta.nvar)  # Use a custom initial guess for x

    # Compute the constraint vector
    violation = res(x, problem)
    println("Constraint vector:")
    println(violation)

    # Compute the Jacobian of the constraint vector
    jacobian = dresdx(x, problem)
    println("Jacobian of the constraint vector:")
    println(jacobian)

    finalize(problem)  # Finalize CUTEst environment
end

# Call examples
example_res()
example_dresdx()
example_combined()
example_hs85_problem()
example_custom_point()
example_hs15_problem()
