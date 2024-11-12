using Sequoia
using Test
#
include("test_structures/Settings_struct_testing.jl");
include("test_structures/Solution_struct_testing.jl");
include("test_structures/History_struct_testing.jl");
include("test_structures/Sequoia_struct_testing.jl");

include("test_checks/Settings_validation_testing.jl");
include("test_checks/Solution_validation_testing.jl");
include("test_checks/Sequoia_validation_testing.jl");

include("test_interfaces/cutest_interface_testing.jl");#
#include("test_interfaces/solve_testing.jl");

include("test_algorithms/sequoia_feasibility_testing.jl");
include("test_algorithms/qpm_testing.jl");
#include("test_algorithms/auglag_testing.jl");
#include("test_algorithms/ipm_testing.jl");
include("test_algorithms/sequoia_testing.jl");
#
include("test_residuals/cutest_constraints_testing.jl")
include("test_residuals/feasibility_residuals_testing.jl")
include("test_residuals/qpm_residuals_testing.jl")
include("test_residuals/alm_residuals_testing.jl")
include("test_residuals/ipm_residuals_testing.jl")
include("test_residuals/sequoia_residuals_testing.jl")#