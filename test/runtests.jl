using Sequoia
using Test

include("test_structures/Settings_struct_testing.jl");
include("test_structures/Solution_struct_testing.jl");
include("test_structures/Sequoia_struct_testing.jl");

#include("test_checks/verifications_testing.jl");

#include("test_interfaces/cutest_interface_testing.jl");
#include("test_interfaces/solve.jl");

#include("test_algorithms/qpm_testing.jl");
#include("test_algorithms/auglag_testing.jl");
#include("test_algorithms/ipm_testing.jl");
#include("test_algorithms/sequoia_testing.jl");