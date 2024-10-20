#using CUTEst

module Sequoia

include("structures/Settings_struct.jl")
include("structures/Solution_struct.jl")
include("structures/Sequoia_struct.jl")

include("checks/verifications.jl");

include("interfaces/cutest_interface.jl");
include("interfaces/solve.jl");

include("algorithms/qpm.jl");
include("algorithms/auglag.jl");
#include("algorithms/ipm.jl");
#include("algorithms/sequoia.jl");

end # module Sequoia