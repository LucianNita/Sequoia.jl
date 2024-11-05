module Sequoia

include("structures/Settings_struct.jl")
include("structures/Solution_struct.jl")
include("structures/History_struct.jl")
include("structures/Sequoia_struct.jl")

include("checks/Settings_validation.jl");
include("checks/Solution_validation.jl");
include("checks/History_validation.jl");
include("checks/Sequoia_validation.jl");

include("interfaces/cutest_interface.jl");
include("interfaces/solve.jl");

include("checks/cutest_check.jl");

include("algorithms/sequoia_feasibility.jl")
include("algorithms/qpm.jl");
include("algorithms/auglag.jl");
include("algorithms/ipm.jl");
include("algorithms/sequoia.jl");

include("algorithms/residuals.jl")

end # module Sequoia