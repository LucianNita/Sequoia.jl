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

include("algorithms/sequoia_feasibility.jl")
include("algorithms/qpm.jl");
include("algorithms/auglag.jl");
include("algorithms/ipm.jl");
include("algorithms/sequoia.jl");

include("residuals/cutest_constraints.jl")
include("residuals/feasibility_residuals.jl")
include("residuals/qpm_residuals.jl")
include("residuals/alm_residuals.jl")
include("residuals/ipm_residuals.jl")
include("residuals/sequoia_residuals.jl")
#include("residuals/residuals.jl")

end # module Sequoia