using CUTEst

module Sequoia

include("Settings_struct.jl")
include("Solution_struct.jl")
include("Sequoia_struct.jl")

include("verifications.jl");

include("algorithms/qpm.jl");
include("algorithms/auglag.jl");

end # module Sequoia