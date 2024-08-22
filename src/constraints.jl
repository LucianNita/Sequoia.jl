function resid(c::Function, x::Vector{Float64})  ##this breaks have to call into lib, so no finalize.  ##Not testing the actual stuff, example: sq_pb.objective(sq_pb.x0)=sq_pb.t
    return c(x);
end