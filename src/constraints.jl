function resid(c::Function, x::Vector{Float64})  ##this breaks have to call into lib, so no finalize.  ##Not testing the actual stuff, example: sq_pb.objective(sq_pb.x0)=sq_pb.t
    return c(x);
end


qp_resid(c::Function, eq::Vector{Int}, ineq::Vector{Int}, x::Vector{Float64}, pk::Float64) = pk*(sum(c[1](x)[eq].^2)+sum(max.(0,c[1](x)[ineq]).^2));

lp_resid(c::Function, eq::Vector{Int}, ineq::Vector{Int}, x::Vector{Float64}, pk::Float64) = pk*(sum(abs(c[1](x)[eq]))+sum(max.(0,c[1](x)[ineq])));

up_resid(c::Function, eq::Vector{Int}, ineq::Vector{Int}, x::Vector{Float64}, pk::Float64, tol) = pk*(float(lp_resid(c,eq,ineq,x,1.0)≥tol));

lagp_resid(c::Function, eq::Vector{Int}, ineq::Vector{Int}, x::Vector{Float64}, pk::Float64, λk::Vector{Float64}) = sum(λk.*c[1](x)[eq]) + pk*(sum(c[1](x)[eq].^2)); # Eq only

barr_resid(c::Function, eq::Vector{Int}, ineq::Vector{Int}, x::Vector{Float64}, pk::Float64) = pk*(sum(log(-c[1](x)[ineq]))); #Ineq only

