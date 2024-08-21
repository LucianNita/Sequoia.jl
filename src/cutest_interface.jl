using CUTEst
using NLPModels

"""
    Cutest2Sequoia(problem::String)

Converts a problem from the CUTEst database into a format compatible with the SEQUOIA solver.

# Arguments
- `problem::String`: The name of the optimization problem in the CUTEst database.

# Returns
- A SEQUOIA problem object that encapsulates the optimization problem.

# Example
```julia
sq_pb = Cutest2Sequoia("HS25")
"""

function Cutest2Sequoia(problem::String)
    if !(problem in CUTEst.select())
        error("Problem $problem does not exist in the CUTEst database, make sure you type the name correctly");
        return
    end
    pb=CUTEstModel(problem);

    jl=length(pb.meta.jlow);
    ju=length(pb.meta.jupp);
    jrng=2*length(pb.meta.jrng);

    il=length(pb.meta.ilow);
    iu=length(pb.meta.iupp);
    irng=2*length(pb.meta.irng);

    jfx=length(pb.meta.jfix);
    ifx=length(pb.meta.ifix);

    ce(x)=x->cons(pb,x)[pb.meta.jfix]-lcon[pb.meta.jfix];
    cev(x)=x->x[pb.meta.ifix]-lvar[pb.meta.ifix];
    
    cil(x)=lcon[pb.meta.jlow]-cons(pb,x)[pb.meta.jlow];
    ciu(x)=cons(pb,x)[pb.meta.jupp]-ucon[pb.meta.jupp];
    cirngl(x)=lcon[pb.meta.jrng]-cons(pb,x)[pb.meta.jrng];
    cirngu(x)=cons(pb,x)[pb.meta.jrng]-ucon[pb.meta.jrng];

    cilv(x)=lvar[pb.meta.ilow]-x[pb.meta.ilow];
    ciuv(x)=x[pb.meta.iupp]-uvar[pb.meta.iupp];
    cirnglv(x)=lvar[pb.meta.irng]-x[pb.meta.irng];
    cirnguv(x)=x[pb.meta.irng]-uvar[pb.meta.irng];
     
    sq_pb=SEQUOIA(x->obj(pb,x), "FEAS+MIN", x->[ce(x);cev(x);cil(x);ciu(x);cirngl(x);cirngu(x);cilv(x);ciuv(x);cirnglv(x);cirnguv(x)], collect(1:jfx+ifx), collect(jfx+ifx+1:jfx+ifx+jl+ju+jrng+il+iu+irng), pb.meta.x0);
    finalize(pb);
    return sq_pb;
end