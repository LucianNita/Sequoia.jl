function r0(x,problem::CUTEstModel)
    con = res(x,problem);
    return 0.5*(sum(con[1:problem.meta.jeq+problem.meta.ieq].^2)+sum( (max.(0,con[problem.meta.jeq+problem.meta.ieq+1:end])).^2 ))
end

function r0_gradient!(g,x,problem::CUTEstModel)
    con = res(x,problem);
    con[problem.meta.jeq+problem.meta.ieq+1:end] = max.(0,con[problem.meta.jeq+problem.meta.ieq+1:end]);
    J = dresdx(x,problem);

    g = J'*con; # Update the gradient storage
end

function r0(x,problem::SEQUOIA_pb)
    constraint_val = problem.constraints(x);
    return 0.5 * ( sum( (constraint_val[problem.eqcon]).^2 ) + sum( (constraint_val[problem.ineqcon]).^2 ) )
end


function r0_gradient!(g, x, problem::SEQUOIA_pb)
    constraint_val = problem.constraints(x)
    constraint_val[problem.ineqcon] = max.(0,constraint_val[problem.ineqcon])
    jac = problem.jacobian(x)

    g = jac'*constraint_val; # Update the gradient storage
end