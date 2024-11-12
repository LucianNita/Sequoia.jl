export ipm_obj, ipm_grad!, r_slack

function r_slack(x,s,problem::SEQUOIA_pb)
    constraint_val = problem.constraints(x)
    constraint_val[problem.ineqcon] .= constraint_val[problem.ineqcon] .+ s.^2 
    return sum(constraint_val.^2)
end

function r_slack(x,s,problem::CUTEstModel)
    cons=res(x,problem);
    eqcon=length(problem.meta.jfix)+length(problem.meta.ifix);
    cons[eqcon+1:end] .= cons[eqcon+1:end] .+ s.^2
    return sum(cons.^2)
end

function ipm_obj(x_a, μ, problem::SEQUOIA_pb)
    
    x=x_a[1:problem.nvar];
    iq=length(problem.ineqcon);
    eq=length(problem.eqcon);
    λ=x_a[problem.nvar+1:problem.nvar+iq+eq];
    ν=λ[problem.ineqcon];
    s=x_a[problem.nvar+iq+eq+1:end];
    jac=problem.jacobian(x);

    return norm(problem.gradient(x)+jac'*λ)^2+ r_slack(x,s,problem) + norm(2*ν.*s.^2 .-μ)^2 
end

function ipm_grad!(g, x_a, μ, problem::SEQUOIA_pb)
    ForwardDiff.gradient!(g, z -> ipm_obj(z, μ, problem), x_a)
end  

function ipm_obj(x_a, μ, problem::CUTEstModel)
    x=x_a[1:problem.meta.nvar];
    eq = length(problem.meta.jfix) + length(problem.meta.ifix);
    iq = length(problem.meta.jlow) + length(problem.meta.ilow)+length(problem.meta.jupp) + length(problem.meta.iupp) + 2*(length(problem.meta.jrng) + length(problem.meta.irng))
    λ=x_a[problem.meta.nvar+1:problem.meta.nvar+iq+eq];
    ν=λ[eq+1:eq+iq];
    s=x_a[problem.meta.nvar+iq+eq+1:end];
    Jac = dresdx(x, problem);


    return norm(grad(problem,x)+Jac'*λ)^2 + r_slack(x,s,problem) + norm(2*ν.*s.^2 .-μ)^2
end

function ipm_grad!(g, x_a, μ, problem::CUTEstModel)

    x=x_a[1:problem.meta.nvar];
    jeq=length(problem.meta.jfix);
    jlo=length(problem.meta.jlow);
    jup=length(problem.meta.jupp);
    jrg=length(problem.meta.jrng);
    ieq=length(problem.meta.ifix);
    ilo=length(problem.meta.ilow);
    iup=length(problem.meta.iupp);
    irg=length(problem.meta.irng);

    eq=jeq+ieq;
    iq=jlo+ilo+jup+iup+2*jrg+2*irg;

    λ=x_a[problem.meta.nvar+1:problem.meta.nvar+iq+eq];
    s=x_a[problem.meta.nvar+iq+eq+1:end];
    Jacobian=jac(problem,x);
    ν=λ[eq+1:eq+iq];

    arr=zeros(jeq+jlo+jup+jrg);
    arr[problem.meta.jfix[1:jeq]]=λ[1:jeq]
    arr[problem.meta.jlow[1:jlo]]=-λ[jeq+ieq+1:jeq+ieq+jlo]
    arr[problem.meta.jupp[1:jup]]=λ[jeq+ieq+jlo+ilo+1:jeq+ieq+jlo+ilo+jup]
    arr[problem.meta.jrng[1:jrg]]=-λ[jeq+ieq+jlo+ilo+jup+iup+1:jeq+ieq+jlo+ilo+jup+iup+jrg] .+ λ[jeq+ieq+jlo+ilo+jup+iup+jrg+1:jeq+ieq+jlo+ilo+jup+iup+2*jrg]

    cons=res(x,problem)

    Hx = hess(problem, x, arr);
    L = grad(problem,x)+Jacobian'*arr;
    for i in 1:ieq
        L[problem.meta.ifix[i]] += λ[jeq+i];
    end
    for i in 1:ilo
        L[problem.meta.ilow[i]] += -λ[jeq+ieq+jlo+i];
    end
    for i in 1:iup
        L[problem.meta.iupp[i]] += λ[jeq+ieq+jlo+ilo+jup+i];
    end
    for i in 1:irg
        L[problem.meta.irng[i]] += -λ[jeq+ieq+jlo+ilo+jup+iup+2*jrg+i]+λ[jeq+ieq+jlo+ilo+jup+iup+2*jrg+irg+i];
    end

    c=zeros(jeq+jlo+jup+jrg);
    for i in 1:jeq
        c[problem.meta.jfix[i]] += cons[i]
    end
    for i in 1:jlo
        c[problem.meta.jlow[i]] += -(cons[i+jeq+ieq]+s[i]^2)
    end
    for i in 1:jup
        c[problem.meta.jupp[i]] += (cons[jeq+ieq+jlo+ilo+i]+s[jlo+ilo+i]^2)
    end
    for i in 1:jrg
        c[problem.meta.jrng[i]] += -(cons[jeq+ieq+jlo+ilo+jup+iup+i]+s[jlo+ilo+jup+iup+i]^2) + (cons[jeq+ieq+jlo+ilo+jup+iup+jrg+i]+s[jlo+ilo+jup+iup+jrg+i]^2)
    end

    cvar=zeros(problem.meta.nvar);
    for i in 1:ieq
        cvar[problem.meta.ifix[i]] += cons[jeq+i]
    end
    for i in 1:ilo
        cvar[problem.meta.ilow[i]] += -(cons[jeq+ieq+jlo+i]+s[jlo+i]^2)
    end
    for i in 1:iup
        cvar[problem.meta.iupp[i]] += (cons[jeq+ieq+jlo+ilo+jupp+i]+s[jlo+ilo+jupp+i]^2)
    end
    for i in 1:irg
        cvar[problem.meta.irng[i]] += -(cons[jeq+ieq+jlo+ilo+jup+iup+2*jrg+i]+s[jlo+ilo+jup+iup+2*jrg+i]^2) + (cons[jeq+ieq+jlo+ilo+jup+iup+2*jrg+irg+i]+s[jlo+ilo+jup+iup+2*jrg+irg+i]^2)
    end

    dlambda=zeros(length(λ));

    for i in 1:ieq
        dlambda[jeq+i] = 2*λ[jeq+i]
    end
    for i in 1:ilo
        dlambda[jeq+ieq+jlo+i] = 2*λ[jeq+ieq+jlo+i]
    end
    for i in 1:iup
        dlambda[jeq+ieq+jlo+ilo+jup+i] = 2*λ[jeq+ieq+jlo+ilo+jup+i]
    end
    for i in 1:irg
        dlambda[jeq+ieq+jlo+ilo+jup+iup+2*jrg+i] = 2*λ[jeq+ieq+jlo+ilo+jup+iup+2*jrg+i]
        dlambda[jeq+ieq+jlo+ilo+jup+iup+2*jrg+irg+i] = 2*λ[jeq+ieq+jlo+ilo+jup+iup+2*jrg+irg+i]
    end

    for i in 1:jeq
        dlambda[i] = 2*λ[i]*norm(Jacobian[problem.meta.jfix[i],:])^2
    end
    for i in 1:jlo
        dlambda[jeq+ieq+i] = 2*λ[jeq+ieq+i]*norm(Jacobian[problem.meta.jlow[i],:])^2
    end
    for i in 1:jup
        dlambda[jeq+ieq+jlo+ilo+i] = 2*λ[jeq+ieq+jlo+ilo+i]*norm(Jacobian[problem.meta.jupp[i],:])^2
    end
    for i in 1:jrg
        dlambda[jeq+ieq+jlo+ilo+jup+iup+i] = 2*λ[jeq+ieq+jlo+ilo+jup+iup+i]*norm(Jacobian[problem.meta.jrng[i],:])^2
        dlambda[jeq+ieq+jlo+ilo+jup+iup+jrg+i] = 2*λ[jeq+ieq+jlo+ilo+jup+iup+jrg+i]*norm(Jacobian[problem.meta.jrng[i],:])^2
    end
    for i in jeq+ieq+1:jeq+ieq+jlo+ilo+jup+iup+2*jrg+2*irg
        dlambda[i]+=4*(2*λ[i]*s[i-jeq-ieq]^2-μ)*s[i-jeq-ieq]^2
    end

    ds=zeros(length(s))
    for i in 1:iq
        ds[i] = 4*(cons[i+jeq+ieq]+s[i]^2)*s[i]
    end
    for i in 1:iq
        ds[i] += 8*(2*λ[i+jeq+ieq]*s[i]^2-μ)*λ[i+jeq+ieq]*s[i]
    end
    #display(2*Jacobian'*c)

    g[1:problem.meta.nvar] = 2*transpose(Hx)*(L) + 2*Jacobian'*c +2*cvar
    g[problem.meta.nvar+1:problem.meta.nvar+length(λ)] = dlambda
    g[problem.meta.nvar+length(λ)+1:problem.meta.nvar+length(λ)+length(s)] = ds


end