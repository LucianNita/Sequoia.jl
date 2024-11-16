using Plots
using JLD2, DataFrames
sequoia_stats = jldopen("sequoia1_cons_O0_tol4.jld2", "r") do file
    data = file["sequoia_stats5"]
    data
end
sequoia_stats2 = jldopen("sequoia2_cons_O0_tol4.jld2", "r") do file
    data = file["sequoia_stats2"]
    data
end
sequoia_stats3 = jldopen("sequoia3_cons_O0_tol4.jld2", "r") do file
    data = file["sequoia_stats3"]
    data
end
sequoia_stats4 = jldopen("sequoia4_cons_O0_tol4.jld2", "r") do file
    data = file["sequoia_stats4"]
    data
end
sequoia_stats5 = jldopen("sequoia5_cons_O0_tol4.jld2", "r") do file
    data = file["sequoia_stats5"]
    data
end
# Example solvers and problems
solvers = ["lin p=11", "lin p=2", "lin p=4", "lin p=6", "lin p=8"]#"sequoia_with_objUB_quad", "sequoia_with_dk_quad", "sequoia_with_objUB_lin", "sequoia_with_dk_lin"]
problems = sequoia_stats.name

time1=sequoia_stats.elapsed_time
for i in 1:length(problems)
    if sequoia_stats.status[i]!=:first_order 
        time1[i]=Inf 
    end 
end

time2=sequoia_stats2.elapsed_time
for i in 1:length(problems)
    if sequoia_stats2.status[i]!=:first_order 
        time2[i]=Inf 
    end 
end

time3=sequoia_stats3.elapsed_time
for i in 1:length(problems)
    if sequoia_stats3.status[i]!=:first_order 
        time3[i]=Inf 
    end 
end

time4=sequoia_stats4.elapsed_time
for i in 1:length(problems)
    if sequoia_stats4.status[i]!=:first_order 
        time4[i]=Inf 
    end 
end

time5=sequoia_stats5.elapsed_time
for i in 1:length(problems)
    if sequoia_stats5.status[i]!=:first_order 
        time4[i]=Inf 
    end 
end
# Metrics: runtime or iterations (rows = problems, columns = solvers)
# Replace these with actual solver timings or iteration counts
metrics = hcat(time1, time2, time3, time4, time5)

# Compute t_*p (best performance for each problem)
t_star = minimum(metrics, dims=2)

# Compute performance ratios r_{s,p}
ratios = metrics ./ t_star

# Initialize τ values for plotting
τ_values = 1:0.01:10  # Adjust as needed
ρ = zeros(length(τ_values), length(solvers))

# Compute performance profile ρ_s(τ) for each solver
for (j, solver) in enumerate(solvers)
    for (i, τ) in enumerate(τ_values)
        ρ[i, j] = count(ratios[:, j] .<= τ) / size(metrics, 1)
    end
end

# Plot performance profiles
plot(τ_values, ρ[:, 1], label=solvers[1], lw=2, xlabel="τ (Performance Factor)", ylabel="ρ_s(τ)", title="Performance Profiles")
for j in 2:length(solvers)
    plot!(τ_values, ρ[:, j], label=solvers[j], lw=2)
end
