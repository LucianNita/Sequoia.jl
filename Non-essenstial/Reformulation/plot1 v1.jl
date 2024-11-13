using JLD2, DataFrames, Plots

# Function to load and combine multiple .jld2 files into one DataFrame
function load_data(files)
    combined_df = DataFrame()
    required_columns = [:solver_name, :id, :name, :nvar, :ncon, :nequ, :status, :objective, :elapsed_time, 
    :iter, :dual_feas, :primal_feas, :neval_obj, :neval_grad, :neval_cons, 
    :neval_cons_lin, :neval_cons_nln, :neval_jcon, :neval_jgrad, :neval_jac, 
    :neval_jac_lin, :neval_jac_nln, :neval_jprod, :neval_jprod_lin, :neval_jprod_nln, 
    :neval_jtprod, :neval_jtprod_lin, :neval_jtprod_nln, :neval_hess, :neval_hprod, 
    :neval_jhess, :neval_jhprod, :neval_residual, :neval_jac_residual, :neval_jprod_residual, 
    :neval_jtprod_residual, :neval_hess_residual, :neval_jhess_residual, :neval_hprod_residual, 
    :extrainfo, :real_time, :internal_msg] # List all required columns

    for file in eachindex(files)
        data = load(files[file])
        df = data[namess[file]]
        #display(df[:,1:end-2])
        #display(df)
        # Ensure all required columns are present
        #for col in required_columns
        #    if !(col in names(df))
        #        df[!, col] = fill(missing, nrow(df))
        #    end
        #end
        #display(df)
        #if mod(file,2)==1
        #    combined_df = vcat(combined_df, df[:,1:end-2]) # Concatenate data from each file
        #else
            combined_df = vcat(combined_df, df[:,1:40]) # Concatenate data from each file
        #end
        #display(combined_df)
    end

    # Assuming dfs is a collection of DataFrames
    #df = DataFrame()
    
        # Ensure all required columns are present
    #    for col in required_columns
    #        if !(col in names(combined_df))
    #            combined_df[!, col] = fill(missing, nrow(combined_df))
    #        end
    #    end
        # Append to the combined DataFrame
    #    append!(df, combined_df)

    # Filter out rows with missing values in essential columns before profiling
    #df = dropmissing(df, [:objective, :elapsed_time])  # Adjust columns as needed
    #display(df)
    return combined_df
end

function performance_profile(df, metric_col::Symbol, algorithm_col::Symbol, problem_col::Symbol)
    # Get unique algorithms and problems
    algorithms = unique(df[!, algorithm_col])
    problems = unique(df[!, problem_col])

    # Create a DataFrame to store the performance ratios
    pivot_data = DataFrame(problems = problems)

    # Calculate performance ratios
    for alg in algorithms
        # Filter data for the current algorithm
        alg_data = df[df[!, algorithm_col] .== alg, :]
        ratios = Float64[]

        for prob in problems
            # Filter data for the current problem
            problem_data = df[df[!, problem_col] .== prob, :]

            # Filter for rows where the problem status is :first_order and metric_col is a valid number
            valid_entries = problem_data[problem_data[!, :status] .== :first_order, :]
            valid_entries = valid_entries[.!isnan.(valid_entries[!, metric_col]) .& .!isinf.(valid_entries[!, metric_col]), :]
            if isempty(valid_entries)
                push!(ratios, NaN)  # If no valid data, add NaN
                continue
            end

            # Calculate the minimum metric value for the problem across all algorithms
            min_time = minimum(valid_entries[!, metric_col])

            # Get the metric value for the current algorithm and problem
            alg_time_data = alg_data[alg_data[!, problem_col] .== prob, metric_col]

            if isempty(alg_time_data) || isnan(alg_time_data[1]) || isinf(alg_time_data[1]) || problem_data[1, :status] != :first_order
                push!(ratios, NaN)  # Add NaN if no valid time for this algorithm-problem pair
            else
                alg_time = alg_time_data[1]
                push!(ratios, alg_time / min_time)
            end
        end
        
        pivot_data[!, Symbol(alg)] = ratios
    end

    # Generate x-values for performance ratios
    flattened_values = filter(x -> !isnan(x) && !isinf(x), collect(Iterators.flatten(eachcol(pivot_data[!, 2:end]))))
    if isempty(flattened_values)
        println("Warning: No valid data found in pivot_data.")
        return [], Dict()
    else
        max_ratio = maximum(flattened_values)
        x_vals = range(1, stop=max_ratio, length=100)
    end

    # Calculate the cumulative distribution function for each algorithm
    y_vals = Dict()
    for alg in algorithms
        alg_col = skipmissing(pivot_data[!, Symbol(alg)])
        y_vals[alg] = [sum(alg_col .<= x) / length(problems) for x in x_vals]
    end

    return x_vals, y_vals
end


# File paths for all .jld2 files
files = ["/home/ln416/Sequoia.jl/ipo_cons_O0_tol4_ipo22.jld2", "/home/ln416/Sequoia.jl/perci_cons_O0_tol4_perci22.jld2", "/home/ln416/Sequoia.jl/seq_cons_O0_tol4_seqip22.jld2", "/home/ln416/Sequoia.jl/seq_cons_O0_tol4_seqp22.jld2"]  # Add paths to your files
namess = ["ipopt_stats", "perci_stats", "seqip_stats", "seqp_stats"]
# Load and combine data from all files
df = load_data(files)

# Generate performance profile data for elapsed time
x_vals, y_vals = performance_profile(df, :elapsed_time, :solver_name, :id)

# Plot the performance profile
plot()
for (algo, y) in y_vals
    plot!(x_vals, y, label=String(algo), xlabel="Performance Ratio", ylabel="Fraction of Problems Solved")
end
title!("Performance Profile for Multiple Algorithms")
#legend()
