"""
Shared experiment utilities for the experiment infrastructure.

Provides:
1. Unified CLI argument parsing
2. Canonical optimizer factory (all optimizers in one place)
3. Threaded experiment runner (parallelizes across optimizers within a sample)
4. Raw JSON serialization (no statistics — that's make_stats.jl's job)

Usage from any run_experiments.jl:
    include("../experiment_utils.jl")
    args = parse_experiment_args(ARGS)
    ...
"""

using JSON
using Printf
using Dates
using Random
using Distributed

const DEFAULT_EPOCHS = 120
const DEFAULT_SAMPLES = 50
const QUICK_EPOCHS = 5
const QUICK_SAMPLES = 5
const DEFAULT_RANDOM_SEED = 42
const SAMPLE_SEED_STRIDE = 100_000

"""
    seed_all_rngs!(base_seed=DEFAULT_RANDOM_SEED)

Seed the RNG on the master process and every distributed worker. Workers get a
deterministic offset from `base_seed` so parallel runs do not inherit unrelated
Julia worker RNG state.

This is mostly hygiene: it controls any incidental randomness outside the
per-sample path, but it does not by itself make `pmap` runs reproducible because
task scheduling can assign a sample to different workers or different stream
positions.
"""
function seed_all_rngs!(base_seed::Integer=DEFAULT_RANDOM_SEED)
    Random.seed!(base_seed)
    for worker in workers()
        remotecall_wait(Random.seed!, worker, base_seed + worker)
    end
    return nothing
end

"""
    seed_sample_rng!(sample_num, config_idx; base_seed=DEFAULT_RANDOM_SEED)

Seed the RNG inside the worker task handling a concrete `(config, sample)` pair.
This means that the random problem generation is pinned to the
sample identity rather than to whichever worker `pmap` assigns the pair
and so on.
"""
function seed_sample_rng!(sample_num::Integer, config_idx::Integer;
                          base_seed::Integer=DEFAULT_RANDOM_SEED)
    Random.seed!(base_seed + SAMPLE_SEED_STRIDE * config_idx + sample_num)
    return nothing
end

# ============================================================================
# CLI Argument Parsing
# ============================================================================

"""
    parse_experiment_args(ARGS) -> NamedTuple

Unified CLI argument parser for all run_experiments.jl scripts.

Returns a NamedTuple with fields:
- quick_mode::Bool
- save_results::Bool
- use_config_file::Bool
- use_paper_config::Bool
- n_epochs::Int
- output_filename::Union{String, Nothing}
- n_samples_override::Union{Int, Nothing}
- selection_mode  (NAML.BestValue, NAML.VisitCount, or NAML.BestLoss)
- tree_degree_override::Union{Int, Nothing}
- description::String
- git_commit::String
"""
function parse_experiment_args(args)
    quick_mode = "--quick" in args
    save_results = "--save" in args
    use_config_file = "--config" in args
    
    # Paper-ready flags
    use_optimizer_comparison = "--paper-optimizer-comparison" in args || "--paper" in args
    use_mcts_branching = "--paper-mcts-branching" in args
    use_dag_mcts_branching = "--paper-dag-mcts-branching" in args
    use_greedy_branching = "--paper-greedy-descent-branching" in args
    use_gradient_branching = "--paper-gradient-descent-branching" in args
    use_mcts_sims = "--paper-mcts-number-of-simulations" in args
    use_dag_mcts_sims = "--paper-dag-mcts-number-of-simulations" in args
    use_mcts_exp = "--paper-mcts-exploration-constant" in args
    use_dag_mcts_exp = "--paper-dag-mcts-exploration-constant" in args

    use_paper_config = use_optimizer_comparison || use_mcts_branching || 
                      use_dag_mcts_branching || use_greedy_branching || 
                      use_gradient_branching || use_mcts_sims || 
                      use_dag_mcts_sims || use_mcts_exp || use_dag_mcts_exp

    # Default run sizes
    n_epochs = quick_mode ? QUICK_EPOCHS : DEFAULT_EPOCHS

    output_filename = nothing
    n_samples_override = quick_mode ? QUICK_SAMPLES : DEFAULT_SAMPLES
    selection_mode = NAML.BestValue
    tree_degree_override = nothing
    description = ""
    git_commit = ""

    for (i, arg) in enumerate(args)
        if arg == "--epochs" && i < length(args)
            n_epochs = parse(Int, args[i+1])
        elseif arg == "--output" && i < length(args)
            output_filename = args[i+1]
        elseif arg == "--samples" && i < length(args)
            n_samples_override = parse(Int, args[i+1])
        elseif arg == "--selection-mode" && i < length(args)
            mode_str = args[i+1]
            if mode_str == "BestValue"
                selection_mode = NAML.BestValue
            elseif mode_str == "VisitCount"
                selection_mode = NAML.VisitCount
            elseif mode_str == "BestLoss"
                selection_mode = NAML.BestLoss
            else
                error("Invalid selection mode: $mode_str. Must be BestValue, VisitCount, or BestLoss")
            end
        elseif arg == "--degree" && i < length(args)
            tree_degree_override = parse(Int, args[i+1])
        elseif startswith(arg, "--degree=")
            tree_degree_override = parse(Int, arg[10:end])
        elseif arg == "--description" && i < length(args)
            description = args[i+1]
        elseif arg == "--git-commit" && i < length(args)
            git_commit = args[i+1]
        end
    end

    return (
        quick_mode = quick_mode,
        save_results = save_results,
        use_config_file = use_config_file,
        use_paper_config = use_paper_config,
        use_optimizer_comparison = use_optimizer_comparison,
        use_mcts_branching = use_mcts_branching,
        use_dag_mcts_branching = use_dag_mcts_branching,
        use_greedy_branching = use_greedy_branching,
        use_gradient_branching = use_gradient_branching,
        use_mcts_sims = use_mcts_sims,
        use_dag_mcts_sims = use_dag_mcts_sims,
        use_mcts_exp = use_mcts_exp,
        use_dag_mcts_exp = use_dag_mcts_exp,
        n_epochs = n_epochs,
        output_filename = output_filename,
        n_samples_override = n_samples_override,
        selection_mode = selection_mode,
        tree_degree_override = tree_degree_override,
        description = description,
        git_commit = git_commit,
    )
end


# ============================================================================
# Load configurations
# ============================================================================

function load_config_file(experiment_dir::String, args)
    if args.use_paper_config
        include(joinpath(experiment_dir, "paper_config.jl"))
    elseif args.use_config_file
        include(joinpath(experiment_dir, "config.jl"))
    end
end

"""
    load_configs(experiment_dir, args, default_configs) -> Vector{Dict}

Load experiment configurations based on CLI flags.
"""
function load_configs(args, default_configs::Vector)
    configs = if args.use_paper_config
        println("Loaded PAPER-READY experiment configurations")
        paper_experiments
    elseif args.use_config_file
        println("Loaded experiment configurations from config.jl")
        experiment_configs
    else
        println("Using default configurations")
        default_configs
    end

    for config in configs
        config["num_samples"] = args.n_samples_override
    end
    println("Using $(args.n_samples_override) samples per config")

    return configs
end


# ============================================================================
# Optimizer Factory
# ============================================================================

# perhaps confusingly in the naming scheme here, k is the branching factor!
"""Canonical display ordering for all experiments."""
const OPTIMIZER_ORDER = [
    "Random", "Best-First", "Best-First-branch2", "Best-First-Gradient",
    "MCTS-k", "MCTS-5k", "MCTS-10k",
    "DAG-MCTS-k", "DAG-MCTS-5k", "DAG-MCTS-10k",
    "DOO"
]

const SUITE_ORDER = [
    "optimizer-comparison",
    "mcts-branching",
    "dag-mcts-branching",
    "greedy-descent-branching",
    "gradient-descent-branching",
    "mcts-number-of-simulations",
    "dag-mcts-number-of-simulations",
    "mcts-exploration-constant",
    "dag-mcts-exploration-constant",
]

const NAME_WIDTH = maximum(length(n) for n in OPTIMIZER_ORDER)

function ordered_optimizer_names(opt_configs::Dict)
    names = collect(keys(opt_configs))
    ordered = String[]
    for name in OPTIMIZER_ORDER
        name in names && push!(ordered, name)
    end
    append!(ordered, sort([name for name in names if !(name in ordered)]))
    return ordered
end

function ordered_suite_names(suite_configs::Dict)
    names = collect(keys(suite_configs))
    ordered = String[]
    for name in SUITE_ORDER
        name in names && push!(ordered, name)
    end
    append!(ordered, sort([name for name in names if !(name in ordered)]))
    return ordered
end

"""
    get_optimizer_configs(config::Dict, args::NamedTuple; doo_delta_scale::Real=1) -> Dict{String, Dict{String, Any}}

Return a nested Dict of SuiteName => { OptimizerName => OptimizerSetup }.
Each Setup contains:
- `"init"`: `(param, loss) -> OptimSetup`
- `"refinement_degree"`: the tree-refinement degree used by that optimizer
- `"branching_factor"`: the number of children generated by one expansion
- `"strict_refinement"`: whether expansion uses strict coordinate scheduling
- `"n_epochs"` (optional): optimizer-specific override for the run length

Results are organized by suite to allow rigorous comparison. Optimizers may 
appear in multiple suites and will be run independently for each.
"""
function get_optimizer_configs(config::Dict, args::NamedTuple; doo_delta_scale::Real=1)
    suites = Dict{String, Dict{String, Any}}()

    prime = config["prime"]
    prec = config["prec"]
    # num_vars is the parameter-polydisc dimension. Some experiments (polynomial
    # learning, function learning) do not set it explicitly — they learn
    # degree+1 polynomial coefficients.
    dim = if haskey(config, "num_vars")
        config["num_vars"]
    elseif haskey(config, "degree")
        config["degree"] + 1
    else
        error("Cannot determine num_vars for config $(get(config, "name", "?"))")
    end
    quick = args.quick_mode
    selection_mode = args.selection_mode
    
    p_float = Float64(prime)
    delta_scale = Float64(doo_delta_scale)

    function optimizer_setup(init_fn, refinement_degree;
                             strict_refinement=false,
                             branching_factor=nothing,
                             n_epochs=nothing)
        bf = isnothing(branching_factor) ?
             tree_branching_factor(prime, dim, refinement_degree;
                                   strict=strict_refinement) :
             branching_factor
        setup = Dict(
            "init" => init_fn,
            "refinement_degree" => refinement_degree,
            "branching_factor" => bf,
            "strict_refinement" => strict_refinement,
        )
        if !isnothing(n_epochs)
            setup["n_epochs"] = n_epochs
        end
        return setup
    end

    # Helper to create standard MCTS config
    function mk_mcts(sims, deg, exp=1.41)
        return optimizer_setup(
            (param, loss) -> begin
                c = NAML.MCTSConfig(
                    num_simulations=sims,
                    exploration_constant=exp,
                    selection_mode=selection_mode,
                    degree=deg
                )
                NAML.mcts_descent_init(param, loss, c)
            end,
            deg
        )
    end

    # Helper to create standard DAG-MCTS config
    function mk_dag_mcts(sims, deg, exp=1.41)
        return optimizer_setup(
            (param, loss) -> begin
                c = NAML.DAGMCTSConfig(
                    num_simulations=sims,
                    exploration_constant=exp,
                    degree=deg,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, c)
            end,
            deg
        )
    end

    # suite 0: Standard Optimizer Comparison
    if args.use_optimizer_comparison
        s = Dict{String, Any}()
        deg = effective_degree(dim, args.tree_degree_override)
        k = tree_branching_factor(prime, dim, deg)
        sims_10k = quick ? 200 : 10 * k
        mcts_steps = effective_mcts_steps(dim, prec, deg, args.n_epochs)
        doo_branching_factor = tree_branching_factor(prime, dim, deg; strict=true)
        doo_steps = doo_epoch_budget(k, mcts_steps, doo_branching_factor;
                                     simulations_per_step=sims_10k)

        s["Random"] = optimizer_setup((param, loss) -> NAML.random_descent_init(param, loss, 1, (false, deg)), deg)
        s["Best-First"] = optimizer_setup((param, loss) -> NAML.greedy_descent_init(param, loss, 1, (false, deg)), deg)
        s["Best-First-Gradient"] = optimizer_setup((param, loss) -> NAML.gradient_descent_init(param, loss, 1, (false, deg)), deg)
        s["MCTS-10k"] = mk_mcts(sims_10k, deg)
        s["DAG-MCTS-10k"] = mk_dag_mcts(sims_10k, deg)
        s["DOO"] = optimizer_setup(
            (param, loss) -> begin
                delta = h -> delta_scale * p_float^(-h)
                c = NAML.DOOConfig(delta=delta, degree=deg, strict=true)
                NAML.doo_descent_init(param, loss, 1, c)
            end,
            deg;
            strict_refinement=true,
            branching_factor=doo_branching_factor,
            n_epochs=doo_steps
        )
        suites["optimizer-comparison"] = s
    end

    # suite 1: MCTS Branching (2+ vars, deg 1 & 2, 10k sims)
    if args.use_mcts_branching && dim >= 2
        s = Dict{String, Any}()
        for deg in [1, 2]
            k = binomial(dim, deg) * prime^deg
            sims = quick ? 200 : 10 * k
            s["MCTS-10k-deg$deg"] = mk_mcts(sims, deg)
        end
        suites["mcts-branching"] = s
    end

    # suite 2: DAG-MCTS Branching (2+ vars, deg 1 & 2, 10k sims)
    if args.use_dag_mcts_branching && dim >= 2
        s = Dict{String, Any}()
        for deg in [1, 2]
            k = binomial(dim, deg) * prime^deg
            sims = quick ? 200 : 10 * k
            s["DAG-MCTS-10k-deg$deg"] = mk_dag_mcts(sims, deg)
        end
        suites["dag-mcts-branching"] = s
    end

    # suite 3: Greedy Branching (2+ vars, deg 1 & 2)
    if args.use_greedy_branching && dim >= 2
        s = Dict{String, Any}()
        s["Greedy-deg1"] = optimizer_setup((param, loss) -> NAML.greedy_descent_init(param, loss, 1, (false, 1)), 1)
        s["Greedy-deg2"] = optimizer_setup((param, loss) -> NAML.greedy_descent_init(param, loss, 1, (false, 2)), 2)
        suites["greedy-descent-branching"] = s
    end

    # suite 4: Gradient Branching (2+ vars, deg 1 & 2)
    if args.use_gradient_branching && dim >= 2
        s = Dict{String, Any}()
        s["Gradient-deg1"] = optimizer_setup((param, loss) -> NAML.gradient_descent_init(param, loss, 1, (false, 1)), 1)
        s["Gradient-deg2"] = optimizer_setup((param, loss) -> NAML.gradient_descent_init(param, loss, 1, (false, 2)), 2)
        suites["gradient-descent-branching"] = s
    end

    # suite 5: MCTS Number of Simulations (2+ vars, deg 2, k/5k/10k sims)
    if args.use_mcts_sims && dim >= 2
        s = Dict{String, Any}()
        deg = 2
        k = binomial(dim, deg) * prime^deg
        s["MCTS-k"] = mk_mcts(quick ? 50 : k, deg)
        s["MCTS-5k"] = mk_mcts(quick ? 100 : 5 * k, deg)
        s["MCTS-10k"] = mk_mcts(quick ? 200 : 10 * k, deg)
        suites["mcts-number-of-simulations"] = s
    end

    # suite 6: DAG-MCTS Number of Simulations (2+ vars, deg 2, k/5k/10k sims)
    if args.use_dag_mcts_sims && dim >= 2
        s = Dict{String, Any}()
        deg = 2
        k = binomial(dim, deg) * prime^deg
        s["DAG-MCTS-k"] = mk_dag_mcts(quick ? 50 : k, deg)
        s["DAG-MCTS-5k"] = mk_dag_mcts(quick ? 100 : 5 * k, deg)
        s["DAG-MCTS-10k"] = mk_dag_mcts(quick ? 200 : 10 * k, deg)
        suites["dag-mcts-number-of-simulations"] = s
    end

    # suite 7: MCTS Exploration Constant (2+ vars, deg 2, 10k sims, sweep exp)
    if args.use_mcts_exp && dim >= 2
        s = Dict{String, Any}()
        deg = 2
        k = binomial(dim, deg) * prime^deg
        sims = quick ? 200 : 10 * k
        for exp in 1.4:0.1:2.4
            name = @sprintf("MCTS-10k-exp%.1f", exp)
            s[name] = mk_mcts(sims, deg, exp)
        end
        suites["mcts-exploration-constant"] = s
    end

    # suite 8: DAG-MCTS Exploration Constant (2+ vars, deg 2, 10k sims, sweep exp)
    if args.use_dag_mcts_exp && dim >= 2
        s = Dict{String, Any}()
        deg = 2
        k = binomial(dim, deg) * prime^deg
        sims = quick ? 200 : 10 * k
        for exp in 1.4:0.1:2.4
            name = @sprintf("DAG-MCTS-10k-exp%.1f", exp)
            s[name] = mk_dag_mcts(sims, deg, exp)
        end
        suites["dag-mcts-exploration-constant"] = s
    end

    return suites
end

"""
    effective_degree(num_dims, tree_degree_override) -> Int

Compute the effective MCTS/tree branching degree.
Default: 1 for 1-dimensional, 2 for ≥2 dimensions.
"""
function effective_degree(num_dims::Int, tree_degree_override)
    auto_degree = num_dims >= 2 ? 2 : 1
    return isnothing(tree_degree_override) ? auto_degree : tree_degree_override
end

"""
    tree_branching_factor(prime, num_dims, degree; strict=false) -> Int

Number of children produced by one optimizer expansion. Strict refinement fixes
one degree-`d` coordinate subset per depth, giving `p^d` children instead of
`binomial(n, d) * p^d`.
"""
function tree_branching_factor(prime::Int, num_dims::Int, degree::Int; strict::Bool=false)
    1 <= degree <= num_dims ||
        error("refinement degree must be between 1 and dimension, got degree=$degree, dimension=$num_dims")
    return strict ? prime^degree : binomial(num_dims, degree) * prime^degree
end

"""
    effective_mcts_steps(num_dims, precision, optimizer_degree, max_epochs) -> Int

Approximate the number of MCTS steps before the run either
hits the configured epoch cap or reaches max precision.

This is used to compute the number of steps we give to DOO to make the number of function evaluations 
roughly match.
"""
function effective_mcts_steps(num_dims::Int, precision::Int,
                              optimizer_degree::Int, max_epochs::Int)
    full_precision_steps = floor(Int, num_dims * precision / optimizer_degree)
    return min(max_epochs, full_precision_steps)
end

"""
    doo_epoch_budget(mcts_branching_factor, mcts_effective_steps,
                     doo_branching_factor; simulations_per_step=nothing) -> Int

Compute the DOO run length for the optimizer-comparison suite.
The goal is to match the number of function evaluations allowed for MCTS.
"""
function doo_epoch_budget(mcts_branching_factor::Int, mcts_effective_steps::Int,
                          doo_branching_factor::Int; simulations_per_step=nothing)
    # Paper runs use 10 * branching factor simulations per step, so the MCTS evaluation
    # budget is 10 * branching factor * effective steps. Quick runs pass their
    # smaller actual simulation count via `simulations_per_step`.
    sims = isnothing(simulations_per_step) ? 10 * mcts_branching_factor :
           simulations_per_step
    return round(Int, sims * mcts_effective_steps / doo_branching_factor)
end


# ============================================================================
# Single optimizer run
# ============================================================================

"""
    run_single_optimizer(opt_name, opt_setup, initial_param, loss, n_epochs;
                         post_run_fn=nothing) -> Dict

Run a single optimizer on a single problem instance.
Returns a Dict with raw results (no ranking or aggregate stats).

Deep-copies `initial_param` to avoid mutation issues when running in parallel.

If `post_run_fn` is provided, it is called as `post_run_fn(optim)` and
the returned Dict is merged into the result. Use this for experiment-specific
fields like classification accuracy.
"""
function run_single_optimizer(opt_name::String, opt_setup::Dict,
                               initial_param, loss, n_epochs::Int;
                               post_run_fn::Union{Function,Nothing}=nothing)
    # Deep copy starting parameter to avoid cross-thread mutation
    param_copy = deepcopy(initial_param)
    initial_loss_val = loss.eval([param_copy])[1]
    refinement_degree = opt_setup["refinement_degree"]
    branching_factor = opt_setup["branching_factor"]
    strict_refinement = opt_setup["strict_refinement"]
    effective_n_epochs = get(opt_setup, "n_epochs", n_epochs)

    try
        # Wrap loss with evaluation counting
        counted_loss, eval_counter = wrap_loss_with_counting(loss)

        optim = opt_setup["init"](param_copy, counted_loss)

        losses = Float64[]
        t_start = time()

        # run until convergence
        for epoch in 1:effective_n_epochs
            current_loss = NAML.eval_loss(optim)
            push!(losses, current_loss)
            NAML.step!(optim)
            NAML.has_converged(optim) && break
        end

        t_end = time()
        elapsed = t_end - t_start

        final_loss = NAML.eval_loss(optim)
        push!(losses, final_loss)

        # Subtract monitoring eval_loss calls
        monitoring_evals = length(losses)
        total_optimizer_evals = eval_counter.eval_count - monitoring_evals + eval_counter.grad_count

        result = Dict{String, Any}(
            "time" => elapsed,
            "final_loss" => final_loss,
            "losses" => losses,
            "improvement" => initial_loss_val - final_loss,
            "improvement_ratio" => (initial_loss_val > 0) ?
                (initial_loss_val - final_loss) / initial_loss_val : 0.0,
            "total_evals" => total_optimizer_evals,
            "refinement_degree" => refinement_degree,
            "branching_factor" => branching_factor,
            "strict_refinement" => strict_refinement,
            "n_epochs" => effective_n_epochs,
        )

        # Run experiment-specific post-processing (e.g., accuracy computation)
        if !isnothing(post_run_fn)
            extra = post_run_fn(optim)
            merge!(result, extra)
        end

        return result
    catch e
        return Dict{String, Any}(
            "error" => string(e),
            "refinement_degree" => refinement_degree,
            "branching_factor" => branching_factor,
            "strict_refinement" => strict_refinement,
            "n_epochs" => effective_n_epochs,
        )
    end
end


# ============================================================================
# Threaded run across all optimizers for one sample
# ============================================================================

"""
    run_all_optimizers_serial(opt_configs, initial_param, loss, n_epochs;
                              post_run_fn=nothing) -> Dict{String,Any}

Run all optimizers on a single problem instance, serially (no threading).

Use this when the caller is already parallelizing at a coarser level (e.g. over
`(config, sample)` pairs) and each task owns its own `loss`. Running optimizers
serially here avoids nested threading and contention on shared evaluator state.
"""
function run_all_optimizers_serial(opt_configs::Dict, initial_param, loss, n_epochs::Int;
                                    post_run_fn::Union{Function,Nothing}=nothing)
    results = Dict{String, Any}()
    for opt_name in ordered_optimizer_names(opt_configs)
        opt_setup = opt_configs[opt_name]
        results[opt_name] = run_single_optimizer(opt_name, opt_setup, initial_param, loss, n_epochs;
                                                  post_run_fn=post_run_fn)
    end
    return results
end


"""
    run_all_optimizers_threaded(opt_configs, initial_param, loss, n_epochs;
                                post_run_fn=nothing) -> Dict{String,Any}

Run all optimizers on a single problem instance, using threads.
Returns a Dict mapping optimizer name => result Dict.
"""
function run_all_optimizers_threaded(opt_configs::Dict, initial_param, loss, n_epochs::Int;
                                     post_run_fn::Union{Function,Nothing}=nothing)
    opt_names = ordered_optimizer_names(opt_configs)

    results = Dict{String, Any}()
    result_lock = ReentrantLock()

    Threads.@threads for i in 1:length(opt_names)
        opt_name = opt_names[i]
        opt_setup = opt_configs[opt_name]
        result = run_single_optimizer(opt_name, opt_setup, initial_param, loss, n_epochs;
                                       post_run_fn=post_run_fn)
        lock(result_lock) do
            results[opt_name] = result
        end
    end

    return results
end


# ============================================================================
# JSON serialization
# ============================================================================

"""
    build_metadata(; experiment_type, n_epochs, quick_mode, optimizer_order,
                     description, git_commit, extra...) -> Dict

Build metadata dict for JSON output.
"""
function build_metadata(; experiment_type::String,
                          n_epochs::Int,
                          quick_mode::Bool,
                          optimizer_order::Vector{String}=String[],
                          suites::Vector{String}=String[],
                          description::String="",
                          git_commit::String="",
                          extra::Dict{String,Any}=Dict{String,Any}())
    naml_version = pkgversion(NAML)
    metadata = Dict{String, Any}(
        "experiment_type" => experiment_type,
        "timestamp" => Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
        "n_epochs" => n_epochs,
        "quick_mode" => quick_mode,
        "optimizer_order" => optimizer_order,
        "suites" => suites,
        "description" => description,
        "experiment_git_commit" => git_commit,
        "naml_version" => isnothing(naml_version) ? "" : string(naml_version),
        "julia_version" => string(VERSION),
    )
    merge!(metadata, extra)
    return metadata
end

"""
    save_raw_results(all_results, metadata, filepath)

Save raw experiment results to JSON. No aggregate stats — just raw per-sample data.
"""
function save_raw_results(all_results::Vector, metadata::Dict, filepath::String)
    json_experiments = []
    for result in all_results
        json_result = Dict{String, Any}()
        json_result["config"] = result["config"]

        if haskey(result, "error")
            json_result["error"] = result["error"]
        else
            json_result["samples"] = result["samples"]
        end

        push!(json_experiments, json_result)
    end

    json_output = Dict{String, Any}(
        "metadata" => metadata,
        "experiments" => json_experiments,
    )

    open(filepath, "w") do f
        JSON.print(f, json_output, 2)
    end

    println("\n✓ Raw results saved to: $filepath")
end


# ============================================================================
# Progress printing
# ============================================================================

"""Print a brief per-sample summary to stdout."""
function print_sample_summary(sample_result::Dict, initial_loss::Float64)
    println(@sprintf("    Initial: %.6e", initial_loss))
    for opt_name in OPTIMIZER_ORDER
        if haskey(sample_result, opt_name)
            opt_result = sample_result[opt_name]
            if !haskey(opt_result, "error")
                println(Printf.format(Printf.Format("    %-$(NAME_WIDTH)s Final: %.6e (Δ: %.6e, %.1f%%)"),
                    opt_name, opt_result["final_loss"], opt_result["improvement"],
                    opt_result["improvement_ratio"] * 100))
            end
        end
    end
end
