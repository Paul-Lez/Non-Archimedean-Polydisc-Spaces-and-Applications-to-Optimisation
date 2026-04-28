"""
Generate figures for the NAML paper.

Reads stats JSON files (one per experiment suite) from a per-run directory
under `logs/` and writes PNG figures into `<run_dir>/figures/`. Each suite is
expected to have a stats file named `<suite>_stats.json`; missing files raise
an error.

Per-suite figures (one of each, four times):
  • <suite>_ranking.png            – mean rank per optimizer (bar)
  • <suite>_final_loss.png         – mean final loss per optimizer (bar, log)
  • <suite>_evals_vs_branching.png – mean evals vs branching factor (lines)
  • <suite>_times_vs_branching.png – mean runtime vs branching factor (lines)

Cross-suite figure:
  • overall_ranking.png            – mean rank across every configuration

Usage:
    # default: read from logs/latest, write to logs/latest/figures/
    julia --project=. src/generate_figures.jl

    # explicit run directory
    julia --project=. src/generate_figures.jl --run-dir logs/20260408_120000
"""

include(joinpath(@__DIR__, "figures_util.jl"))
include(joinpath(@__DIR__, "table_utils.jl"))   # provides load_stats_json

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))

"""Canonical experiment suite identifiers used for filenames and titles."""
const SUITES = [
    "polynomial_learning",
    "function_learning",
    "absolute_sum_minimization",
    "polynomial_solving",
]

"""Suites where raw loss is plotted instead of log_p(loss)."""

"""
    parse_run_dir(args) -> String

Parse a `--run-dir PATH` flag from `args`. Defaults to `<repo>/logs/latest`.
Relative paths are resolved against the repo root.
"""
function parse_run_dir(args)
    run_dir = joinpath(REPO_ROOT, "logs", "latest")
    for (i, arg) in enumerate(args)
        if arg == "--run-dir" && i < length(args)
            run_dir = args[i+1]
        elseif startswith(arg, "--run-dir=")
            run_dir = split(arg, "="; limit=2)[2]
        end
    end
    return isabspath(run_dir) ? run_dir : joinpath(REPO_ROOT, run_dir)
end

const RUN_DIR    = parse_run_dir(ARGS)
const FIGURE_DIR = joinpath(RUN_DIR, "figures")

"""Path to a suite's stats JSON inside the run directory."""
suite_stats_path(suite::String) = joinpath(RUN_DIR, "$(suite)_stats.json")

"""Path for an output figure under `<run_dir>/figures/`."""
figure_path(name::String) = joinpath(FIGURE_DIR, name)

# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

"""
    load_suite(suite) -> (experiments, optimizer_order, ablation_suite)

Load a suite's stats JSON. Errors loudly if the file is missing — the script
assumes all four stats files have already been produced. `ablation_suite` is
the key inside each experiment's `suites_aggregate` that figures should read
from (preferring `optimizer-comparison` when present).
"""
function load_suite(suite::String)
    path = suite_stats_path(suite)
    isfile(path) || error("Missing stats file for suite '$(suite)': $(path)")
    experiments, _, _ = load_stats_json(path)
    suites = list_suites(experiments)
    isempty(suites) && error("Stats file has no suites_aggregate entries: $(path)")
    return experiments, suites
end

# ----------------------------------------------------------------------------
# Per-suite figure generation
# ----------------------------------------------------------------------------

function generate_suite_figures(suite::String)
    println("\n=== $(suite) ===")
    experiments, ablation_suites = load_suite(suite)

    results = Tuple[]
    for ablation_suite in ablation_suites
        optimizer_order = get_optimizer_names(experiments; suite_name=ablation_suite)
        println("  Suite: $(ablation_suite)  Optimizers: $(join(optimizer_order, ", "))")

        save_figure(
            generate_ranking_plot_per_experiment(experiments, optimizer_order;
                suite_name = ablation_suite),
            figure_path("$(suite)_ranking_$(ablation_suite).png"))

        save_figure(
            generate_average_final_loss(experiments, optimizer_order;
                suite_name = ablation_suite),
            figure_path("$(suite)_final_loss_$(ablation_suite).png"))

        save_figure(
            generate_number_of_evals_plot(experiments, optimizer_order;
                suite_name = ablation_suite),
            figure_path("$(suite)_evals_vs_branching_$(ablation_suite).png"))

        save_figure(
            generate_times_plot(experiments, optimizer_order;
                suite_name = ablation_suite),
            figure_path("$(suite)_times_vs_branching_$(ablation_suite).png"))

        # Loss vs prime and loss vs dimension (categorical x-axis)
        # mean_final_loss is already log_p-normalised for non-function-learning
        # experiments (done in compute_aggregate_stats), so no extra transform needed.

        save_figure(
            generate_loss_vs_prime(experiments, optimizer_order;
                suite_name = ablation_suite),
            figure_path("$(suite)_loss_vs_prime_$(ablation_suite).png"))

        save_figure(
            generate_loss_vs_dimension(experiments, optimizer_order;
                suite_name = ablation_suite),
            figure_path("$(suite)_loss_vs_dimension_$(ablation_suite).png"))

        # Per-prime: mean evals and mean runtime vs dimension
        for p in experiment_primes(experiments)
            save_figure(
                generate_evals_by_dimension(experiments, optimizer_order;
                    prime = p, suite_name = ablation_suite),
                figure_path("$(suite)_evals_vs_dim_p$(p)_$(ablation_suite).png"))

            save_figure(
                generate_times_by_dimension(experiments, optimizer_order;
                    prime = p, suite_name = ablation_suite),
                figure_path("$(suite)_times_vs_dim_p$(p)_$(ablation_suite).png"))
        end

        push!(results, (experiments, optimizer_order, ablation_suite))
    end

    return results
end

# ----------------------------------------------------------------------------
# Cross-suite figure generation
# ----------------------------------------------------------------------------

"""
    merge_optimizer_orders(orders) -> Vector{String}

Combine the optimizer-name lists from several suites into a single ordering
that respects `DISPLAY_ORDER` and appends any unknown optimizers afterwards.
"""
function merge_optimizer_orders(orders)
    seen = Set{String}()
    for order in orders, name in order
        push!(seen, name)
    end
    ordered = String[name for name in DISPLAY_ORDER if name in seen]
    extras = sort([name for name in seen if !(name in DISPLAY_ORDER)])
    return vcat(ordered, extras)
end

function generate_cross_suite_figures(per_suite_data)
    println("\n=== overall ===")
    # Group by ablation suite across all experiment types
    ablation_names = unique(s for results in per_suite_data for (_, _, s) in results)
    for ablation_suite in ablation_names
        matching = [(exps, order) for results in per_suite_data
                    for (exps, order, s) in results if s == ablation_suite]
        all_experiments = vcat([exps for (exps, _) in matching]...)
        optimizer_order = merge_optimizer_orders([order for (_, order) in matching])

        save_figure(
            generate_overall_ranking_plot(all_experiments, optimizer_order;
                suite_name = ablation_suite),
            figure_path("overall_ranking_$(ablation_suite).png"))
    end
end

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

function main()
    isdir(RUN_DIR) || error("Run directory does not exist: $(RUN_DIR)")
    println("Reading stats from: $(RUN_DIR)")
    println("Writing figures to: $(FIGURE_DIR)")

    per_suite_data = [generate_suite_figures(suite) for suite in SUITES]
    generate_cross_suite_figures(per_suite_data)

    println("\nDone.")
end

main()
