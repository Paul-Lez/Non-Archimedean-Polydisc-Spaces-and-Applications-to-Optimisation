"""
Figure-generation utilities for the NAML paper.

All plots use Plots.jl with the GR backend. Pure aggregation logic lives in
`stats_utils.jl`; this file is responsible only for turning aggregated numbers
into figures.

Each `generate_*` function returns a `Plots.Plot` so callers can either save it
(via `save_figure`) or further customise it.
"""

using Plots
gr()

# Display names, branching-factor helpers, and aggregation primitives.
include(joinpath(@__DIR__, "stats_utils.jl"))

# ============================================================================
# Theme — change these constants in one place to restyle every figure
# ============================================================================

"""Half an A4 page (210mm wide) at ≈100 dpi."""
const FIGURE_SIZE             = (520, 400)

"""Pastel categorical palette for optimizers. Swap to taste."""
const FIGURE_PALETTE          = palette(:Pastel1)

const FIGURE_BACKGROUND       = :white
const FIGURE_GRID_COLOR       = :gray85
const FIGURE_FONT_FAMILY      = "Helvetica"
const FIGURE_GUIDE_FONT_SIZE  = 9
const FIGURE_TICK_FONT_SIZE   = 7
const FIGURE_LEGEND_FONT_SIZE = 7

"""Show ±1 std-dev error bars on bar charts."""
const FIGURE_SHOW_ERROR_BARS  = false

"""Common keyword arguments applied to every figure."""
function _base_attrs()
    return (
        size              = FIGURE_SIZE,
        background_color  = FIGURE_BACKGROUND,
        gridcolor         = FIGURE_GRID_COLOR,
        fontfamily        = FIGURE_FONT_FAMILY,
        guidefontsize     = FIGURE_GUIDE_FONT_SIZE,
        tickfontsize      = FIGURE_TICK_FONT_SIZE,
        legendfontsize    = FIGURE_LEGEND_FONT_SIZE,
        framestyle        = :box,
    )
end

"""One palette colour per optimizer (cycles if there are more optimizers than
palette entries)."""
function _optimizer_color(idx::Int)
    n = length(FIGURE_PALETTE)
    return FIGURE_PALETTE[mod1(idx, n)]
end

"""Plain-text version of `display_name` for plot labels — GR cannot render the
LaTeX dollar-math used by the table generators."""
figure_label(name) = replace(display_name(name), "\$" => "")

# ============================================================================
# Shared rendering helpers
# ============================================================================

"""
    _bar_per_optimizer(values::Dict, optimizer_order; ylabel, yscale, stds)

Render a bar chart with one bar per optimizer that has data. Optimizers are
plotted in `optimizer_order`; those whose value is `NaN` are skipped.

When `stds` is provided (a `Dict` matching `values`), error bars showing ±1
standard deviation are drawn on each bar.
"""
function _bar_per_optimizer(values::AbstractDict,
                            optimizer_order::AbstractVector;
                            ylabel::String,
                            yscale::Symbol=:identity,
                            stds::Union{AbstractDict,Nothing}=nothing)
    present = [opt for opt in optimizer_order if !isnan(values[opt])]
    isempty(present) && return plot(; _base_attrs()...)

    heights = [values[opt] for opt in present]
    labels  = [figure_label(opt) for opt in present]
    colors  = [_optimizer_color(i) for i in eachindex(present)]

    # Build error-bar vector: use the std when available, 0 otherwise.
    errs = if stds !== nothing
        [isnan(get(stds, opt, NaN)) ? 0.0 : stds[opt] for opt in present]
    else
        zeros(length(present))
    end

    # Add headroom above the tallest bar+error so it doesn't touch the frame.
    ymax = maximum(heights[i] + errs[i] for i in eachindex(heights))
    ylims_kw = if yscale == :log10
        ymin = minimum(h for h in heights if h > 0)
        (ymin * 0.5, ymax * 2.0)
    else
        (0, ymax * 1.1)
    end

    plt = bar(
        labels, heights;
        ylabel        = ylabel,
        xrotation     = 45,
        legend        = false,
        color         = colors,
        yscale        = yscale,
        ylims         = ylims_kw,
        bottom_margin = 12Plots.mm,
        _base_attrs()...
    )

    # Overlay error bars as a scatter series so they are centred on the mean.
    # Plots.jl bar charts use the same categorical labels as x-coordinates.
    # On log-scale axes, clamp the lower whisker so it never goes non-positive.
    if any(e -> e > 0, errs)
        whiskers = if yscale == :log10
            # Asymmetric error bars: (lower, upper) per point.
            [(min(e, h - h * 0.01), e) for (h, e) in zip(heights, errs)]
        else
            errs
        end
        scatter!(plt, labels, heights;
                 yerror      = whiskers,
                 markersize  = 0,
                 markercolor = :black,
                 linecolor   = :black,
                 label       = false)
    end

    return plt
end

"""
    _lines_by_branching_factor(series::Dict, optimizer_order; ylabel, title, yscale)

One line per optimizer; x-axis is branching factor, y-axis is the supplied
metric. Each entry in `series` is a sorted vector of `(branching_factor, value)`
pairs as produced by `mean_metric_by_branching_factor`.
"""
function _lines_by_branching_factor(series::AbstractDict,
                                    optimizer_order::AbstractVector;
                                    ylabel::String,
                                    yscale::Symbol=:identity)
    plt = plot(;
        xlabel = "Branching factor",
        ylabel = ylabel,
        xscale = :log10,
        yscale = yscale,
        legend = :outerright,
        _base_attrs()...
    )

    for (i, opt) in enumerate(optimizer_order)
        pts = get(series, opt, Tuple{Int,Float64}[])
        # Drop non-positive values — both axes are (or may be) log-scaled.
        pts = filter(p -> p[1] > 0 && p[2] > 0, pts)
        isempty(pts) && continue
        xs = Float64[p[1] for p in pts]
        ys = Float64[p[2] for p in pts]
        plot!(plt, xs, ys;
              label  = figure_label(opt),
              marker = :circle,
              color  = _optimizer_color(i),
              linewidth = 1.5,
              markersize = 4)
    end
    return plt
end

"""
    _lines_categorical(series::Dict, optimizer_order; xlabel, ylabel, yscale)

One line per optimizer; x-axis values from `series` are mapped to evenly-spaced
integer positions (categorical axis). Each entry in `series` is a sorted vector
of `(x_value, y_value)` pairs. Tick labels show the original x values.
"""
function _lines_categorical(series::AbstractDict,
                            optimizer_order::AbstractVector;
                            xlabel::String,
                            ylabel::String,
                            yscale::Symbol=:identity)
    # Collect all x values across optimizers
    all_xs = sort(collect(Set(
        p[1] for opt in optimizer_order
              for p in get(series, opt, Tuple{Int,Float64}[])
    )))
    x_to_idx = Dict(x => i for (i, x) in enumerate(all_xs))

    plt = plot(;
        xlabel = xlabel,
        ylabel = ylabel,
        yscale = yscale,
        legend = :outerright,
        xticks = (1:length(all_xs), string.(all_xs)),
        _base_attrs()...
    )

    for (i, opt) in enumerate(optimizer_order)
        pts = get(series, opt, Tuple{Int,Float64}[])
        isempty(pts) && continue
        xs = Float64[x_to_idx[p[1]] for p in pts]
        ys = Float64[p[2] for p in pts]
        plot!(plt, xs, ys;
              label  = figure_label(opt),
              marker = :circle,
              color  = _optimizer_color(i),
              linewidth = 1.5,
              markersize = 4)
    end
    return plt
end

# ============================================================================
# Per-experiment-suite plots
# ============================================================================

"""
    generate_ranking_plot_per_experiment(experiments, optimizer_order; title)

Bar plot of the average rank of each optimizer across one experiment suite.
Lower is better.
"""
function generate_ranking_plot_per_experiment(experiments,
                                              optimizer_order::AbstractVector;
                                              suite_name=nothing)
    if FIGURE_SHOW_ERROR_BARS
        means, stds = mean_and_std_metric_across_experiments(experiments, optimizer_order,
                          s -> optimizer_metric(s, "mean_rank"); suite_name=suite_name)
    else
        means = mean_metric_across_experiments(experiments, optimizer_order,
                    s -> optimizer_metric(s, "mean_rank"); suite_name=suite_name)
        stds = nothing
    end
    plt = _bar_per_optimizer(means, optimizer_order;
        ylabel = "Mean rank", stds = stds)
    ymax = ceil(Int, last(ylims(plt)))
    plot!(plt; yticks = 0:ymax)
    return plt
end

"""
    generate_average_final_loss(experiments, optimizer_order; title)

Bar plot of the mean final loss of each optimizer across one experiment suite.
For non-function-learning experiments, `mean_final_loss` is already on the
log_p scale so a linear y-axis is used; the values represent log_p(loss).
"""
function generate_average_final_loss(experiments,
                                     optimizer_order::AbstractVector;
                                     suite_name=nothing)
    if FIGURE_SHOW_ERROR_BARS
        means, stds = mean_and_std_metric_across_experiments(experiments, optimizer_order,
                          s -> optimizer_metric(s, "mean_final_loss"); suite_name=suite_name)
    else
        means = mean_metric_across_experiments(experiments, optimizer_order,
                    s -> optimizer_metric(s, "mean_final_loss"); suite_name=suite_name)
        stds = nothing
    end
    return _bar_per_optimizer(means, optimizer_order;
        ylabel = "log_p(mean final loss)",
        stds   = stds)
end

"""
    generate_number_of_evals_plot(experiments, optimizer_order; title)

Line plot — one line per optimizer — of mean function-evaluation count vs
branching factor. Y-axis is log-scaled because eval counts span many orders.
"""
function generate_number_of_evals_plot(experiments,
                                       optimizer_order::AbstractVector;
                                       suite_name=nothing)
    series = mean_metric_by_branching_factor(experiments, optimizer_order,
                s -> optimizer_metric(s, "mean_total_evals"); suite_name=suite_name)
    return _lines_by_branching_factor(series, optimizer_order;
        ylabel = "Mean function evaluations",
        yscale = :log10)
end

"""
    generate_times_plot(experiments, optimizer_order; title)

Line plot — one line per optimizer — of mean wall-clock runtime vs branching
factor.
"""
function generate_times_plot(experiments,
                             optimizer_order::AbstractVector;
                             suite_name=nothing)
    series = mean_metric_by_branching_factor(experiments, optimizer_order,
                s -> optimizer_metric(s, "mean_time"); suite_name=suite_name)
    return _lines_by_branching_factor(series, optimizer_order;
        ylabel = "Mean runtime (s)",
        yscale = :log10)
end

# ============================================================================
# Per-prime plots (metric vs dimension)
# ============================================================================

"""
    _lines_by_dimension(series::Dict, optimizer_order; ylabel, yscale)

One line per optimizer; x-axis is polydisc dimension, y-axis is the supplied
metric. Each entry in `series` is a sorted vector of `(dimension, value)`
pairs as produced by `mean_metric_by_dimension`.
"""
function _lines_by_dimension(series::AbstractDict,
                             optimizer_order::AbstractVector;
                             ylabel::String,
                             yscale::Symbol=:identity)
    plt = plot(;
        xlabel = "Dimension",
        ylabel = ylabel,
        yscale = yscale,
        legend = :outerright,
        _base_attrs()...
    )

    for (i, opt) in enumerate(optimizer_order)
        pts = get(series, opt, Tuple{Int,Float64}[])
        pts = filter(p -> p[2] > 0, pts)
        isempty(pts) && continue
        xs = Float64[p[1] for p in pts]
        ys = Float64[p[2] for p in pts]
        plot!(plt, xs, ys;
              label  = figure_label(opt),
              marker = :circle,
              color  = _optimizer_color(i),
              linewidth = 1.5,
              markersize = 4)
    end
    return plt
end

"""
    generate_evals_by_dimension(experiments, optimizer_order; prime, suite_name)

Line plot — one line per optimizer — of mean function-evaluation count vs
dimension, filtered to configs with the given `prime`.
"""
function generate_evals_by_dimension(experiments,
                                     optimizer_order::AbstractVector;
                                     prime::Int,
                                     suite_name=nothing)
    series = mean_metric_by_dimension(experiments, optimizer_order,
                s -> optimizer_metric(s, "mean_total_evals");
                prime=prime, suite_name=suite_name)
    return _lines_by_dimension(series, optimizer_order;
        ylabel = "Mean function evaluations",
        yscale = :log10)
end

"""
    generate_times_by_dimension(experiments, optimizer_order; prime, suite_name)

Line plot — one line per optimizer — of mean wall-clock runtime vs dimension,
filtered to configs with the given `prime`.
"""
function generate_times_by_dimension(experiments,
                                     optimizer_order::AbstractVector;
                                     prime::Int,
                                     suite_name=nothing)
    series = mean_metric_by_dimension(experiments, optimizer_order,
                s -> optimizer_metric(s, "mean_time");
                prime=prime, suite_name=suite_name)
    return _lines_by_dimension(series, optimizer_order;
        ylabel = "Mean runtime (s)",
        yscale = :log10)
end

# ============================================================================
# Loss vs prime / loss vs dimension (categorical x-axis)
# ============================================================================

"""
    generate_loss_vs_prime(experiments, optimizer_order; suite_name)

Line plot — one line per optimizer — of mean final loss vs prime. The y-axis
shows `mean_final_loss` which is already on the log_p scale (computed during
aggregation in `compute_aggregate_stats`). The x-axis is categorical (evenly
spaced primes).
"""
function generate_loss_vs_prime(experiments,
                                optimizer_order::AbstractVector;
                                suite_name=nothing)
    series = mean_metric_by_prime(experiments, optimizer_order,
                s -> optimizer_metric(s, "mean_final_loss");
                suite_name=suite_name)
    return _lines_categorical(series, optimizer_order;
        xlabel = "Prime", ylabel = "log_p(mean final loss)")
end

"""
    generate_loss_vs_dimension(experiments, optimizer_order; suite_name)

Line plot — one line per optimizer — of mean final loss vs dimension (averaged
across all primes). The y-axis shows `mean_final_loss` which is already on the
log_p scale (computed during aggregation in `compute_aggregate_stats`). The
x-axis is categorical (evenly spaced dimensions).
"""
function generate_loss_vs_dimension(experiments,
                                    optimizer_order::AbstractVector;
                                    suite_name=nothing)
    series = mean_metric_by_dimension_all_primes(experiments, optimizer_order,
                s -> optimizer_metric(s, "mean_final_loss");
                suite_name=suite_name)
    return _lines_categorical(series, optimizer_order;
        xlabel = "Dimension", ylabel = "log_p(mean final loss)")
end

# ============================================================================
# Cross-suite plot
# ============================================================================

"""
    generate_overall_ranking_plot(all_experiments, optimizer_order; title)

Bar plot of the mean rank of each optimizer across the *concatenation* of all
experiment suites' configurations.
"""
function generate_overall_ranking_plot(all_experiments,
                                       optimizer_order::AbstractVector;
                                       suite_name=nothing)
    if FIGURE_SHOW_ERROR_BARS
        means, stds = mean_and_std_metric_across_experiments(all_experiments, optimizer_order,
                          s -> optimizer_metric(s, "mean_rank"); suite_name=suite_name)
    else
        means = mean_metric_across_experiments(all_experiments, optimizer_order,
                    s -> optimizer_metric(s, "mean_rank"); suite_name=suite_name)
        stds = nothing
    end
    plt = _bar_per_optimizer(means, optimizer_order;
        ylabel = "Mean rank", stds = stds)
    ymax = ceil(Int, last(ylims(plt)))
    plot!(plt; yticks = 0:ymax)
    return plt
end

# ============================================================================
# I/O
# ============================================================================

"""
    save_figure(plt, path)

Create the parent directory if needed and save `plt` to `path`. Format is
inferred from the extension by Plots.jl.
"""
function save_figure(plt, path::String)
    mkpath(dirname(path))
    plot!(plt; size = FIGURE_SIZE)          # enforce consistent dimensions
    savefig(plt, path)
    println("✓ Wrote $path")
    return path
end
