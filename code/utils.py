import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simpy
from matplotlib.patches import Circle, FancyArrowPatch

def draw_queue_network(
    positions,
    edges,
    title,
    entry_nodes=None,
    exit_nodes=None,
    feedback_edges=None,
    figsize=(9, 4),
):
    entry_nodes = entry_nodes or []
    exit_nodes = exit_nodes or []
    feedback_edges = set(feedback_edges or [])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')

    xs = [x for x, y in positions.values()]
    ys = [y for x, y in positions.values()]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)

    for source, target in edges:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        is_feedback = (source, target) in feedback_edges
        color = 'crimson' if is_feedback else 'black'
        linestyle = '--' if is_feedback else '-'
        rad = -0.35 if is_feedback else 0.0

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle='-|>',
            mutation_scale=18,
            linewidth=2.0,
            color=color,
            linestyle=linestyle,
            shrinkA=24,
            shrinkB=24,
            connectionstyle=f'arc3,rad={rad}',
        )
        ax.add_patch(arrow)

    for node in entry_nodes:
        x, y = positions[node]
        arrow = FancyArrowPatch(
            (x - 0.9, y),
            (x - 0.25, y),
            arrowstyle='-|>',
            mutation_scale=18,
            linewidth=2.0,
            color='tab:blue',
        )
        ax.add_patch(arrow)
        ax.text(x - 1.05, y + 0.25, 'Arrivals', ha='center', fontsize=10, color='tab:blue')

    for node in exit_nodes:
        x, y = positions[node]
        arrow = FancyArrowPatch(
            (x + 0.25, y),
            (x + 0.9, y),
            arrowstyle='-|>',
            mutation_scale=18,
            linewidth=2.0,
            color='tab:green',
        )
        ax.add_patch(arrow)
        ax.text(x + 1.05, y + 0.25, 'Exit', ha='center', fontsize=10, color='tab:green')

    for node, (x, y) in positions.items():
        circle = Circle((x, y), 0.28, facecolor='white', edgecolor='black', linewidth=2.0, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, node, ha='center', va='center', fontsize=12, fontweight='bold', zorder=4)

    if feedback_edges:
        ax.plot([], [], color='crimson', linestyle='--', linewidth=2, label='feedback')
        ax.legend(loc='upper right')

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.show()


def plot_distribution_vs_samples(distribution, sample_size):
    values = distribution["sample"](sample_size)
    support = distribution["support"]
    density = distribution["density"](support)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(distribution["name"], fontsize=14)

    if distribution["kind"] == "discrete":
        axes[0].bar(support, density, color="tab:blue", alpha=0.75)
        bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1)
        axes[1].hist(values, bins=bins, density=True, color="tab:orange", alpha=0.7, edgecolor="white")
    else:
        axes[0].plot(support, density, color="tab:blue", linewidth=2.5)
        axes[1].hist(values, bins=80, density=True, color="tab:orange", alpha=0.7, edgecolor="white")
        axes[1].plot(support, density, color="black", linewidth=2, label="theoretical pdf")
        axes[1].legend()

    axes[0].set_title("Theoretical density / PMF")
    axes[1].set_title(f"Histogram of {sample_size:,} samples")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_original_vs_sample_means(distribution, mean_repetitions, mean_sample_size):
    values = distribution["sample"](mean_repetitions * mean_sample_size)
    means = values.reshape(mean_repetitions, mean_sample_size).mean(axis=1)
    support = distribution["support"]
    density = distribution["density"](support)

    mean_of_average = distribution["mean"]
    std_of_average = np.sqrt(distribution["variance"] / mean_sample_size)
    normal_support = np.linspace(means.min(), means.max(), 500)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(distribution["name"], fontsize=14)

    if distribution["kind"] == "discrete":
        axes[0].bar(support, density, color="tab:blue", alpha=0.75)
    else:
        axes[0].plot(support, density, color="tab:blue", linewidth=2.5)

    axes[1].hist(means, bins=70, density=True, color="tab:green", alpha=0.7, edgecolor="white")
    axes[1].plot(
        normal_support,
        normal_pdf(normal_support, mean_of_average, std_of_average),
        "k--",
        linewidth=2,
        label="normal approximation",
    )
    axes[1].legend()

    axes[0].set_title("Original distribution")
    axes[1].set_title(f"{mean_repetitions:,} averages of {mean_sample_size} samples")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


def normal_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def erlang_pdf(x, shape, rate):
    x = np.asarray(x)
    density = (rate ** shape) * (x ** (shape - 1)) * np.exp(-rate * x) / math.factorial(shape - 1)
    return np.where(x >= 0, density, 0.0)


def hyper_erlang_rates_for_mean(target_mean, probabilities, shapes, relative_means):
    target_mean = np.asarray(target_mean)

    if target_mean.size != 1:
        raise ValueError(
            f"target_mean must be a single scalar value, got shape {target_mean.shape}"
        )

    probabilities = np.asarray(probabilities, dtype=float)
    shapes = np.asarray(shapes, dtype=float)
    relative_means = np.asarray(relative_means, dtype=float)

    if not (probabilities.shape == shapes.shape == relative_means.shape):
        raise ValueError("probabilities, shapes, and relative_means must have the same shape")

    target_mean = float(target_mean)
    normalized_means = relative_means / np.dot(probabilities, relative_means)
    branch_means = target_mean * normalized_means
    return shapes / branch_means


def make_hyper_erlang_branches(target_mean, probabilities, shapes, relative_means):
    rates = hyper_erlang_rates_for_mean(
        target_mean=target_mean,
        probabilities=probabilities,
        shapes=shapes,
        relative_means=relative_means,
    )
    return [
        {"prob": probability, "shape": int(shape), "rate": rate}
        for probability, shape, rate in zip(probabilities, shapes, rates)
    ]


def hyper_erlang_pdf(x, target_mean=None, probabilities=None, shapes=None, relative_means=None, branches=None):
    if branches is None:
        if target_mean is None or probabilities is None or shapes is None or relative_means is None:
            raise ValueError(
                "Provide either branches, or target_mean with probabilities, shapes, and relative_means"
            )
        branches = make_hyper_erlang_branches(
            target_mean=target_mean,
            probabilities=probabilities,
            shapes=shapes,
            relative_means=relative_means,
        )

    density = np.zeros_like(np.asarray(x, dtype=float))
    for branch in branches:
        density += branch["prob"] * erlang_pdf(x, branch["shape"], branch["rate"])
    return density


def sample_hyper_erlang(size, branches, rng):
    probabilities = np.array([branch["prob"] for branch in branches], dtype=float)
    probabilities = probabilities / probabilities.sum()
    branch_indices = rng.choice(len(branches), size=size, p=probabilities)
    samples = np.empty(size)

    for index, branch in enumerate(branches):
        mask = branch_indices == index
        samples[mask] = rng.gamma(
            shape=branch["shape"],
            scale=1 / branch["rate"],
            size=mask.sum(),
        )

    return samples


def make_hyper_erlang_sampler(rng, target_mean, probabilities, shapes, relative_means):
    branches = make_hyper_erlang_branches(
        target_mean=target_mean,
        probabilities=probabilities,
        shapes=shapes,
        relative_means=relative_means,
    )

    def sampler():
        return sample_hyper_erlang(1, branches, rng)[0]

    return sampler


def queue_customer(env, server, customer_index, service_time_sampler, waiting_times, sojourn_times, warmup_customers):
    arrival_time = env.now

    with server.request() as request:
        yield request
        service_start_time = env.now
        service_time = service_time_sampler()
        yield env.timeout(service_time)

    if customer_index >= warmup_customers:
        waiting_times.append(service_start_time - arrival_time)
        sojourn_times.append(env.now - arrival_time)


def queue_arrivals(
    env,
    server,
    interarrival_time_sampler,
    service_time_sampler,
    waiting_times,
    sojourn_times,
    warmup_customers,
    total_customers,
):
    for customer_index in range(total_customers):
        env.process(
            queue_customer(
                env,
                server,
                customer_index,
                service_time_sampler,
                waiting_times,
                sojourn_times,
                warmup_customers,
            )
        )

        if customer_index < total_customers - 1:
            yield env.timeout(interarrival_time_sampler())


def simulate_mean_time_in_system(
    service_rate,
    seed,
    arrival_rate,
    warmup_customers,
    measured_customers,
    probabilities,
    shapes,
    relative_means,
):
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)

    interarrival_time_sampler = make_hyper_erlang_sampler(
        rng=rng,
        target_mean=1 / arrival_rate,
        probabilities=probabilities,
        shapes=shapes,
        relative_means=relative_means,
    )
    service_time_sampler = make_hyper_erlang_sampler(
        rng=rng,
        target_mean=1 / service_rate,
        probabilities=probabilities,
        shapes=shapes,
        relative_means=relative_means,
    )

    waiting_times = []
    sojourn_times = []
    total_customers = warmup_customers + measured_customers

    env.process(
        queue_arrivals(
            env,
            server,
            interarrival_time_sampler,
            service_time_sampler,
            waiting_times,
            sojourn_times,
            warmup_customers,
            total_customers,
        )
    )
    env.run()

    return np.mean(sojourn_times)


def simulate_server_count(
    server_count,
    seed,
    arrival_rate,
    service_rate,
    warmup_customers,
    measured_customers,
    probabilities,
    shapes,
    relative_means,
):
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=int(server_count))

    interarrival_time_sampler = make_hyper_erlang_sampler(
        rng=rng,
        target_mean=1 / arrival_rate,
        probabilities=probabilities,
        shapes=shapes,
        relative_means=relative_means,
    )
    service_time_sampler = make_hyper_erlang_sampler(
        rng=rng,
        target_mean=1 / service_rate,
        probabilities=probabilities,
        shapes=shapes,
        relative_means=relative_means,
    )

    waiting_times = []
    sojourn_times = []
    total_customers = warmup_customers + measured_customers

    env.process(
        queue_arrivals(
            env,
            server,
            interarrival_time_sampler,
            service_time_sampler,
            waiting_times,
            sojourn_times,
            warmup_customers,
            total_customers,
        )
    )
    env.run()

    return {
        "servers": int(server_count),
        "utilization": arrival_rate / (server_count * service_rate),
        "mean_waiting_time": np.mean(waiting_times),
        "mean_time_in_system": np.mean(sojourn_times),
    }


def _time_weighted_distribution(event_times, event_deltas, horizon):
    if horizon <= 0:
        return np.array([1.0])

    order = np.argsort(event_times, kind="mergesort")
    event_times = np.asarray(event_times, dtype=float)[order]
    event_deltas = np.asarray(event_deltas, dtype=int)[order]

    state = 0
    current_time = 0.0
    state_durations = {}
    index = 0

    while index < len(event_times):
        event_time = event_times[index]
        if event_time > horizon:
            break

        duration = event_time - current_time
        if duration > 0:
            state_durations[state] = state_durations.get(state, 0.0) + duration
            current_time = event_time

        while index < len(event_times) and event_times[index] == event_time:
            state += int(event_deltas[index])
            index += 1

    if current_time < horizon:
        state_durations[state] = state_durations.get(state, 0.0) + (horizon - current_time)

    max_state = max(state_durations, default=0)
    distribution = np.zeros(max_state + 1)
    for state_value, duration in state_durations.items():
        distribution[state_value] = duration / horizon

    return distribution


def simulate_single_server_queue_from_times(inter_arrival_times, service_times):
    inter_arrival_times = np.asarray(inter_arrival_times, dtype=float)
    service_times = np.asarray(service_times, dtype=float)

    if inter_arrival_times.ndim != 1 or service_times.ndim != 1:
        raise ValueError("inter_arrival_times and service_times must be one-dimensional arrays")
    if len(inter_arrival_times) != len(service_times):
        raise ValueError("inter_arrival_times and service_times must have the same length")
    if len(inter_arrival_times) == 0:
        return np.array([]), np.array([]), np.array([1.0]), np.array([1.0])
    if np.any(inter_arrival_times < 0) or np.any(service_times < 0):
        raise ValueError("inter_arrival_times and service_times must be non-negative")

    arrival_times = np.cumsum(inter_arrival_times)
    service_start_times = np.empty_like(arrival_times)
    departure_times = np.empty_like(arrival_times)

    previous_departure_time = 0.0
    for customer_index, (arrival_time, service_time) in enumerate(zip(arrival_times, service_times)):
        service_start_time = max(arrival_time, previous_departure_time)
        departure_time = service_start_time + service_time

        service_start_times[customer_index] = service_start_time
        departure_times[customer_index] = departure_time
        previous_departure_time = departure_time

    waiting_times = service_start_times - arrival_times
    sojourn_times = departure_times - arrival_times

    horizon = arrival_times[-1]
    event_times = np.concatenate([arrival_times, departure_times])
    event_deltas = np.concatenate([
        np.ones(len(arrival_times), dtype=int),
        -np.ones(len(departure_times), dtype=int),
    ])

    number_in_system_distribution = _time_weighted_distribution(
        event_times=event_times,
        event_deltas=event_deltas,
        horizon=horizon,
    )

    queue_length_distribution = np.zeros(max(len(number_in_system_distribution) - 1, 1))
    for number_in_system, probability in enumerate(number_in_system_distribution):
        queue_length = max(number_in_system - 1, 0)
        if queue_length >= len(queue_length_distribution):
            queue_length_distribution = np.pad(
                queue_length_distribution,
                (0, queue_length - len(queue_length_distribution) + 1),
            )
        queue_length_distribution[queue_length] += probability

    return (
        waiting_times,
        sojourn_times,
        queue_length_distribution,
        number_in_system_distribution,
    )


def _choose_transient_arrival_matrix_width(horizon, arrival_shape, arrival_scale):
    mean_inter_arrival = arrival_shape * arrival_scale
    expected_arrivals = horizon / mean_inter_arrival
    return max(32, int(np.ceil(expected_arrivals + 12 * np.sqrt(expected_arrivals) + 50)))


def simulate_transient_waiting_time_gamma_exponential_queue(
    num_replications,
    horizon,
    num_bins,
    seed=2026,
    arrival_shape=2,
    arrival_scale=1 / 2,
    service_scale=1 / 1.25,
):
    rng = np.random.default_rng(seed)
    bin_edges = np.linspace(0, horizon, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    matrix_width = _choose_transient_arrival_matrix_width(
        horizon=horizon,
        arrival_shape=arrival_shape,
        arrival_scale=arrival_scale,
    )
    inter_arrival_times = rng.gamma(
        shape=arrival_shape,
        scale=arrival_scale,
        size=(num_replications, matrix_width),
    )
    arrival_times = np.cumsum(inter_arrival_times, axis=1)

    while np.any(arrival_times[:, -1] <= horizon):
        extra_inter_arrival_times = rng.gamma(
            shape=arrival_shape,
            scale=arrival_scale,
            size=(num_replications, 32),
        )
        extra_arrival_times = arrival_times[:, [-1]] + np.cumsum(extra_inter_arrival_times, axis=1)
        arrival_times = np.concatenate([arrival_times, extra_arrival_times], axis=1)

    service_times = rng.exponential(scale=service_scale, size=arrival_times.shape)
    arrivals_inside_horizon = arrival_times <= horizon

    waiting_time_sum_by_replication_and_bin = np.zeros((num_replications, num_bins))
    waiting_time_count_by_replication_and_bin = np.zeros((num_replications, num_bins), dtype=int)
    overall_waiting_time_sum_by_replication = np.zeros(num_replications)
    overall_waiting_time_count_by_replication = np.zeros(num_replications, dtype=int)
    previous_departure_times = np.zeros(num_replications)
    replication_indices = np.arange(num_replications)

    for customer_index in range(arrival_times.shape[1]):
        active = arrivals_inside_horizon[:, customer_index]
        if not np.any(active):
            continue

        active_replications = replication_indices[active]
        active_arrival_times = arrival_times[active, customer_index]
        active_service_times = service_times[active, customer_index]

        service_start_times = np.maximum(active_arrival_times, previous_departure_times[active])
        waiting_times = service_start_times - active_arrival_times
        previous_departure_times[active] = service_start_times + active_service_times

        bin_ids = np.minimum((active_arrival_times / horizon * num_bins).astype(int), num_bins - 1)

        waiting_time_sum_by_replication_and_bin[active_replications, bin_ids] += waiting_times
        waiting_time_count_by_replication_and_bin[active_replications, bin_ids] += 1
        overall_waiting_time_sum_by_replication[active_replications] += waiting_times
        overall_waiting_time_count_by_replication[active_replications] += 1

    waiting_time_sum_by_bin = waiting_time_sum_by_replication_and_bin.sum(axis=0)
    waiting_time_count_by_bin = waiting_time_count_by_replication_and_bin.sum(axis=0)
    mean_waiting_time = np.divide(
        waiting_time_sum_by_bin,
        waiting_time_count_by_bin,
        out=np.full(num_bins, np.nan),
        where=waiting_time_count_by_bin > 0,
    )

    replication_mean_waiting_by_bin = np.divide(
        waiting_time_sum_by_replication_and_bin,
        waiting_time_count_by_replication_and_bin,
        out=np.full((num_replications, num_bins), np.nan),
        where=waiting_time_count_by_replication_and_bin > 0,
    )
    replication_count_by_bin = np.sum(~np.isnan(replication_mean_waiting_by_bin), axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        standard_error = np.nanstd(replication_mean_waiting_by_bin, axis=0, ddof=1) / np.sqrt(replication_count_by_bin)

    replication_overall_mean_waiting = np.divide(
        overall_waiting_time_sum_by_replication,
        overall_waiting_time_count_by_replication,
        out=np.full(num_replications, np.nan),
        where=overall_waiting_time_count_by_replication > 0,
    )

    return {
        "time": bin_centers,
        "bin_edges": bin_edges,
        "mean_waiting_time": mean_waiting_time,
        "standard_error": standard_error,
        "lower_95": mean_waiting_time - 1.96 * standard_error,
        "upper_95": mean_waiting_time + 1.96 * standard_error,
        "arrivals_per_bin": waiting_time_count_by_bin,
        "replications_per_bin": replication_count_by_bin,
        "replication_overall_mean_waiting": replication_overall_mean_waiting,
        "overall_mean_waiting": np.nanmean(replication_overall_mean_waiting),
        "pooled_overall_mean_waiting": overall_waiting_time_sum_by_replication.sum() / overall_waiting_time_count_by_replication.sum(),
    }


def estimate_convergence_time(
    time,
    values,
    relative_tolerance=0.05,
    absolute_tolerance=0.05,
    tail_fraction=0.2,
    consecutive_points=8,
    smoothing_window=5,
):
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = ~np.isnan(time) & ~np.isnan(values)
    time = time[valid]
    values = values[valid]

    if len(values) == 0:
        return np.nan, np.nan

    if smoothing_window > 1 and len(values) >= smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed_values = np.convolve(values, kernel, mode="same")
        half_window = smoothing_window // 2
        smoothed_values[:half_window] = values[:half_window]
        smoothed_values[-half_window:] = values[-half_window:]
    else:
        smoothed_values = values

    tail_size = max(consecutive_points, int(np.ceil(tail_fraction * len(smoothed_values))))
    tail_size = min(tail_size, len(smoothed_values))
    stationary_estimate = np.nanmean(smoothed_values[-tail_size:])
    tolerance = max(absolute_tolerance, relative_tolerance * abs(stationary_estimate))

    within_tolerance = np.abs(smoothed_values - stationary_estimate) <= tolerance

    for index in range(0, len(within_tolerance) - consecutive_points + 1):
        if np.all(within_tolerance[index:index + consecutive_points]):
            return time[index], stationary_estimate

    return np.nan, stationary_estimate

import numpy as np


def _stationary_ctmc(Q, tol=1e-10):
    """
    Stationary row-vector distribution beta of a CTMC generator Q:
        beta Q = 0, beta 1 = 1
    """
    Q = np.asarray(Q, dtype=float)
    k = Q.shape[0]

    A = Q.T.copy()
    b = np.zeros(k)

    A[-1, :] = 1.0
    b[-1] = 1.0

    x = np.linalg.solve(A, b)

    # Clean small numerical negatives
    x[np.logical_and(x < 0, x > -100 * tol)] = 0.0
    x = x / x.sum()

    return x


def map_ph1_steady_probs(
    D0,
    D1,
    alpha,
    S,
    n,
    tol=1e-12,
    max_iter=10_000,
    check=True,
    return_details=True,
):
    """
    Computes steady-state probabilities P(N=0),...,P(N=n)
    for a MAP/PH/1 queue.

    MAP representation:
        D0 : m x m matrix, transitions without arrivals
        D1 : m x m matrix, transitions with one arrival

    PH service-time representation:
        alpha : initial PH row-vector, length p
        S     : p x p transient generator matrix

    The system is a single-server FCFS MAP/PH/1 queue.

    Returns
    -------
    result["level_probs"][k] = P(N=k), for k=0,...,n.
    """

    D0 = np.asarray(D0, dtype=float)
    D1 = np.asarray(D1, dtype=float)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    S = np.asarray(S, dtype=float)

    if n < 0:
        raise ValueError("n must be nonnegative.")

    m = D0.shape[0]
    p = S.shape[0]

    if D0.shape != (m, m) or D1.shape != (m, m):
        raise ValueError("D0 and D1 must be square matrices with the same shape.")

    if S.shape != (p, p):
        raise ValueError("S must be square.")

    if alpha.shape[0] != p:
        raise ValueError("alpha must have length equal to S.shape[0].")

    one_m = np.ones(m)
    one_p = np.ones(p)

    if check:
        if not np.allclose((D0 + D1) @ one_m, 0.0, atol=1e-8):
            raise ValueError("Rows of D0 + D1 must sum to zero.")

        if not np.allclose(alpha.sum(), 1.0, atol=1e-8):
            raise ValueError("alpha must sum to one.")

        if np.any(alpha < -1e-12):
            raise ValueError("alpha must be nonnegative.")

        if np.any((-S @ one_p) < -1e-10):
            raise ValueError("PH exit rates -S @ 1 must be nonnegative.")

    # MAP stationary distribution at arbitrary time
    beta = _stationary_ctmc(D0 + D1)

    arrival_rate = float(beta @ D1 @ one_m)

    # PH mean service time: alpha (-S)^(-1) 1
    mean_service = float(alpha @ np.linalg.solve(-S, one_p))

    rho = arrival_rate * mean_service

    if rho >= 1.0:
        raise ValueError(f"The MAP/PH/1 queue is unstable: rho={rho:.6g} >= 1.")

    # PH absorption vector
    s = -S @ one_p

    I_m = np.eye(m)
    I_p = np.eye(p)

    d = m * p  # dimension of each positive level

    # ------------------------------------------------------------------
    # QBD construction
    #
    # Level 0 phase dimension: m
    # Level k>=1 phase dimension: m*p
    #
    # Positive-level phase ordering:
    #     state index = arrival_phase * p + service_phase
    # ------------------------------------------------------------------

    # Boundary blocks
    B00 = D0

    # Empty system -> one customer in service
    # Arrival changes MAP phase and initializes PH service phase by alpha
    B01 = np.kron(D1, alpha.reshape(1, p))

    # One customer completes service -> system becomes empty
    B10 = np.kron(I_m, s.reshape(p, 1))

    # Repeating blocks for levels k >= 1
    A_up = np.kron(D1, I_p)

    A_mid = np.kron(D0, I_p) + np.kron(I_m, S)

    # Service completion when queue remains nonempty:
    # level decreases by 1 and the next customer starts service with alpha
    A_down = np.kron(I_m, np.outer(s, alpha))

    # ------------------------------------------------------------------
    # Matrix-geometric solution
    #
    # For k >= 1:
    #     pi_{k+1} = pi_k R
    #
    # R solves:
    #     A_up + R A_mid + R^2 A_down = 0
    # ------------------------------------------------------------------

    R = np.zeros((d, d))

    for it in range(max_iter):
        M = A_mid + R @ A_down

        # Solve:
        #     R_next M = -A_up
        #
        # Avoid explicit inverse:
        #     M.T R_next.T = -A_up.T
        R_next = np.linalg.solve(M.T, -A_up.T).T

        # Clean tiny numerical negatives
        small_neg = np.logical_and(R_next < 0, R_next > -1000 * tol)
        R_next[small_neg] = 0.0

        err = np.max(np.abs(R_next - R))
        R = R_next

        if err < tol:
            break
    else:
        residual = np.linalg.norm(
            A_up + R @ A_mid + R @ R @ A_down,
            ord=np.inf,
        )
        raise RuntimeError(
            f"R iteration did not converge after {max_iter} iterations. "
            f"last change={err:.3e}, residual={residual:.3e}"
        )

    residual = np.linalg.norm(
        A_up + R @ A_mid + R @ R @ A_down,
        ord=np.inf,
    )

    spec_R = max(abs(np.linalg.eigvals(R)))

    if spec_R >= 1 + 1e-8:
        raise RuntimeError(f"Computed R has spectral radius >= 1: {spec_R:.6g}")

    # ------------------------------------------------------------------
    # Boundary equations for pi0 and pi1
    #
    # pi0 B00 + pi1 B10 = 0
    # pi0 B01 + pi1 (A_mid + R A_down) = 0
    #
    # Normalization:
    # pi0 1 + pi1 (I - R)^(-1) 1 = 1
    # ------------------------------------------------------------------

    K = np.zeros((m + d, m + d))

    K[:m, :m] = B00
    K[:m, m:] = B01
    K[m:, :m] = B10
    K[m:, m:] = A_mid + R @ A_down

    geom_weights = np.linalg.solve(np.eye(d) - R, np.ones(d))
    weights = np.r_[np.ones(m), geom_weights]

    A = K.T.copy()
    b = np.zeros(m + d)

    # Replace one balance equation with normalization
    A[-1, :] = weights
    b[-1] = 1.0

    x = np.linalg.solve(A, b)

    # Clean tiny numerical negatives
    x[np.logical_and(x < 0, x > -1000 * tol)] = 0.0

    pi0 = x[:m]
    pi1 = x[m:]

    # Renormalize using the infinite matrix-geometric tail
    norm = float(pi0.sum() + pi1 @ geom_weights)
    pi0 = pi0 / norm
    pi1 = pi1 / norm

    # Compute P(N=0),...,P(N=n)
    probs = np.zeros(n + 1)

    probs[0] = pi0.sum()

    level_vec = pi1.copy()

    for k in range(1, n + 1):
        probs[k] = level_vec.sum()
        level_vec = level_vec @ R

    probs[np.logical_and(probs < 0, probs > -1000 * tol)] = 0.0

    tail_after_n = max(0.0, 1.0 - float(probs.sum()))

    result = {
        "level_probs": probs,
        "tail_prob_after_n": tail_after_n,
        "rho": rho,
        "arrival_rate": arrival_rate,
        "mean_service": mean_service,
        "R": R,
        "pi0": pi0,
        "pi1": pi1,
        "R_residual_inf_norm": residual,
        "R_iterations": it + 1,
        "R_spectral_radius": float(spec_R),
    }

    return result if return_details else probs


import numpy as np


def repair_map_rows(D0, D1, tol=1e-8, max_allowed_error=3e-5, verbose=False):
    """
    Repairs a MAP representation by adjusting only diag(D0) so that
        rowsum(D0 + D1) = 0.

    Parameters
    ----------
    D0, D1 : ndarray
        MAP matrices.
    tol : float
        Numerical tolerance for considering row sums already zero.
    max_allowed_error : float
        If the largest row-sum error is larger than this, raise an error,
        because the MAP may not be merely suffering from numerical noise.
    verbose : bool
        Print correction diagnostics.

    Returns
    -------
    D0_fixed, D1_fixed
    """

    D0 = np.asarray(D0, dtype=float).copy()
    D1 = np.asarray(D1, dtype=float).copy()

    row_sums = (D0 + D1).sum(axis=1)
    max_err = np.max(np.abs(row_sums))

    if verbose:
        print("Max row-sum error before repair:", max_err)

    if max_err <= tol:
        if verbose:
            print("No repair needed.")
        return D0, D1

    if max_err > max_allowed_error:
        raise ValueError(
            f"Row-sum error is {max_err:.3e}, which is larger than "
            f"max_allowed_error={max_allowed_error:.3e}. "
            "This may indicate an invalid MAP, not just numerical noise."
        )

    # Correct only the diagonal of D0
    # New row sum:
    # old row sum - old row sum = 0
    for i in range(D0.shape[0]):
        D0[i, i] -= row_sums[i]

    row_sums_after = (D0 + D1).sum(axis=1)
    max_err_after = np.max(np.abs(row_sums_after))

    if verbose:
        print("Max row-sum error after repair:", max_err_after)

    return D0, D1


def probability_above_threshold(distribution, threshold):
    """Return P(X > threshold) from a discrete distribution array."""
    if len(distribution) <= threshold + 1:
        return 0.0
    return float(np.sum(distribution[threshold + 1:]))


def simulate_mm1_with_queue_function(arrival_rate, service_rate, num_customers, rng):
    """
    Generate M/M/1 inter-arrival and service times, then simulate only through
    simulate_single_server_queue_from_times.
    """
    if service_rate <= arrival_rate:
        raise ValueError("The queue is unstable when service_rate <= arrival_rate")

    inter_arrival_times = rng.exponential(scale=1 / arrival_rate, size=num_customers)
    service_times = rng.exponential(scale=1 / service_rate, size=num_customers)

    waiting_times, sojourn_times, queue_length_distribution, number_in_system_distribution = (
        simulate_single_server_queue_from_times(
            inter_arrival_times=inter_arrival_times,
            service_times=service_times,
        )
    )

    return {
        "inter_arrival_times": inter_arrival_times,
        "service_times": service_times,
        "waiting_times": waiting_times,
        "sojourn_times": sojourn_times,
        "queue_length_distribution": queue_length_distribution,
        "number_in_system_distribution": number_in_system_distribution,
    }


def summarize_simulation_outputs(simulation_outputs, queue_threshold=5, wait_threshold=3.0):
    """
    Compute performance measures from arrays returned by
    simulate_single_server_queue_from_times.
    """
    waiting_times = simulation_outputs["waiting_times"]
    sojourn_times = simulation_outputs["sojourn_times"]
    queue_length_distribution = simulation_outputs["queue_length_distribution"]
    number_in_system_distribution = simulation_outputs["number_in_system_distribution"]

    p_queue_above = probability_above_threshold(queue_length_distribution, queue_threshold)
    p_number_in_system_above = probability_above_threshold(
        number_in_system_distribution,
        queue_threshold,
    )

    return {
        "mean_waiting_time": float(np.mean(waiting_times)),
        "mean_sojourn_time": float(np.mean(sojourn_times)),
        "p_waiting_above_threshold": float(np.mean(waiting_times > wait_threshold)),
        "p_sojourn_above_threshold": float(np.mean(sojourn_times > wait_threshold)),
        "p_queue_above_threshold": p_queue_above,
        "p_queue_at_most_threshold": 1 - p_queue_above,
        "p_number_in_system_above_threshold": p_number_in_system_above,
        "max_queue_observed": len(queue_length_distribution) - 1,
        "max_number_in_system_observed": len(number_in_system_distribution) - 1,
    }


def run_queue_replications(
    arrival_rate,
    service_rate,
    num_customers=30_000,
    num_replications=5,
    seed=2026,
    queue_threshold=5,
    wait_threshold=3.0,
):
    """
    Run independent M/M/1 replications.

    The only queue simulator used here is simulate_single_server_queue_from_times.
    All probabilities and means are estimated from its returned arrays.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(num_replications):
        simulation_outputs = simulate_mm1_with_queue_function(
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            num_customers=num_customers,
            rng=rng,
        )
        rows.append(
            summarize_simulation_outputs(
                simulation_outputs,
                queue_threshold=queue_threshold,
                wait_threshold=wait_threshold,
            )
        )

    summary = {}
    for key in rows[0]:
        values = np.array([row[key] for row in rows], dtype=float)
        summary[key] = float(np.mean(values))
        summary[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    summary["arrival_rate"] = float(arrival_rate)
    summary["service_rate"] = float(service_rate)
    summary["mean_inter_arrival_time"] = float(1 / arrival_rate)
    summary["mean_service_time"] = float(1 / service_rate)
    summary["utilization"] = float(arrival_rate / service_rate)
    summary["num_customers"] = int(num_customers)
    summary["num_replications"] = int(num_replications)

    return summary


def evaluate_service_grid(arrival_rate, service_rates, **kwargs):
    """Evaluate a grid of service rates and return a list of summary dictionaries."""
    base_seed = kwargs.pop("seed", 2026)
    results = []
    for index, service_rate in enumerate(service_rates):
        if service_rate <= arrival_rate:
            continue
        result = run_queue_replications(
            arrival_rate=arrival_rate,
            service_rate=float(service_rate),
            seed=base_seed + index * 100,
            **kwargs,
        )
        results.append(result)
    return results


def results_to_dataframe(results, columns, decimals=4):
    """Convert selected simulation result columns into a notebook-friendly DataFrame."""
    df = pd.DataFrame(results)[columns].copy()
    float_columns = df.select_dtypes(include="number").columns
    df[float_columns] = df[float_columns].round(decimals)
    return df


def _service_time_for_server(service_times, customer_index, server_index):
    service_times = np.asarray(service_times, dtype=float)

    if service_times.ndim == 1:
        return float(service_times[customer_index])
    if service_times.ndim == 2:
        return float(service_times[customer_index, server_index])

    raise ValueError("service_times must be either a vector or a customer-by-server matrix")


def _choose_idle_server(idle_servers, service_times, customer_index, assignment_rule):
    if assignment_rule == "first_available":
        return min(idle_servers)
    if assignment_rule == "fastest_available":
        return min(
            idle_servers,
            key=lambda server_index: _service_time_for_server(
                service_times,
                customer_index,
                server_index,
            ),
        )

    raise ValueError(
        "assignment_rule must be either 'first_available' or 'fastest_available'"
    )


def simulate_multiserver_queue_with_abandonment_from_times(
    inter_arrival_times,
    service_times,
    patience_times,
    server_count,
    assignment_rule="fastest_available",
):
    """
    Simulate a FCFS multi-server queue with customer abandonment.

    Parameters
    ----------
    inter_arrival_times : array-like, shape (n_customers,)
        Time between consecutive arrivals.
    service_times : array-like, shape (n_customers,) or (n_customers, n_servers)
        If one-dimensional, all servers use the same realized service time for a
        customer. If two-dimensional, entry [i, j] is the service time of
        customer i on server j, allowing heterogeneous servers.
    patience_times : array-like, shape (n_customers,)
        Customer i abandons if waiting time is greater than patience_times[i].
    server_count : int
        Number of active servers. For matrix service_times, the first
        server_count columns are used.
    assignment_rule : {"fastest_available", "first_available"}
        Rule for choosing among currently idle servers.

    Returns
    -------
    dict
        Arrays and summary measures for served and abandoned customers.
    """
    inter_arrival_times = np.asarray(inter_arrival_times, dtype=float)
    service_times = np.asarray(service_times, dtype=float)
    patience_times = np.asarray(patience_times, dtype=float)
    server_count = int(server_count)

    if server_count < 1:
        raise ValueError("server_count must be at least 1")
    if inter_arrival_times.ndim != 1:
        raise ValueError("inter_arrival_times must be one-dimensional")
    if patience_times.ndim != 1:
        raise ValueError("patience_times must be one-dimensional")
    if len(inter_arrival_times) != len(patience_times):
        raise ValueError("inter_arrival_times and patience_times must have the same length")
    if service_times.ndim == 1 and len(service_times) != len(inter_arrival_times):
        raise ValueError("one-dimensional service_times must match the number of arrivals")
    if service_times.ndim == 2:
        if service_times.shape[0] != len(inter_arrival_times):
            raise ValueError("service_times rows must match the number of arrivals")
        if service_times.shape[1] < server_count:
            raise ValueError("service_times must have at least server_count columns")
    if service_times.ndim not in (1, 2):
        raise ValueError("service_times must be one- or two-dimensional")
    if np.any(inter_arrival_times < 0) or np.any(service_times < 0) or np.any(patience_times < 0):
        raise ValueError("interarrival, service, and patience times must be non-negative")

    num_customers = len(inter_arrival_times)
    arrival_times = np.cumsum(inter_arrival_times)

    server_available_times = np.zeros(server_count)
    server_busy = np.zeros(server_count, dtype=bool)
    service_counts_by_server = np.zeros(server_count, dtype=int)

    queue = deque()
    next_arrival_index = 0
    now = 0.0

    waiting_times = np.full(num_customers, np.nan)
    service_start_times = np.full(num_customers, np.nan)
    departure_times = np.full(num_customers, np.nan)
    abandonment_times = np.full(num_customers, np.nan)
    assigned_servers = np.full(num_customers, -1, dtype=int)
    served = np.zeros(num_customers, dtype=bool)
    abandoned = np.zeros(num_customers, dtype=bool)

    def start_service(customer_index, server_index, start_time):
        service_time = _service_time_for_server(service_times, customer_index, server_index)
        waiting_times[customer_index] = start_time - arrival_times[customer_index]
        service_start_times[customer_index] = start_time
        departure_times[customer_index] = start_time + service_time
        assigned_servers[customer_index] = server_index
        served[customer_index] = True
        server_busy[server_index] = True
        server_available_times[server_index] = departure_times[customer_index]
        service_counts_by_server[server_index] += 1

    def assign_waiting_customers_to_idle_servers():
        while queue and np.any(~server_busy):
            customer_index = queue.popleft()
            while arrival_times[customer_index] + patience_times[customer_index] < now:
                abandoned[customer_index] = True
                abandonment_times[customer_index] = arrival_times[customer_index] + patience_times[customer_index]
                if not queue:
                    return
                customer_index = queue.popleft()

            idle_servers = np.flatnonzero(~server_busy).tolist()
            server_index = _choose_idle_server(
                idle_servers,
                service_times,
                customer_index,
                assignment_rule,
            )
            start_service(customer_index, server_index, now)

    while next_arrival_index < num_customers or queue or np.any(server_busy):
        next_arrival_time = (
            arrival_times[next_arrival_index]
            if next_arrival_index < num_customers
            else np.inf
        )
        next_completion_time = (
            np.min(server_available_times[server_busy])
            if np.any(server_busy)
            else np.inf
        )

        if next_completion_time <= next_arrival_time:
            now = next_completion_time
            completed_servers = np.flatnonzero(server_busy & (server_available_times <= now))
            server_busy[completed_servers] = False
            assign_waiting_customers_to_idle_servers()
        else:
            now = next_arrival_time
            customer_index = next_arrival_index
            next_arrival_index += 1

            if np.any(~server_busy):
                idle_servers = np.flatnonzero(~server_busy).tolist()
                server_index = _choose_idle_server(
                    idle_servers,
                    service_times,
                    customer_index,
                    assignment_rule,
                )
                start_service(customer_index, server_index, now)
            else:
                queue.append(customer_index)

    waiting_times_for_served = waiting_times[served]
    sojourn_times_for_served = departure_times[served] - arrival_times[served]
    final_time = max(
        float(np.nanmax(departure_times)) if np.any(served) else 0.0,
        float(arrival_times[-1]) if num_customers > 0 else 0.0,
    )

    return {
        "arrival_times": arrival_times,
        "waiting_times": waiting_times,
        "waiting_times_for_served": waiting_times_for_served,
        "sojourn_times_for_served": sojourn_times_for_served,
        "service_start_times": service_start_times,
        "departure_times": departure_times,
        "abandonment_times": abandonment_times,
        "assigned_servers": assigned_servers,
        "served": served,
        "abandoned": abandoned,
        "server_count": server_count,
        "service_counts_by_server": service_counts_by_server,
        "num_arrivals": int(num_customers),
        "num_served": int(np.sum(served)),
        "num_abandoned": int(np.sum(abandoned)),
        "fraction_served": float(np.mean(served)) if num_customers > 0 else np.nan,
        "fraction_abandoned": float(np.mean(abandoned)) if num_customers > 0 else np.nan,
        "mean_waiting_time_served": float(np.mean(waiting_times_for_served)) if np.any(served) else np.nan,
        "mean_sojourn_time_served": float(np.mean(sojourn_times_for_served)) if np.any(served) else np.nan,
        "throughput_rate": float(np.sum(served) / final_time) if final_time > 0 else np.nan,
        "abandonment_rate": float(np.sum(abandoned) / final_time) if final_time > 0 else np.nan,
        "horizon": final_time,
    }


def evaluate_staffing_grid_with_abandonment(
    inter_arrival_times,
    service_times,
    patience_times,
    server_counts,
    server_cost_rate,
    abandonment_penalty,
    service_reward=0.0,
    assignment_rule="fastest_available",
):
    """Evaluate a staffing grid using common random numbers."""
    rows = []

    for server_count in server_counts:
        result = simulate_multiserver_queue_with_abandonment_from_times(
            inter_arrival_times=inter_arrival_times,
            service_times=service_times,
            patience_times=patience_times,
            server_count=server_count,
            assignment_rule=assignment_rule,
        )

        staffing_cost_rate = server_cost_rate * int(server_count)
        abandonment_cost_rate = abandonment_penalty * result["abandonment_rate"]
        reward_rate = service_reward * result["throughput_rate"]
        total_cost_rate = staffing_cost_rate + abandonment_cost_rate
        profit_rate = reward_rate - total_cost_rate

        rows.append(
            {
                "servers": int(server_count),
                "throughput_rate": result["throughput_rate"],
                "abandonment_rate": result["abandonment_rate"],
                "fraction_served": result["fraction_served"],
                "fraction_abandoned": result["fraction_abandoned"],
                "mean_waiting_time_served": result["mean_waiting_time_served"],
                "mean_sojourn_time_served": result["mean_sojourn_time_served"],
                "staffing_cost_rate": staffing_cost_rate,
                "abandonment_cost_rate": abandonment_cost_rate,
                "total_cost_rate": total_cost_rate,
                "reward_rate": reward_rate,
                "profit_rate": profit_rate,
                "num_arrivals": result["num_arrivals"],
                "num_served": result["num_served"],
                "num_abandoned": result["num_abandoned"],
            }
        )

    return pd.DataFrame(rows)


def evaluate_two_server_type_grid_with_abandonment(
    inter_arrival_times,
    slow_service_times,
    fast_service_times,
    patience_times,
    slow_server_counts,
    fast_server_counts,
    slow_server_cost_rate,
    fast_server_cost_rate,
    abandonment_penalty,
    max_total_servers=None,
    assignment_rule="first_available",
):
    """
    Evaluate all feasible mixes of slow/cheap and fast/expensive servers.

    slow_service_times and fast_service_times are matrices. Entry [i, j] is the
    realized service time of customer i if handled by server j of that type.
    Active fast-server columns are placed before slow-server columns, so the
    default first_available rule gives idle fast servers priority.
    """
    slow_service_times = np.asarray(slow_service_times, dtype=float)
    fast_service_times = np.asarray(fast_service_times, dtype=float)
    rows = []

    if slow_service_times.ndim != 2 or fast_service_times.ndim != 2:
        raise ValueError("slow_service_times and fast_service_times must be matrices")
    if slow_service_times.shape[0] != fast_service_times.shape[0]:
        raise ValueError("slow_service_times and fast_service_times must have the same number of rows")

    for fast_count in fast_server_counts:
        for slow_count in slow_server_counts:
            fast_count = int(fast_count)
            slow_count = int(slow_count)
            total_servers = fast_count + slow_count

            if total_servers == 0:
                continue
            if max_total_servers is not None and total_servers > max_total_servers:
                continue
            if fast_count > fast_service_times.shape[1]:
                raise ValueError("fast_count exceeds the number of fast service-time columns")
            if slow_count > slow_service_times.shape[1]:
                raise ValueError("slow_count exceeds the number of slow service-time columns")

            active_service_times = []
            if fast_count > 0:
                active_service_times.append(fast_service_times[:, :fast_count])
            if slow_count > 0:
                active_service_times.append(slow_service_times[:, :slow_count])
            active_service_times = np.column_stack(active_service_times)

            result = simulate_multiserver_queue_with_abandonment_from_times(
                inter_arrival_times=inter_arrival_times,
                service_times=active_service_times,
                patience_times=patience_times,
                server_count=total_servers,
                assignment_rule=assignment_rule,
            )

            staffing_cost_rate = (
                fast_count * fast_server_cost_rate
                + slow_count * slow_server_cost_rate
            )
            abandonment_cost_rate = abandonment_penalty * result["abandonment_rate"]
            total_cost_rate = staffing_cost_rate + abandonment_cost_rate

            rows.append(
                {
                    "fast_servers": fast_count,
                    "slow_servers": slow_count,
                    "total_servers": total_servers,
                    "throughput_rate": result["throughput_rate"],
                    "abandonment_rate": result["abandonment_rate"],
                    "fraction_abandoned": result["fraction_abandoned"],
                    "mean_waiting_time_served": result["mean_waiting_time_served"],
                    "fast_staffing_cost_rate": fast_count * fast_server_cost_rate,
                    "slow_staffing_cost_rate": slow_count * slow_server_cost_rate,
                    "staffing_cost_rate": staffing_cost_rate,
                    "abandonment_cost_rate": abandonment_cost_rate,
                    "total_cost_rate": total_cost_rate,
                    "num_arrivals": result["num_arrivals"],
                    "num_served": result["num_served"],
                    "num_abandoned": result["num_abandoned"],
                }
            )

    return pd.DataFrame(rows)


def make_hyper_erlang_10_branches(target_mean):
    """Create a 10-branch Hyper-Erlang distribution with the requested mean."""
    probabilities = np.ones(10) / 10
    shapes = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
    relative_means = np.array([0.35, 0.50, 0.70, 0.90, 1.05, 1.25, 1.50, 1.80, 2.20, 2.80])
    return make_hyper_erlang_branches(
        target_mean=target_mean,
        probabilities=probabilities,
        shapes=shapes,
        relative_means=relative_means,
    )


def generate_feed_forward_4_station_hyper_erlang_inputs(
    num_customers=20_000,
    seed=2026,
    arrival_rate=0.8,
    service_means=None,
    split_probability_to_s2=0.55,
):
    """Generate all inputs for the S1 -> {S2,S3} -> S4 example."""
    rng = np.random.default_rng(seed)
    station_names = ["S1", "S2", "S3", "S4"]

    if service_means is None:
        service_means = {
            "S1": 0.75,
            "S2": 0.80,
            "S3": 0.90,
            "S4": 0.70,
        }

    interarrival_branches = make_hyper_erlang_10_branches(target_mean=1 / arrival_rate)
    inter_arrival_times = sample_hyper_erlang(
        size=num_customers,
        branches=interarrival_branches,
        rng=rng,
    )

    service_times_by_station = {}
    service_branches_by_station = {}
    for station in station_names:
        branches = make_hyper_erlang_10_branches(target_mean=service_means[station])
        service_branches_by_station[station] = branches
        service_times_by_station[station] = sample_hyper_erlang(
            size=num_customers,
            branches=branches,
            rng=rng,
        )

    routing_matrix = pd.DataFrame(
        0.0,
        index=station_names,
        columns=station_names + ["EXIT"],
    )
    routing_matrix.loc["S1", "S2"] = split_probability_to_s2
    routing_matrix.loc["S1", "S3"] = 1 - split_probability_to_s2
    routing_matrix.loc["S2", "S4"] = 1.0
    routing_matrix.loc["S3", "S4"] = 1.0
    routing_matrix.loc["S4", "EXIT"] = 1.0

    expected_arrival_rates = {
        "S1": arrival_rate,
        "S2": arrival_rate * split_probability_to_s2,
        "S3": arrival_rate * (1 - split_probability_to_s2),
        "S4": arrival_rate,
    }
    utilization = {
        station: expected_arrival_rates[station] * service_means[station]
        for station in station_names
    }

    return {
        "station_names": station_names,
        "inter_arrival_times": inter_arrival_times,
        "service_times_by_station": service_times_by_station,
        "routing_matrix": routing_matrix,
        "arrival_branches": interarrival_branches,
        "service_branches_by_station": service_branches_by_station,
        "arrival_rate": arrival_rate,
        "service_means": service_means,
        "expected_arrival_rates": expected_arrival_rates,
        "utilization": utilization,
    }


def _time_weighted_distribution_between(event_times, event_deltas, start_time, end_time):
    event_times = np.asarray(event_times, dtype=float)
    event_deltas = np.asarray(event_deltas, dtype=int)

    if end_time <= start_time:
        return np.array([1.0])

    if len(event_times) == 0:
        return np.array([1.0])

    order = np.argsort(event_times, kind="mergesort")
    event_times = event_times[order]
    event_deltas = event_deltas[order]

    state = int(event_deltas[event_times <= start_time].sum())
    current_time = float(start_time)
    state_durations = {}
    index = int(np.searchsorted(event_times, start_time, side="right"))

    while index < len(event_times):
        event_time = float(event_times[index])
        if event_time > end_time:
            break

        duration = event_time - current_time
        if duration > 0:
            state_durations[state] = state_durations.get(state, 0.0) + duration
            current_time = event_time

        while index < len(event_times) and event_times[index] == event_time:
            state += int(event_deltas[index])
            index += 1

    if current_time < end_time:
        state_durations[state] = state_durations.get(state, 0.0) + (end_time - current_time)

    max_state = max(state_durations, default=0)
    distribution = np.zeros(max_state + 1)
    horizon = end_time - start_time
    for state_value, duration in state_durations.items():
        if state_value >= 0:
            distribution[state_value] = duration / horizon

    return distribution


def _validate_routing_matrix(routing_matrix, station_names):
    if isinstance(routing_matrix, pd.DataFrame):
        matrix = routing_matrix.loc[station_names, station_names + ["EXIT"]].copy()
    else:
        matrix = pd.DataFrame(
            routing_matrix,
            index=station_names,
            columns=station_names + ["EXIT"],
        )

    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums.to_numpy(dtype=float), 1.0):
        raise ValueError("Each row of the routing matrix must sum to 1")
    if (matrix < 0).any().any():
        raise ValueError("Routing probabilities must be non-negative")

    return matrix


def simulate_queue_network_from_times(
    inter_arrival_times,
    service_times_by_station,
    routing_matrix,
    station_names=None,
    seed=2026,
    capacity_by_station=None,
    warmup_fraction=0.1,
):
    """Simulate a queueing network from pre-generated arrival and service times."""
    rng = np.random.default_rng(seed)
    inter_arrival_times = np.asarray(inter_arrival_times, dtype=float)
    if not 0 <= warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")

    if station_names is None:
        station_names = list(service_times_by_station.keys())
    station_names = list(station_names)

    routing_matrix = _validate_routing_matrix(routing_matrix, station_names)
    capacity_by_station = capacity_by_station or {station: 1 for station in station_names}

    env = simpy.Environment()
    resources = {
        station: simpy.Resource(env, capacity=int(capacity_by_station.get(station, 1)))
        for station in station_names
    }

    service_times_by_station = {
        station: np.asarray(service_times_by_station[station], dtype=float)
        for station in station_names
    }
    service_indices = {station: 0 for station in station_names}
    queue_event_times = {station: [] for station in station_names}
    queue_event_deltas = {station: [] for station in station_names}
    waiting_time_records = []
    route_records = []

    def next_service_time(station):
        index = service_indices[station]
        if index >= len(service_times_by_station[station]):
            raise RuntimeError(f"Not enough pre-generated service times for {station}")
        service_indices[station] += 1
        return float(service_times_by_station[station][index])

    def choose_next_station(station):
        probabilities = routing_matrix.loc[station].to_numpy(dtype=float)
        destinations = routing_matrix.columns.to_list()
        return rng.choice(destinations, p=probabilities)

    def visit_station(customer_id, station):
        arrival_time = env.now
        resource = resources[station]
        joined_queue = resource.count >= resource.capacity

        if joined_queue:
            queue_event_times[station].append(env.now)
            queue_event_deltas[station].append(1)

        with resource.request() as request:
            yield request

            service_start_time = env.now
            waiting_time = service_start_time - arrival_time

            if joined_queue:
                queue_event_times[station].append(env.now)
                queue_event_deltas[station].append(-1)

            service_time = next_service_time(station)
            yield env.timeout(service_time)

        departure_time = env.now
        waiting_time_records.append(
            {
                "customer_id": customer_id,
                "station": station,
                "arrival_time": arrival_time,
                "service_start_time": service_start_time,
                "departure_time": departure_time,
                "waiting_time": waiting_time,
                "service_time": service_time,
            }
        )

        next_station = choose_next_station(station)
        route_records.append(
            {
                "customer_id": customer_id,
                "from_station": station,
                "to_station": next_station,
                "time": departure_time,
            }
        )
        return next_station

    def customer_process(customer_id):
        station = station_names[0]
        while station != "EXIT":
            station = yield env.process(visit_station(customer_id, station))

    def external_arrivals():
        for customer_id, inter_arrival_time in enumerate(inter_arrival_times):
            yield env.timeout(float(inter_arrival_time))
            env.process(customer_process(customer_id))

    env.process(external_arrivals())
    env.run()

    horizon = float(env.now)
    warmup_time = float(warmup_fraction * horizon)
    warmup_details = {
        "warmup_fraction": float(warmup_fraction),
    }

    waiting_df = pd.DataFrame(waiting_time_records)
    route_df = pd.DataFrame(route_records)
    stationary_waiting_df = waiting_df[waiting_df["arrival_time"] >= warmup_time].copy()

    waiting_times_by_station = {}
    queue_length_distributions = {}
    summary_rows = []

    for station in station_names:
        station_waits = stationary_waiting_df.loc[
            stationary_waiting_df["station"] == station,
            "waiting_time",
        ].to_numpy()
        waiting_times_by_station[station] = station_waits

        distribution = _time_weighted_distribution_between(
            event_times=queue_event_times[station],
            event_deltas=queue_event_deltas[station],
            start_time=warmup_time,
            end_time=horizon,
        )
        queue_length_distributions[station] = distribution

        summary_rows.append(
            {
                "station": station,
                "num_visits_after_warmup": int(len(station_waits)),
                "mean_waiting_time": float(np.mean(station_waits)) if len(station_waits) else np.nan,
                "p_wait_positive": float(np.mean(station_waits > 0)) if len(station_waits) else np.nan,
                "mean_queue_length": float(np.dot(np.arange(len(distribution)), distribution)),
                "p_queue_empty": float(distribution[0]) if len(distribution) else np.nan,
                "max_queue_length_observed": int(len(distribution) - 1),
            }
        )

    return {
        "station_names": station_names,
        "routing_matrix": routing_matrix,
        "horizon": horizon,
        "warmup_time": warmup_time,
        "warmup_details": warmup_details,
        "waiting_time_records": waiting_df,
        "stationary_waiting_time_records": stationary_waiting_df,
        "route_records": route_df,
        "waiting_times_by_station": waiting_times_by_station,
        "queue_length_distributions": queue_length_distributions,
        "summary": pd.DataFrame(summary_rows),
        "service_counts_by_station": service_indices,
    }


def plot_network_input_densities(input_data, bins=70):
    """Plot empirical and theoretical densities for arrival and service inputs."""
    plots = [
        (
            "External inter-arrival to S1",
            input_data["inter_arrival_times"],
            input_data["arrival_branches"],
        )
    ]
    for station in input_data["station_names"]:
        plots.append(
            (
                f"Service time {station}",
                input_data["service_times_by_station"][station],
                input_data["service_branches_by_station"][station],
            )
        )

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.ravel()

    for ax, (title, values, branches) in zip(axes, plots):
        values = np.asarray(values, dtype=float)
        support = np.linspace(0, np.quantile(values, 0.995), 500)
        ax.hist(values, bins=bins, density=True, alpha=0.55, color="tab:blue", label="sample")
        ax.plot(support, hyper_erlang_pdf(support, branches=branches), color="black", linewidth=2, label="pdf")
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend()

    for ax in axes[len(plots):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_network_waiting_time_histograms(result, bins=60):
    station_names = result["station_names"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()

    for ax, station in zip(axes, station_names):
        waits = result["waiting_times_by_station"][station]
        ax.hist(waits, bins=bins, density=True, color="tab:orange", alpha=0.7, edgecolor="white")
        ax.set_title(f"Waiting times after warmup - {station}")
        ax.set_xlabel("waiting time")
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_network_queue_length_distributions(result):
    station_names = result["station_names"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()

    for ax, station in zip(axes, station_names):
        distribution = result["queue_length_distributions"][station]
        ax.bar(np.arange(len(distribution)), distribution, color="tab:green", alpha=0.75)
        ax.set_title(f"Queue length distribution after warmup - {station}")
        ax.set_xlabel("queue length")
        ax.set_ylabel("time fraction")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


def network_warmup_summary(result):
    """Return a one-row DataFrame describing the fixed warmup deletion."""
    return pd.DataFrame(
        [
            {
                "horizon": result["horizon"],
                "warmup_time": result["warmup_time"],
                "warmup_fraction": result["warmup_time"] / result["horizon"] if result["horizon"] > 0 else np.nan,
            }
        ]
    )


def queue_length_distributions_to_dataframe(result):
    """Convert station queue-length distributions to a rectangular DataFrame."""
    rows = []
    for station, distribution in result["queue_length_distributions"].items():
        for queue_length, probability in enumerate(distribution):
            rows.append(
                {
                    "station": station,
                    "queue_length": queue_length,
                    "time_fraction": probability,
                }
            )
    return pd.DataFrame(rows)


def feed_forward_input_summary(input_data):
    """Return expected arrival rates, mean service times, and utilizations."""
    return pd.DataFrame(
        [
            {
                "station": station,
                "expected_arrival_rate": input_data["expected_arrival_rates"][station],
                "mean_service_time": input_data["service_means"][station],
                "utilization": input_data["utilization"][station],
            }
            for station in input_data["station_names"]
        ]
    )


def run_feed_forward_4_station_simpy_example(
    num_customers=20_000,
    seed=2026,
    arrival_rate=0.8,
    split_probability_to_s2=0.55,
    warmup_fraction=0.1,
):
    """Generate inputs and simulate the 4-station split-merge queueing network."""
    input_data = generate_feed_forward_4_station_hyper_erlang_inputs(
        num_customers=num_customers,
        seed=seed,
        arrival_rate=arrival_rate,
        split_probability_to_s2=split_probability_to_s2,
    )
    result = simulate_queue_network_from_times(
        inter_arrival_times=input_data["inter_arrival_times"],
        service_times_by_station=input_data["service_times_by_station"],
        routing_matrix=input_data["routing_matrix"],
        station_names=input_data["station_names"],
        seed=seed + 1,
        warmup_fraction=warmup_fraction,
    )
    return input_data, result


def solve_feedback_network_utilization(
    external_arrival_rates,
    routing_matrix,
    service_rates,
    servers=None,
    station_names=None,
):
    """Solve traffic equations and compute utilization for an open feedback network."""
    if isinstance(external_arrival_rates, pd.Series):
        if station_names is None:
            station_names = list(external_arrival_rates.index)
        external_arrival_rates = external_arrival_rates.loc[station_names].to_numpy(dtype=float)
    else:
        external_arrival_rates = np.asarray(external_arrival_rates, dtype=float)

    if station_names is None:
        if isinstance(routing_matrix, pd.DataFrame):
            station_names = list(routing_matrix.index)
        else:
            station_names = [f"S{i + 1}" for i in range(len(external_arrival_rates))]
    station_names = list(station_names)

    if isinstance(routing_matrix, pd.DataFrame):
        routing = routing_matrix.loc[station_names, station_names].to_numpy(dtype=float)
    else:
        routing = np.asarray(routing_matrix, dtype=float)

    if isinstance(service_rates, pd.Series):
        service_rates = service_rates.loc[station_names].to_numpy(dtype=float)
    else:
        service_rates = np.asarray(service_rates, dtype=float)

    if servers is None:
        servers = np.ones(len(station_names), dtype=float)
    elif isinstance(servers, pd.Series):
        servers = servers.loc[station_names].to_numpy(dtype=float)
    else:
        servers = np.asarray(servers, dtype=float)

    n = len(station_names)
    if routing.shape != (n, n):
        raise ValueError("routing_matrix must be an n by n matrix over the stations")
    if len(external_arrival_rates) != n or len(service_rates) != n or len(servers) != n:
        raise ValueError("arrival rates, service rates, servers, and station_names must have the same length")
    if np.any(external_arrival_rates < 0):
        raise ValueError("external arrival rates must be non-negative")
    if np.any(service_rates <= 0):
        raise ValueError("service rates must be positive")
    if np.any(servers <= 0):
        raise ValueError("servers must be positive")
    if np.any(routing < 0):
        raise ValueError("routing probabilities must be non-negative")
    if np.any(routing.sum(axis=1) >= 1):
        raise ValueError("each station must have positive probability of leaving the network")

    effective_arrival_rates = np.linalg.solve(
        np.eye(n) - routing.T,
        external_arrival_rates,
    )
    utilizations = effective_arrival_rates / (servers * service_rates)

    summary = pd.DataFrame(
        {
            "station": station_names,
            "external_arrival_rate": external_arrival_rates,
            "effective_arrival_rate": effective_arrival_rates,
            "service_rate": service_rates,
            "servers": servers,
            "utilization": utilizations,
            "stable": utilizations < 1,
        }
    )

    routing_df = pd.DataFrame(routing, index=station_names, columns=station_names)
    routing_df["EXIT"] = 1 - routing_df.sum(axis=1)

    return {
        "station_names": station_names,
        "routing_matrix": routing_df,
        "external_arrival_rates": pd.Series(external_arrival_rates, index=station_names),
        "service_rates": pd.Series(service_rates, index=station_names),
        "servers": pd.Series(servers, index=station_names),
        "effective_arrival_rates": pd.Series(effective_arrival_rates, index=station_names),
        "summary": summary,
        "stable": bool(np.all(utilizations < 1)),
    }


def feedback_loop_utilization_examples():
    """Return stable and unstable examples for a three-station feedback network."""
    station_names = ["S1", "S2", "S3"]
    routing_matrix = pd.DataFrame(
        [
            [0.00, 0.60, 0.00],
            [0.25, 0.00, 0.55],
            [0.20, 0.00, 0.00],
        ],
        index=station_names,
        columns=station_names,
    )
    external_arrival_rates = pd.Series(
        [0.55, 0.10, 0.00],
        index=station_names,
    )

    stable = solve_feedback_network_utilization(
        external_arrival_rates=external_arrival_rates,
        routing_matrix=routing_matrix,
        service_rates=pd.Series([1.20, 1.10, 0.90], index=station_names),
        station_names=station_names,
    )
    unstable = solve_feedback_network_utilization(
        external_arrival_rates=external_arrival_rates,
        routing_matrix=routing_matrix,
        service_rates=pd.Series([0.70, 0.45, 0.60], index=station_names),
        station_names=station_names,
    )

    return {
        "stable": stable,
        "unstable": unstable,
    }


def simulate_feedback_loop_waiting_times(
    external_arrival_rates,
    routing_matrix,
    service_rates,
    horizon=4_000,
    seed=2026,
    station_names=None,
):
    """Simulate an open feedback network and record waiting times by station visit."""
    rng = np.random.default_rng(seed)

    if isinstance(external_arrival_rates, pd.Series):
        if station_names is None:
            station_names = list(external_arrival_rates.index)
        external_arrival_rates = external_arrival_rates.loc[station_names].to_numpy(dtype=float)
    else:
        external_arrival_rates = np.asarray(external_arrival_rates, dtype=float)

    if station_names is None:
        if isinstance(routing_matrix, pd.DataFrame):
            station_names = list(routing_matrix.index)
        else:
            station_names = [f"S{i + 1}" for i in range(len(external_arrival_rates))]
    station_names = list(station_names)

    if isinstance(routing_matrix, pd.DataFrame):
        routing = routing_matrix.loc[station_names, station_names].to_numpy(dtype=float)
    else:
        routing = np.asarray(routing_matrix, dtype=float)

    if isinstance(service_rates, pd.Series):
        service_rates = service_rates.loc[station_names].to_numpy(dtype=float)
    else:
        service_rates = np.asarray(service_rates, dtype=float)

    if np.any(routing.sum(axis=1) >= 1):
        raise ValueError("each station must have positive probability of leaving the network")

    env = simpy.Environment()
    resources = {station: simpy.Resource(env, capacity=1) for station in station_names}
    waiting_records = []
    route_records = []
    customer_counter = {"value": 0}

    destinations = station_names + ["EXIT"]
    routing_with_exit = np.column_stack([routing, 1 - routing.sum(axis=1)])

    def choose_next_station(station):
        station_index = station_names.index(station)
        return rng.choice(destinations, p=routing_with_exit[station_index])

    def visit_station(customer_id, station):
        arrival_time = env.now
        with resources[station].request() as request:
            yield request
            service_start_time = env.now
            waiting_time = service_start_time - arrival_time
            waiting_records.append(
                {
                    "customer_id": customer_id,
                    "station": station,
                    "arrival_time": arrival_time,
                    "service_start_time": service_start_time,
                    "waiting_time": waiting_time,
                }
            )
            station_index = station_names.index(station)
            service_time = rng.exponential(1 / service_rates[station_index])
            yield env.timeout(service_time)

        next_station = choose_next_station(station)
        route_records.append(
            {
                "customer_id": customer_id,
                "from_station": station,
                "to_station": next_station,
                "time": env.now,
            }
        )
        return next_station

    def customer_process(customer_id, first_station):
        station = first_station
        while station != "EXIT":
            station = yield env.process(visit_station(customer_id, station))

    def external_arrival_process(station, rate):
        if rate <= 0:
            return
        while True:
            inter_arrival_time = rng.exponential(1 / rate)
            if env.now + inter_arrival_time > horizon:
                break
            yield env.timeout(inter_arrival_time)
            customer_id = customer_counter["value"]
            customer_counter["value"] += 1
            env.process(customer_process(customer_id, station))

    for station, rate in zip(station_names, external_arrival_rates):
        env.process(external_arrival_process(station, rate))

    env.run()

    return {
        "station_names": station_names,
        "horizon": float(horizon),
        "waiting_records": pd.DataFrame(waiting_records),
        "route_records": pd.DataFrame(route_records),
        "num_external_customers": customer_counter["value"],
        "end_time_after_draining": float(env.now),
    }


def waiting_time_mean_by_arrival_time(simulation_result, num_bins=50):
    """Compute mean waiting time in equal-width bins of arrival time."""
    waiting_df = simulation_result["waiting_records"].copy()
    horizon = simulation_result["horizon"]
    bin_edges = np.linspace(0, horizon, num_bins + 1)
    waiting_df = waiting_df[
        (waiting_df["arrival_time"] >= 0)
        & (waiting_df["arrival_time"] <= horizon)
    ].copy()
    waiting_df["time_bin"] = np.minimum(
        np.searchsorted(bin_edges, waiting_df["arrival_time"], side="right") - 1,
        num_bins - 1,
    )
    waiting_df = waiting_df[(waiting_df["time_bin"] >= 0) & (waiting_df["time_bin"] < num_bins)]

    summary = (
        waiting_df.groupby("time_bin")
        .agg(
            mean_waiting_time=("waiting_time", "mean"),
            num_visits=("waiting_time", "size"),
        )
        .reindex(range(num_bins))
        .reset_index()
    )
    summary["time"] = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    summary["total_waiting_time"] = summary["mean_waiting_time"] * summary["num_visits"]
    summary["cumulative_waiting_time"] = summary["total_waiting_time"].fillna(0).cumsum()
    summary["cumulative_visits"] = summary["num_visits"].fillna(0).cumsum()
    summary["cumulative_mean_waiting_time"] = (
        summary["cumulative_waiting_time"] / summary["cumulative_visits"]
    )
    return summary[
        [
            "time",
            "mean_waiting_time",
            "cumulative_mean_waiting_time",
            "num_visits",
            "cumulative_visits",
        ]
    ]


def simulate_feedback_stability_comparison(horizon=4_000, seed=2026, num_bins=50):
    """Run stable and unstable feedback simulations and summarize waiting over time."""
    examples = feedback_loop_utilization_examples()
    stable_input = examples["stable"]
    unstable_input = examples["unstable"]

    stable_simulation = simulate_feedback_loop_waiting_times(
        external_arrival_rates=stable_input["external_arrival_rates"],
        routing_matrix=stable_input["routing_matrix"].drop(columns=["EXIT"]),
        service_rates=stable_input["service_rates"],
        horizon=horizon,
        seed=seed,
        station_names=stable_input["station_names"],
    )
    unstable_simulation = simulate_feedback_loop_waiting_times(
        external_arrival_rates=unstable_input["external_arrival_rates"],
        routing_matrix=unstable_input["routing_matrix"].drop(columns=["EXIT"]),
        service_rates=unstable_input["service_rates"],
        horizon=horizon,
        seed=seed + 1,
        station_names=unstable_input["station_names"],
    )

    stable_waiting_by_time = waiting_time_mean_by_arrival_time(
        stable_simulation,
        num_bins=num_bins,
    )
    unstable_waiting_by_time = waiting_time_mean_by_arrival_time(
        unstable_simulation,
        num_bins=num_bins,
    )
    stable_waiting_by_time["system"] = "stable"
    unstable_waiting_by_time["system"] = "unstable"

    return {
        "stable": stable_simulation,
        "unstable": unstable_simulation,
        "waiting_by_time": pd.concat(
            [stable_waiting_by_time, unstable_waiting_by_time],
            ignore_index=True,
        ),
    }


def plot_feedback_waiting_time_over_time(comparison_result):
    """Plot cumulative mean waiting time for stable and unstable feedback networks."""
    waiting_by_time = comparison_result["waiting_by_time"]
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {
        "stable": "tab:blue",
        "unstable": "tab:red",
    }
    labels = {
        "stable": "Stable system",
        "unstable": "Unstable system",
    }

    for system_name, system_df in waiting_by_time.groupby("system"):
        ax.plot(
            system_df["time"],
            system_df["cumulative_mean_waiting_time"],
            marker="o",
            linewidth=2,
            markersize=4,
            color=colors.get(system_name),
            label=labels.get(system_name, system_name),
        )

    ax.set_title("Cumulative mean waiting time as a function of simulation time")
    ax.set_xlabel("arrival time")
    ax.set_ylabel("cumulative mean waiting time")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


def mm1_mean_time_in_system(arrival_rate, service_rate):
    """Mean time in an M/M/1 station, including waiting and service."""
    if arrival_rate >= service_rate:
        return np.inf
    return 1 / (service_rate - arrival_rate)


def braess_queue_paradox_results(
    arrival_rate=1.0,
    congested_service_rate=1.4,
    fixed_delay=2.6,
    shortcut_delay=0.05,
):
    """Compute route delays for a queueing-network Braess example."""
    no_shortcut_flow = arrival_rate / 2
    no_shortcut_queue_delay = mm1_mean_time_in_system(
        arrival_rate=no_shortcut_flow,
        service_rate=congested_service_rate,
    )
    no_shortcut_route_delay = no_shortcut_queue_delay + fixed_delay

    shortcut_flow = arrival_rate
    shortcut_queue_delay = mm1_mean_time_in_system(
        arrival_rate=shortcut_flow,
        service_rate=congested_service_rate,
    )
    shortcut_route_delay = 2 * shortcut_queue_delay + shortcut_delay
    deviation_delay = shortcut_queue_delay + fixed_delay

    comparison = pd.DataFrame(
        [
            {
                "network": "without shortcut",
                "active_route": "top and bottom split equally",
                "flow_on_congested_queue_1": no_shortcut_flow,
                "flow_on_congested_queue_2": no_shortcut_flow,
                "mean_time_per_customer": no_shortcut_route_delay,
            },
            {
                "network": "with shortcut",
                "active_route": "all customers use shortcut route",
                "flow_on_congested_queue_1": shortcut_flow,
                "flow_on_congested_queue_2": shortcut_flow,
                "mean_time_per_customer": shortcut_route_delay,
            },
        ]
    )

    route_details = pd.DataFrame(
        [
            {
                "network": "without shortcut",
                "route": "Entry -> Q1 -> F1 -> Exit",
                "status": "used",
                "route_flow": no_shortcut_flow,
                "route_proportion": no_shortcut_flow / arrival_rate,
                "mean_time": no_shortcut_route_delay,
            },
            {
                "network": "without shortcut",
                "route": "Entry -> F2 -> Q2 -> Exit",
                "status": "used",
                "route_flow": no_shortcut_flow,
                "route_proportion": no_shortcut_flow / arrival_rate,
                "mean_time": no_shortcut_route_delay,
            },
            {
                "network": "with shortcut",
                "route": "Entry -> Q1 -> Shortcut -> Q2 -> Exit",
                "status": "used",
                "route_flow": shortcut_flow,
                "route_proportion": shortcut_flow / arrival_rate,
                "mean_time": shortcut_route_delay,
            },
            {
                "network": "with shortcut",
                "route": "Entry -> Q1 -> F1 -> Exit",
                "status": "unused deviation check",
                "route_flow": 0.0,
                "route_proportion": 0.0,
                "mean_time": deviation_delay,
            },
            {
                "network": "with shortcut",
                "route": "Entry -> F2 -> Q2 -> Exit",
                "status": "unused deviation check",
                "route_flow": 0.0,
                "route_proportion": 0.0,
                "mean_time": deviation_delay,
            },
        ]
    )

    route_proportions = route_details[
        [
            "network",
            "route",
            "status",
            "route_flow",
            "route_proportion",
        ]
    ].copy()

    parameters = pd.DataFrame(
        [
            {"parameter": "external arrival rate", "value": arrival_rate},
            {"parameter": "Q1 service rate", "value": congested_service_rate},
            {"parameter": "Q2 service rate", "value": congested_service_rate},
            {"parameter": "fixed station delay F1", "value": fixed_delay},
            {"parameter": "fixed station delay F2", "value": fixed_delay},
            {"parameter": "shortcut delay", "value": shortcut_delay},
        ]
    )

    return {
        "parameters": parameters,
        "comparison": comparison,
        "route_details": route_details,
        "route_proportions": route_proportions,
        "no_shortcut_mean_time": no_shortcut_route_delay,
        "with_shortcut_mean_time": shortcut_route_delay,
        "deviation_delay_with_shortcut": deviation_delay,
    }


def plot_braess_queue_comparison(results):
    """Plot mean customer time with and without the shortcut."""
    comparison = results["comparison"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["tab:blue", "tab:red"]
    ax.bar(
        comparison["network"],
        comparison["mean_time_per_customer"],
        color=colors,
        alpha=0.8,
    )
    ax.set_title("Mean time per customer")
    ax.set_ylabel("mean time in the network")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def draw_braess_queue_network(include_shortcut=False):
    """Draw the queueing network used in the Braess example."""
    positions = {
        "Entry": (0.0, 0.0),
        "Q1": (1.4, 1.0),
        "F1": (3.0, 1.0),
        "F2": (1.4, -1.0),
        "Q2": (3.0, -1.0),
        "Exit": (4.4, 0.0),
    }
    edges = [
        ("Entry", "Q1"),
        ("Q1", "F1"),
        ("F1", "Exit"),
        ("Entry", "F2"),
        ("F2", "Q2"),
        ("Q2", "Exit"),
    ]
    if include_shortcut:
        edges.append(("Q1", "Q2"))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    ax.set_xlim(-0.8, 5.0)
    ax.set_ylim(-1.8, 1.8)

    for source, target in edges:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        is_shortcut = source == "Q1" and target == "Q2"
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color="crimson" if is_shortcut else "black",
            linestyle="--" if is_shortcut else "-",
            shrinkA=22,
            shrinkB=22,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    for node, (x, y) in positions.items():
        if node == "Entry":
            label = "Entry\nlambda=1"
            facecolor = "#eef7ee"
        elif node == "Exit":
            label = "Exit"
            facecolor = "#f7f7f7"
        elif node in {"Q1", "Q2"}:
            label = f"{node}\nM/M/1\nmu=1.4"
            facecolor = "#f7fbff"
        else:
            label = f"{node}\nfixed\n2.6"
            facecolor = "#fff8e6"

        circle = Circle((x, y), radius=0.42, facecolor=facecolor, edgecolor="#1f4e79", linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold")

    if include_shortcut:
        ax.text(
            3.35,
            0.0,
            "shortcut\n0.05",
            ha="left",
            va="center",
            fontsize=10,
            color="crimson",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="crimson"),
        )

    title = "Queueing network with shortcut" if include_shortcut else "Queueing network without shortcut"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def _lognormal_realizations(rng, mean, cv, size):
    """Generate lognormal samples from a target mean and coefficient of variation."""
    sigma = np.sqrt(np.log(1 + cv ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return rng.lognormal(mean=mu, sigma=sigma, size=size)


def draw_queue_optimization_network(example):
    """Draw one of the queueing-network optimization examples."""
    if example == "split":
        positions = {
            "Entry": (0.0, 0.0),
            "S1": (1.8, 0.8),
            "S2": (1.8, -0.8),
            "Exit": (3.6, 0.0),
        }
        edges = [("Entry", "S1"), ("Entry", "S2"), ("S1", "Exit"), ("S2", "Exit")]
        title = "Flow split optimization"
    elif example == "server_mix":
        positions = {
            "Entry": (0.0, 0.0),
            "Queue": (1.4, 0.0),
            "Slow": (3.0, 0.8),
            "Fast": (3.0, -0.8),
            "Exit": (4.6, 0.0),
        }
        edges = [("Entry", "Queue"), ("Queue", "Slow"), ("Queue", "Fast"), ("Slow", "Exit"), ("Fast", "Exit")]
        title = "Server mix optimization"
    elif example == "throughput":
        positions = {
            "Raw material": (0.0, 0.0),
            "S1": (1.4, 0.0),
            "S2": (2.8, 0.8),
            "S3": (2.8, -0.8),
            "S4": (4.2, 0.0),
            "Exit": (5.5, 0.0),
        }
        edges = [
            ("Raw material", "S1"),
            ("S1", "S2"),
            ("S1", "S3"),
            ("S2", "S4"),
            ("S3", "S4"),
            ("S4", "Exit"),
        ]
        title = "Saturated production throughput maximization"
    else:
        raise ValueError("example must be one of: split, server_mix, throughput")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)

    for source, target in edges:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color="black",
            shrinkA=24,
            shrinkB=24,
        )
        ax.add_patch(arrow)

    for node, (x, y) in positions.items():
        if node in {"Entry", "Raw material", "Exit"}:
            facecolor = "#f7f7f7"
        elif node in {"Fast", "Slow"}:
            facecolor = "#fff7e6"
        else:
            facecolor = "#f7fbff"
        circle = Circle((x, y), radius=0.32, facecolor=facecolor, edgecolor="#1f4e79", linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, node, ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def simulate_parallel_split_from_inputs(
    inter_arrival_times,
    service_times_s1,
    service_times_s2,
    split_probability_to_s1,
    random_numbers,
):
    """Simulate Entry -> S1/S2 -> Exit with pre-generated non-Markovian inputs."""
    env = simpy.Environment()
    resources = {
        "S1": simpy.Resource(env, capacity=1),
        "S2": simpy.Resource(env, capacity=1),
    }
    service_times = {
        "S1": np.asarray(service_times_s1, dtype=float),
        "S2": np.asarray(service_times_s2, dtype=float),
    }
    service_index = {"S1": 0, "S2": 0}
    records = []

    def get_service_time(station):
        index = service_index[station]
        if index >= len(service_times[station]):
            raise RuntimeError(f"Not enough service-time realizations for {station}")
        service_index[station] += 1
        return float(service_times[station][index])

    def customer(customer_id, station):
        arrival_time = env.now
        with resources[station].request() as request:
            yield request
            service_start = env.now
            service_time = get_service_time(station)
            yield env.timeout(service_time)
        records.append(
            {
                "customer_id": customer_id,
                "station": station,
                "arrival_time": arrival_time,
                "waiting_time": service_start - arrival_time,
                "service_time": service_time,
                "departure_time": env.now,
                "system_time": env.now - arrival_time,
            }
        )

    def arrivals():
        for customer_id, inter_arrival_time in enumerate(inter_arrival_times):
            yield env.timeout(float(inter_arrival_time))
            station = "S1" if random_numbers[customer_id] < split_probability_to_s1 else "S2"
            env.process(customer(customer_id, station))

    env.process(arrivals())
    env.run()
    return pd.DataFrame(records)


def evaluate_split_proportion_optimization(
    num_customers=2_500,
    seed=2026,
    split_grid=None,
):
    """Evaluate split proportions using simulation with the same generated inputs."""
    rng = np.random.default_rng(seed)
    split_grid = np.asarray(split_grid if split_grid is not None else np.linspace(0.05, 0.95, 19), dtype=float)

    inter_arrival_times = _lognormal_realizations(rng, mean=0.95, cv=1.1, size=num_customers)
    service_times_s1 = _lognormal_realizations(rng, mean=0.75, cv=0.9, size=num_customers)
    service_times_s2 = _lognormal_realizations(rng, mean=1.15, cv=0.55, size=num_customers)
    routing_random_numbers = rng.random(num_customers)

    rows = []
    for split_probability in split_grid:
        records = simulate_parallel_split_from_inputs(
            inter_arrival_times=inter_arrival_times,
            service_times_s1=service_times_s1,
            service_times_s2=service_times_s2,
            split_probability_to_s1=split_probability,
            random_numbers=routing_random_numbers,
        )
        rows.append(
            {
                "split_to_S1": split_probability,
                "split_to_S2": 1 - split_probability,
                "mean_waiting_time": records["waiting_time"].mean(),
                "p90_waiting_time": records["waiting_time"].quantile(0.90),
                "mean_system_time": records["system_time"].mean(),
                "visits_S1": int((records["station"] == "S1").sum()),
                "visits_S2": int((records["station"] == "S2").sum()),
            }
        )

    results = pd.DataFrame(rows)
    best_row = results.loc[results["mean_waiting_time"].idxmin()].copy()
    return {
        "results": results,
        "best": best_row,
        "input_summary": pd.DataFrame(
            [
                {"input": "inter-arrival", "distribution": "lognormal", "sample_mean": inter_arrival_times.mean()},
                {"input": "service S1", "distribution": "lognormal", "sample_mean": service_times_s1.mean()},
                {"input": "service S2", "distribution": "lognormal", "sample_mean": service_times_s2.mean()},
            ]
        ),
    }


def plot_split_proportion_results(result):
    """Plot waiting-time metrics by split proportion."""
    results = result["results"]
    best = result["best"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(results["split_to_S1"], results["mean_waiting_time"], marker="o", label="mean waiting time")
    ax.plot(results["split_to_S1"], results["p90_waiting_time"], marker="s", label="90th percentile waiting")
    ax.axvline(best["split_to_S1"], color="tab:red", linestyle="--", label="best simulated split")
    ax.set_xlabel("proportion routed to S1")
    ax.set_ylabel("waiting time")
    ax.set_title("Split proportion optimization by simulation")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig('cost_as_funciton_p.png')
    plt.show()


def simulate_heterogeneous_server_pool_from_inputs(
    inter_arrival_times,
    slow_service_times,
    fast_service_times,
    num_slow_servers,
    num_fast_servers,
):
    """Simulate one FIFO queue served by slow and fast non-identical servers."""
    if num_slow_servers + num_fast_servers <= 0:
        raise ValueError("At least one server is required")

    env = simpy.Environment()
    queue = simpy.Store(env)
    records = []
    service_pools = {
        "slow": np.asarray(slow_service_times, dtype=float),
        "fast": np.asarray(fast_service_times, dtype=float),
    }
    service_index = {"slow": 0, "fast": 0}

    def get_service_time(server_type):
        index = service_index[server_type]
        if index >= len(service_pools[server_type]):
            raise RuntimeError(f"Not enough service times for {server_type} servers")
        service_index[server_type] += 1
        return float(service_pools[server_type][index])

    def server(server_id, server_type):
        while True:
            customer = yield queue.get()
            service_start = env.now
            service_time = get_service_time(server_type)
            yield env.timeout(service_time)
            records.append(
                {
                    "customer_id": customer["customer_id"],
                    "arrival_time": customer["arrival_time"],
                    "server_id": server_id,
                    "server_type": server_type,
                    "waiting_time": service_start - customer["arrival_time"],
                    "service_time": service_time,
                    "departure_time": env.now,
                    "system_time": env.now - customer["arrival_time"],
                }
            )

    def arrivals():
        for customer_id, inter_arrival_time in enumerate(inter_arrival_times):
            yield env.timeout(float(inter_arrival_time))
            yield queue.put({"customer_id": customer_id, "arrival_time": env.now})

    for index in range(num_slow_servers):
        env.process(server(f"slow_{index + 1}", "slow"))
    for index in range(num_fast_servers):
        env.process(server(f"fast_{index + 1}", "fast"))
    env.process(arrivals())
    env.run()
    return pd.DataFrame(records)


def evaluate_server_mix_optimization(
    num_customers=2_500,
    seed=3030,
    budget=12,
    slow_server_cost=2,
    fast_server_cost=6,
    waiting_cost=3.0,
):
    """Evaluate slow/fast server combinations by simulation."""
    rng = np.random.default_rng(seed)
    inter_arrival_times = _lognormal_realizations(rng, mean=0.34, cv=1.0, size=num_customers)
    slow_service_times = _lognormal_realizations(rng, mean=1.45, cv=0.85, size=num_customers)
    fast_service_times = _lognormal_realizations(rng, mean=0.55, cv=0.65, size=num_customers)

    rows = []
    for num_slow in range(0, 6):
        for num_fast in range(0, 5):
            if num_slow + num_fast == 0:
                continue
            staffing_cost = slow_server_cost * num_slow + fast_server_cost * num_fast
            if staffing_cost > budget:
                continue
            records = simulate_heterogeneous_server_pool_from_inputs(
                inter_arrival_times=inter_arrival_times,
                slow_service_times=slow_service_times,
                fast_service_times=fast_service_times,
                num_slow_servers=num_slow,
                num_fast_servers=num_fast,
            )
            mean_wait = records["waiting_time"].mean()
            rows.append(
                {
                    "slow_servers": num_slow,
                    "fast_servers": num_fast,
                    "staffing_cost": staffing_cost,
                    "mean_waiting_time": mean_wait,
                    "p90_waiting_time": records["waiting_time"].quantile(0.90),
                    "objective": staffing_cost + waiting_cost * mean_wait,
                }
            )

    results = pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)
    return {
        "results": results,
        "best": results.iloc[0].copy(),
        "input_summary": pd.DataFrame(
            [
                {"input": "inter-arrival", "distribution": "lognormal", "sample_mean": inter_arrival_times.mean()},
                {"input": "slow service", "distribution": "lognormal", "sample_mean": slow_service_times.mean()},
                {"input": "fast service", "distribution": "lognormal", "sample_mean": fast_service_times.mean()},
            ]
        ),
        "budget": budget,
    }


def plot_server_mix_results(result):
    """Plot server-mix objective values."""
    results = result["results"].copy()
    results["design"] = results["slow_servers"].astype(str) + " slow, " + results["fast_servers"].astype(str) + " fast"
    top = results.sort_values("objective").head(10)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["design"], top["objective"], color="tab:green", alpha=0.75)
    ax.invert_yaxis()
    ax.set_xlabel("simulated objective")
    ax.set_title("Best feasible server mixes")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.show()


def simulate_feedforward_throughput_from_inputs(
    service_times_by_station,
    capacities,
    split_probability_to_s2=0.55,
    random_numbers=None,
    production_horizon=1_000,
):
    """Simulate a saturated S1 -> S2/S3 -> S4 production line."""
    station_names = ["S1", "S2", "S3", "S4"]
    if random_numbers is None:
        random_numbers = np.zeros(10_000)
    random_numbers = np.asarray(random_numbers, dtype=float)
    env = simpy.Environment()

    service_times = {
        station: np.asarray(service_times_by_station[station], dtype=float)
        for station in station_names
    }
    service_index = {station: 0 for station in station_names}
    buffers = {station: simpy.Store(env) for station in ["S2", "S3", "S4"]}
    records = []
    completion_times = []
    created_jobs = 0

    def get_service_time(station):
        index = service_index[station]
        if index >= len(service_times[station]):
            raise RuntimeError(f"Not enough service times for {station}")
        service_index[station] += 1
        return float(service_times[station][index])

    def next_customer_id():
        nonlocal created_jobs
        customer_id = created_jobs
        created_jobs += 1
        if customer_id >= len(random_numbers):
            raise RuntimeError("Not enough routing random numbers")
        return customer_id

    def record_visit(customer_id, station, arrival_time, service_start, departure_time):
        records.append(
            {
                "customer_id": customer_id,
                "station": station,
                "arrival_time": arrival_time,
                "waiting_time": service_start - arrival_time,
                "departure_time": departure_time,
            }
        )

    def s1_worker():
        while True:
            customer_id = next_customer_id()
            arrival_time = env.now
            service_start = env.now
            service_time = get_service_time("S1")
            yield env.timeout(service_time)
            departure_time = env.now
            record_visit(customer_id, "S1", arrival_time, service_start, departure_time)
            next_station = "S2" if random_numbers[customer_id] < split_probability_to_s2 else "S3"
            yield buffers[next_station].put({"customer_id": customer_id, "arrival_time": departure_time})

    def intermediate_worker(station):
        while True:
            item = yield buffers[station].get()
            arrival_time = item["arrival_time"]
            service_start = env.now
            service_time = get_service_time(station)
            yield env.timeout(service_time)
            departure_time = env.now
            record_visit(item["customer_id"], station, arrival_time, service_start, departure_time)
            yield buffers["S4"].put({"customer_id": item["customer_id"], "arrival_time": departure_time})

    def s4_worker():
        while True:
            item = yield buffers["S4"].get()
            arrival_time = item["arrival_time"]
            service_start = env.now
            service_time = get_service_time("S4")
            yield env.timeout(service_time)
            departure_time = env.now
            record_visit(item["customer_id"], "S4", arrival_time, service_start, departure_time)
            completion_times.append(departure_time)

    for _ in range(int(capacities["S1"])):
        env.process(s1_worker())
    for station in ["S2", "S3"]:
        for _ in range(int(capacities[station])):
            env.process(intermediate_worker(station))
    for _ in range(int(capacities["S4"])):
        env.process(s4_worker())

    env.run(until=production_horizon)

    records_df = pd.DataFrame(records)
    completed_by_horizon = int(np.sum(np.asarray(completion_times) <= production_horizon))
    return {
        "records": records_df,
        "jobs_started": int(created_jobs),
        "completed_by_horizon": completed_by_horizon,
        "throughput": completed_by_horizon / production_horizon,
        "work_in_process_at_horizon": int(created_jobs - completed_by_horizon),
        "production_horizon": production_horizon,
    }


def evaluate_throughput_capacity_optimization(
    seed=4040,
    production_horizon=1_000,
    total_servers=7,
    split_probability_to_s2=0.55,
):
    """Maximize simulated output rate for a saturated feed-forward production network."""
    rng = np.random.default_rng(seed)
    num_service_samples = int(max(20_000, production_horizon * total_servers * 3))
    random_numbers = rng.random(num_service_samples)
    service_times_by_station = {
        "S1": _lognormal_realizations(rng, mean=0.62, cv=0.75, size=num_service_samples),
        "S2": _lognormal_realizations(rng, mean=0.95, cv=0.95, size=num_service_samples),
        "S3": _lognormal_realizations(rng, mean=0.85, cv=0.85, size=num_service_samples),
        "S4": _lognormal_realizations(rng, mean=0.90, cv=0.80, size=num_service_samples),
    }

    rows = []
    for c1 in range(1, total_servers - 2):
        for c2 in range(1, total_servers - c1 - 1):
            for c3 in range(1, total_servers - c1 - c2):
                c4 = total_servers - c1 - c2 - c3
                if c4 < 1:
                    continue
                capacities = {"S1": c1, "S2": c2, "S3": c3, "S4": c4}
                sim = simulate_feedforward_throughput_from_inputs(
                    service_times_by_station=service_times_by_station,
                    capacities=capacities,
                    split_probability_to_s2=split_probability_to_s2,
                    random_numbers=random_numbers,
                    production_horizon=production_horizon,
                )
                mean_wait = sim["records"]["waiting_time"].mean() if not sim["records"].empty else np.nan
                rows.append(
                    {
                        "S1_servers": c1,
                        "S2_servers": c2,
                        "S3_servers": c3,
                        "S4_servers": c4,
                        "jobs_started": sim["jobs_started"],
                        "completed_by_horizon": sim["completed_by_horizon"],
                        "throughput": sim["throughput"],
                        "work_in_process_at_horizon": sim["work_in_process_at_horizon"],
                        "mean_waiting_time": mean_wait,
                    }
                )

    results = pd.DataFrame(rows).sort_values(["throughput", "mean_waiting_time"], ascending=[False, True]).reset_index(drop=True)
    return {
        "results": results,
        "best": results.iloc[0].copy(),
        "total_servers": total_servers,
        "production_horizon": production_horizon,
        "input_summary": pd.DataFrame(
            [
                {"input": "raw material before S1", "distribution": "infinite inventory", "sample_mean": np.nan},
            ]
            + [
                {"input": f"service {station}", "distribution": "lognormal", "sample_mean": values.mean()}
                for station, values in service_times_by_station.items()
            ]
        ),
    }


def plot_throughput_optimization_results(result):
    """Plot top capacity allocations by simulated throughput."""
    results = result["results"].copy()
    results["allocation"] = (
        "("
        + results["S1_servers"].astype(str)
        + ","
        + results["S2_servers"].astype(str)
        + ","
        + results["S3_servers"].astype(str)
        + ","
        + results["S4_servers"].astype(str)
        + ")"
    )
    top = results.head(12)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(top["allocation"], top["throughput"], color="tab:purple", alpha=0.75)
    ax.set_xlabel("capacity allocation (S1,S2,S3,S4)")
    ax.set_ylabel("completed customers per time unit")
    ax.set_title("Throughput by simulated capacity allocation")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

