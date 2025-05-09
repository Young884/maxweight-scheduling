#!/usr/bin/env python3
import time
import random
from typing import List, Tuple, Dict

import networkx as nx
from independent_sets import (
    exact_enum_independent_sets,
    random_sample_independent_sets,
    greedy_approx_independent_sets,
)

def simulate_maxweight(
    G: nx.Graph,
    flows: List[List[int]],
    arrival_rates: List[float],
    Isets: List[List[int]],
    sim_slots: int,
    method: str = "ExactEnum",
    rsample_k: int = 50,
) -> Tuple[List[float], float, float]:
    """
    Run Max-Weight scheduling simulation.

    Args:
        G: conflict graph (nodes are link IDs, edges represent conflicts)
        flows: list of flows, each a list of link IDs in path order
        arrival_rates: list of per-flow Bernoulli arrival probabilities
        Isets: precomputed independent sets (used if method != 'GreedyApprox')
        sim_slots: number of time slots to simulate
        method: 'ExactEnum' | 'RandomSample' | 'GreedyApprox'
        rsample_k: sample size for RandomSample

    Returns:
        throughputs: delivered packets per slot for each flow
        avg_queue_len: time-average total queue length
        avg_decision_time: average decision computation time per slot (s)
    """
    # Initialize per-(flow,hop) queues
    queues: Dict[Tuple[int,int], int] = {
        (i, h): 0
        for i, path in enumerate(flows)
        for h in range(len(path))
    }
    delivered = [0 for _ in flows]

    total_queue = 0.0
    total_decision_time = 0.0

    for t in range(sim_slots):
        # 1) arrivals
        for i, lam in enumerate(arrival_rates):
            if random.random() < lam:
                queues[(i, 0)] += 1

        # 2) select candidate independent sets
        if method == "ExactEnum":
            candidate_sets = Isets
        elif method == "RandomSample":
            candidate_sets = Isets
        else:  # GreedyApprox
            # compute current queue lengths per link
            queue_lengths: Dict[int, int] = {node: 0 for node in G.nodes()}
            for (i, path) in enumerate(flows):
                for h, link in enumerate(path):
                    queue_lengths[link] += queues[(i, h)]
            candidate_sets = [greedy_approx_independent_sets(G, queue_lengths)]

        # 3) compute weights and pick best
        start = time.perf_counter()
        best_set: List[int] = []
        best_weight = -1
        for S in candidate_sets:
            weight = 0
            for link in S:
                # sum all queued packets on this link across flows
                for (i, path) in enumerate(flows):
                    for h, l in enumerate(path):
                        if l == link:
                            weight += queues[(i, h)]
            if weight > best_weight:
                best_weight = weight
                best_set = S
        total_decision_time += time.perf_counter() - start

        # 4) serve one packet per activated link
        for link in best_set:
            for (i, path) in enumerate(flows):
                for h, l in enumerate(path):
                    if l == link and queues[(i, h)] > 0:
                        queues[(i, h)] -= 1
                        if h + 1 < len(path):
                            queues[(i, h + 1)] += 1
                        else:
                            delivered[i] += 1
                        # only one packet per link per slot
                        break

        total_queue += sum(queues.values())

    throughputs = [d / sim_slots for d in delivered]
    avg_queue_len = total_queue / sim_slots
    avg_decision_time = total_decision_time / sim_slots

    return throughputs, avg_queue_len, avg_decision_time


if __name__ == "__main__":
    # quick sanity check on a simple chain 3-node graph
    G = nx.path_graph([0, 1, 2])
    flows = [[0, 1], [1, 2]]
    arrival = [0.3, 0.2]
    # precompute for ExactEnum
    Isets = exact_enum_independent_sets(G)
    thr, avg_q, avg_dt = simulate_maxweight(
        G, flows, arrival, Isets, sim_slots=10000, method="GreedyApprox"
    )
    print("Throughputs:", thr)
    print("Avg queue len:", avg_q)
    print("Avg decision time:", avg_dt)
