#!/usr/bin/env python3
import argparse
import os
import csv
import time

from config import SIM_SLOTS, RSAMPLE_K, ARRIVAL_RATES, RESULTS_DIR
from network import build_conflict_graph, build_flows
from independent_sets import exact_enum_independent_sets, random_sample_independent_sets
from scheduler import simulate_maxweight

def main():
    parser = argparse.ArgumentParser(description="Run a single Max-Weight experiment")
    parser.add_argument('--links',      type=int,   required=True,
                        help="Number of links in the conflict graph")
    parser.add_argument('--flows',      type=int,   required=True,
                        help="Number of flows")
    parser.add_argument('--rate_pattern', choices=ARRIVAL_RATES.keys(), required=True,
                        help="Arrival rate pattern: low, medium, random, etc.")
    parser.add_argument('--method',     choices=['ExactEnum','GreedyApprox','RandomSample'], required=True,
                        help="Independent-set generation method")
    args = parser.parse_args()

    # 1) 构造到达率列表、冲突图、流路径
    # build arrival list, conflict graph and flows
    arrival_rates = ARRIVAL_RATES[args.rate_pattern](args.flows)
    G = build_conflict_graph(args.links)
    flows = build_flows(G, args.flows)

    # 2) 对 ExactEnum 和 RandomSample 提前生成 Isets；GreedyApprox 留空
    # use pre-generated Isets for ExactEnum and RandomSample. GreedyApprax leave blank
    if args.method == 'ExactEnum':
        Isets = exact_enum_independent_sets(G)
    elif args.method == 'RandomSample':
        Isets = random_sample_independent_sets(G, RSAMPLE_K)
    else:  # GreedyApprox
        Isets = []

    # 3) 运行仿真并计时
    # run simulation and time
    start = time.time()
    throughputs, avg_queue, avg_decision_time = simulate_maxweight(
        G=G,
        flows=flows,
        arrival_rates=arrival_rates,
        Isets=Isets,
        sim_slots=SIM_SLOTS,
        method=args.method,
        rsample_k=RSAMPLE_K
    )
    runtime = time.time() - start

    # 4) 保存结果到 CSV（附加 total_load 列）
    # save to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = f"L{args.links}_F{args.flows}_{args.rate_pattern}_{args.method}.csv"
    path = os.path.join(RESULTS_DIR, filename)

    total_load = sum(arrival_rates)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'throughputs',
            'avg_queue',
            'avg_decision_time',
            'runtime',
            'total_load'
        ])
        writer.writerow([
            throughputs,
            avg_queue,
            avg_decision_time,
            runtime,
            total_load
        ])

    print(f"Saved results to {path}")

if __name__ == '__main__':
    main()
