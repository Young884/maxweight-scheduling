#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Topology Tests for MaxWeight Scheduling Algorithm
This script contains different network topology tests and a command line interface
to select which topology to run.
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import networkx as nx
import random


# Import your existing classes from the main file
# Assuming your main file is named maxweight_scheduling.py
from max1 import (Node, Link, Flow, Network, MaxWeightScheduler,
                                  Simulation, CapacityRegionExplorer, create_line_network)


# Define test topologies
def setup_line_test():
    """
    Creates a simple 4-node line network with two flows.
    Flow 1: Node 0 -> Node 3 (end-to-end)
    Flow 2: Node 0 -> Node 2 (shorter flow)
    """
    # Create a 4-node line network
    network = create_line_network(4)

    # Add two flows
    flow1 = network.add_flow(0, network.nodes[0], network.nodes[3], 0.3)  # End-to-end flow
    flow2 = network.add_flow(1, network.nodes[0], network.nodes[2], 0.3)  # Shorter flow

    return network


def setup_ring_test():
    """
    Creates a 6-node ring network with two flows that can take multiple paths.
    Flow 1: Node 0 -> Node 3
    Flow 2: Node 2 -> Node 5
    """
    network = Network()

    # Add nodes
    nodes = []
    for i in range(6):
        nodes.append(network.add_node(Node(i)))

    # Create ring
    for i in range(5):
        network.add_link(nodes[i], nodes[i + 1], 1)
    network.add_link(nodes[5], nodes[0], 1)  # Close the ring

    # Add flows
    flow1 = network.add_flow(0, nodes[0], nodes[3], 0.3)  # Can take two paths
    flow2 = network.add_flow(1, nodes[2], nodes[5], 0.3)  # Can take two paths

    return network


def setup_butterfly_test():
    """
    Creates a butterfly network with two flows sharing bottleneck links.
    Flow 1: Node 4 -> Node 5
    Flow 2: Node 1 -> Node 5
    This is a classic topology to test network coding and shared bottlenecks.
    """
    network = Network()

    # Add nodes
    nodes = []
    for i in range(6):
        nodes.append(network.add_node(Node(i)))

    # Add links
    network.add_link(nodes[0], nodes[2], 1)
    network.add_link(nodes[1], nodes[2], 1)
    network.add_link(nodes[0], nodes[3], 1)
    network.add_link(nodes[1], nodes[3], 1)
    network.add_link(nodes[4], nodes[2], 0.8)  # Bottleneck link
    network.add_link(nodes[3], nodes[5], 0.8)  # Bottleneck link
    network.add_link(nodes[2], nodes[3], 0.8)  # bottleneck link

    # Add flows
    flow1 = network.add_flow(0, nodes[4], nodes[5], 0.3)
    flow2 = network.add_flow(1, nodes[1], nodes[5], 0.3)

    return network


def setup_dense_5node_test(p_conflict=0.3):
    """
    创建一个只有5个节点但链路较多的网络
    参数:
    p_conflict -- 冲突概率 (0到1之间)
    """
    # 创建网络对象
    network = Network()

    # 明确创建5个节点
    nodes = []
    for i in range(5):
        nodes.append(network.add_node(Node(i)))

    # 创建多条链路（不仅仅是线性连接）
    links = []

    # 基础线性链路
    for i in range(4):
        capacity = 0.2 + random.random() * 0.2
        link = network.add_link(nodes[i], nodes[i + 1], capacity)
        links.append(link)

    # 添加额外链路
    link1 = network.add_link(nodes[0], nodes[2], 0.2)  # 0到2的链路
    link2 = network.add_link(nodes[0], nodes[3], 0.2)  # 0到3的链路
    link3 = network.add_link(nodes[1], nodes[3], 0.3)  # 1到3的链路
    link4 = network.add_link(nodes[1], nodes[4], 0.2)  # 1到4的链路

    links.append(link1)
    links.append(link2)
    links.append(link3)
    links.append(link4)

    # 创建冲突图
    G = nx.Graph()
    G.add_nodes_from(range(len(links)))

    # 添加随机冲突
    for i in range(len(links)):
        for j in range(i + 1, len(links)):
            if random.random() < p_conflict:
                G.add_edge(i, j)

    # 打印冲突信息
    print(f"Created dense 5-node network with {len(links)} links and {G.number_of_edges()} conflicts")

    # 创建流
    # 一个从节点0到节点4
    flow1 = network.add_flow(0, nodes[0], nodes[4], 0.5)
    # 一个从节点2到节点4
    flow2 = network.add_flow(1, nodes[2], nodes[4], 0.4)
    # 一个从节点4到节点2
    flow3 = network.add_flow(2, nodes[1], nodes[3], 0.4)  # 1到3的流

    # 更新网络类以考虑冲突图
    network.conflict_graph = G

    # 重写is_conflicting方法以使用冲突图
    original_is_conflicting = network.is_conflicting

    def new_is_conflicting(self, link1, link2):
        # 首先检查原始的半双工冲突
        if original_is_conflicting(link1, link2):
            return True

        # 然后检查冲突图中的冲突
        try:
            link1_idx = links.index(link1)
            link2_idx = links.index(link2)
            return G.has_edge(link1_idx, link2_idx)
        except ValueError:
            # 如果链路不在links列表中，使用原始方法
            return False

    # 替换方法
    network.is_conflicting = new_is_conflicting.__get__(network, Network)

    return network

def setup_random_conflict_test(num_links=10, num_flows=2, p_conflict=0.2):
    """
    Generate network topology and flows with random conflict graph

    Parameters:
    num_links -- Number of links in the conflict graph
    num_flows -- Number of flows to generate
    p_conflict -- Conflict probability (between 0 and 1)
    """
    # Create network object
    network = Network()

    # Create nodes
    nodes = []
    for i in range(num_links + 1):  # 需要比链路数多1个节点
        nodes.append(network.add_node(Node(i)))

    # Create links
    links = []
    for i in range(num_links):
        # 每个链路连接相邻节点，容量随机在0.3到1.0之间
        capacity = 0.3 + random.random() * 0.7
        link = network.add_link(nodes[i], nodes[i + 1], capacity)
        links.append(link)

    # Create conflict graph
    G = nx.Graph()
    G.add_nodes_from(range(num_links))

    # Add random conflicts
    for i in range(num_links):
        for j in range(i + 1, num_links):
            if random.random() < p_conflict:
                G.add_edge(i, j)

    # Print conflict information
    print(f"Created random conflict graph with {num_links} links and {G.number_of_edges()} conflicts")

    # Create flows
    # For simplicity, we choose random source and destination nodes
    for i in range(num_flows):
        src_idx = random.randint(0, len(nodes) // 2)  # 从前半部分选源节点
        dst_idx = random.randint(len(nodes) // 2, len(nodes) - 1)  # 从后半部分选目标节点

        # Add flow, initial rate 0.3
        flow = network.add_flow(i, nodes[src_idx], nodes[dst_idx], 0.3)

    # Update network class to consider conflict graph
    network.conflict_graph = G

    # Rewrite is_conflicting method to use conflict graph
    original_is_conflicting = network.is_conflicting

    def new_is_conflicting(self, link1, link2):
        # First check original half-duplex conflicts
        if original_is_conflicting(link1, link2):
            return True

        # Then check conflicts in the conflict graph
        try:
            link1_idx = links.index(link1)
            link2_idx = links.index(link2)
            return G.has_edge(link1_idx, link2_idx)
        except ValueError:
            # If the link is not in the links list, use the original method
            return False

    # Replace method
    network.is_conflicting = new_is_conflicting.__get__(network, Network)

    return network

def save_network_topology(network, output_dir, test_name):
    """Save network topology information to a file"""
    filename = os.path.join(output_dir, f"{test_name}_network_topology.txt")
    with open(filename, 'w') as f:
        f.write("Network Topology Structure\n")
        f.write("========================\n\n")

        f.write("Nodes List:\n")
        for node in network.nodes:
            f.write(f"  Node {node.id}\n")

        f.write("\nLinks List:\n")
        for link in network.links:
            f.write(f"  Link {link.source.id} -> {link.destination.id} (Capacity: {link.capacity})\n")


def save_independent_sets(network, output_dir, test_name):
    """Save independent sets information to a file"""
    filename = os.path.join(output_dir, f"{test_name}_independent_sets.txt")
    with open(filename, 'w') as f:
        f.write("Independent Sets List\n")
        f.write("====================\n\n")

        for i, ind_set in enumerate(network.independent_sets):
            f.write(f"Independent Set {i + 1}:\n")
            for link in ind_set:
                f.write(f"  Link {link.source.id} -> {link.destination.id}\n")
            f.write("\n")


def save_flow_paths(network, output_dir, test_name):
    """Save flow paths information to a file"""
    filename = os.path.join(output_dir, f"{test_name}_flow_paths.txt")
    with open(filename, 'w') as f:
        f.write("Flow Configuration and Paths\n")
        f.write("==========================\n\n")

        for flow in network.flows:
            f.write(f"Flow {flow.id}:\n")
            f.write(f"  Source Node: {flow.source.id}\n")
            f.write(f"  Destination Node: {flow.destination.id}\n")
            f.write(f"  Rate: {flow.rate}\n")

            f.write("  Path: ")
            if flow.path:
                path_str = " -> ".join([str(node.id) for node in flow.path])
                f.write(path_str + "\n")
            else:
                f.write("Not set\n")
            f.write("\n")


def save_capacity_region_data(explorer, output_dir, test_name):
    """Save capacity region data to CSV file"""
    filename = os.path.join(output_dir, f"{test_name}_capacity_region.csv")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        header = ["Point Type"]
        for i in range(len(explorer.network.flows)):
            header.append(f"Flow {i} Rate")
        writer.writerow(header)

        # Write stable points
        for point in explorer.stable_points:
            row = ["Stable"] + list(point)
            writer.writerow(row)

        # Write unstable points
        for point in explorer.unstable_points:
            row = ["Unstable"] + list(point)
            writer.writerow(row)


def run_simulation_and_save_data(network, output_dir, test_name, simulation_time=2000):
    """Run a simulation and save queue evolution data"""
    # Create simulation
    simulation = Simulation(network, max_time=simulation_time)
    is_stable, _ = simulation.run()

    # Save queue evolution to CSV
    filename = os.path.join(output_dir, f"{test_name}_queue_evolution.csv")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Create header
        header = ["Time Step"]
        for node in network.nodes:
            for flow in network.flows:
                header.append(f"Node {node.id} Flow {flow.id}")
        writer.writerow(header)

        # Write each time step's queue lengths
        for t, state in enumerate(simulation.queue_history):
            row = [t]
            for node in network.nodes:
                for flow in network.flows:
                    row.append(state[node.id].get(flow.id, 0))
            writer.writerow(row)

    # Save scheduling decisions to CSV
    filename = os.path.join(output_dir, f"{test_name}_scheduling_decisions.csv")

    # Reset simulation to collect scheduling decisions
    simulation = Simulation(network, max_time=min(100, simulation_time))
    scheduler = simulation.scheduler

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Create header - all links
        header = ["Time Step"]
        link_labels = []
        for link in network.links:
            label = f"Link {link.source.id}->{link.destination.id}"
            link_labels.append(label)
            header.append(label)
        writer.writerow(header)

        # Run simulation and record scheduling decisions
        for t in range(simulation.max_time):
            # Generate packets
            simulation.generate_packets()

            # Get scheduling decision
            selected_set = scheduler.schedule()

            # Prepare row data
            row = [t]
            for link in network.links:
                if selected_set and link in selected_set:
                    row.append(1)  # Active
                else:
                    row.append(0)  # Inactive
            writer.writerow(row)

            # Transmit packets
            simulation.transmit_packets(selected_set)

    return is_stable


def generate_analysis_plots(output_dir, plots_dir, test_name):
    """Generate analysis plots from output files"""
    # Create plots directory if it doesn't exist
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Get file paths
    capacity_file = os.path.join(output_dir, f"{test_name}_capacity_region.csv")
    queue_file = os.path.join(output_dir, f"{test_name}_queue_evolution.csv")
    scheduling_file = os.path.join(output_dir, f"{test_name}_scheduling_decisions.csv")

    # Plot capacity region
    if os.path.exists(capacity_file):
        plot_capacity_region(capacity_file, plots_dir, test_name)

    # Plot queue evolution
    if os.path.exists(queue_file):
        plot_queue_evolution(queue_file, plots_dir, test_name)

    # Plot scheduling decisions
    if os.path.exists(scheduling_file):
        plot_scheduling_decisions(scheduling_file, plots_dir, test_name)
        plot_node_activity(scheduling_file, plots_dir, test_name)

    print(f"Analysis plots have been saved to: {plots_dir}")


def plot_capacity_region(capacity_file, plots_dir, test_name):
    """Plot capacity region from CSV file"""
    # Read data
    stable_points = []
    unstable_points = []

    with open(capacity_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            point_type = row[0]
            rates = [float(x) for x in row[1:]]

            if point_type == "Stable":
                stable_points.append(rates)
            else:
                unstable_points.append(rates)

    # Convert to numpy arrays
    stable_points = np.array(stable_points) if stable_points else np.array([])
    unstable_points = np.array(unstable_points) if unstable_points else np.array([])

    # Determine dimension
    if len(stable_points) > 0:
        dim = stable_points.shape[1]
    elif len(unstable_points) > 0:
        dim = unstable_points.shape[1]
    else:
        print("Warning: No data points found in capacity region file.")
        return

    # Create appropriate plot based on dimension
    if dim == 2:
        # 2D plot
        plt.figure(figsize=(10, 8))

        if len(stable_points) > 0:
            plt.scatter(stable_points[:, 0], stable_points[:, 1], c='green', marker='o', label='Stable')

        if len(unstable_points) > 0:
            plt.scatter(unstable_points[:, 0], unstable_points[:, 1], c='red', marker='x', label='Unstable')

        plt.xlabel('Flow 0 Rate')
        plt.ylabel('Flow 1 Rate')
        plt.title(f'Capacity Region - {test_name}')
        plt.grid(True)
        plt.legend()

        # Try to draw an approximate boundary if unstable points exist
        if len(unstable_points) > 0:
            try:
                # Simple approximation - more sophisticated methods could be used
                max_stable_x = max([p[0] for p in stable_points]) if len(stable_points) > 0 else 0
                max_stable_y = max([p[1] for p in stable_points]) if len(stable_points) > 0 else 0

                min_unstable_x = min([p[0] for p in unstable_points])
                min_unstable_y = min([p[1] for p in unstable_points])

                plt.plot([0, max_stable_x, min_unstable_x], [max_stable_y, max_stable_y, 0],
                         'k--', label='Estimated Boundary')
                plt.legend()
            except:
                pass

        plt.savefig(os.path.join(plots_dir, f"{test_name}_capacity_region_2d.png"))
        plt.close()

    elif dim == 3:
        # 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if len(stable_points) > 0:
            ax.scatter(stable_points[:, 0], stable_points[:, 1], stable_points[:, 2],
                       c='green', marker='o', label='Stable')

        if len(unstable_points) > 0:
            ax.scatter(unstable_points[:, 0], unstable_points[:, 1], unstable_points[:, 2],
                       c='red', marker='x', label='Unstable')

        ax.set_xlabel('Flow 0 Rate')
        ax.set_ylabel('Flow 1 Rate')
        ax.set_zlabel('Flow 2 Rate')
        ax.set_title(f'3D Capacity Region - {test_name}')
        ax.legend()

        plt.savefig(os.path.join(plots_dir, f"{test_name}_capacity_region_3d.png"))
        plt.close()


def plot_queue_evolution(queue_file, plots_dir, test_name):
    """Plot queue evolution for each node and flow"""
    # Read data
    with open(queue_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Get header

        # Extract node and flow info from header
        node_flow_pairs = header[1:]  # Skip time step column

        # Initialize data structure
        queue_data = {pair: [] for pair in node_flow_pairs}
        time_steps = []

        # Read data
        for row in reader:
            time_steps.append(int(row[0]))
            for i, pair in enumerate(node_flow_pairs):
                queue_data[pair].append(float(row[i + 1]))

    # Create plots
    # 1. All queues in one plot
    plt.figure(figsize=(12, 8))
    for pair, values in queue_data.items():
        plt.plot(time_steps, values, label=pair)

    plt.xlabel('Time Step')
    plt.ylabel('Queue Length')
    plt.title(f'Queue Evolution for All Nodes and Flows - {test_name}')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(plots_dir, f"{test_name}_all_queues.png"))
    plt.close()

    # 2. Separate plots for each node-flow pair
    for pair, values in queue_data.items():
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, values)
        plt.xlabel('Time Step')
        plt.ylabel('Queue Length')
        plt.title(f'Queue Evolution for {pair} - {test_name}')
        plt.grid(True)

        # Save with sanitized filename
        filename = pair.replace(" ", "_").replace(":", "_")
        plt.savefig(os.path.join(plots_dir, f"{test_name}_queue_{filename}.png"))
        plt.close()


def plot_scheduling_decisions(scheduling_file, plots_dir, test_name):
    """Plot scheduling decisions over time"""
    # Read data
    with open(scheduling_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Get header

        # Extract link labels
        link_labels = header[1:]  # Skip time step column

        # Initialize data
        time_steps = []
        decisions = {link: [] for link in link_labels}

        # Read data
        for row in reader:
            time_steps.append(int(row[0]))
            for i, link in enumerate(link_labels):
                decisions[link].append(int(row[i + 1]))

    # Plot scheduling decisions as a heatmap
    plt.figure(figsize=(12, 8))

    # Prepare data for heatmap
    decision_matrix = np.zeros((len(link_labels), len(time_steps)))
    for i, link in enumerate(link_labels):
        decision_matrix[i] = decisions[link]

    plt.imshow(decision_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Link Active')
    plt.yticks(range(len(link_labels)), link_labels)
    plt.xlabel('Time Step')
    plt.ylabel('Link')
    plt.title(f'Scheduling Decisions Over Time - {test_name}')

    plt.savefig(os.path.join(plots_dir, f"{test_name}_scheduling_decisions.png"))
    plt.close()

    # Plot scheduling frequency for each link
    activity_rates = {}
    for link in link_labels:
        activity_rates[link] = sum(decisions[link]) / len(time_steps)

    plt.figure(figsize=(12, 8))
    links = list(activity_rates.keys())
    rates = list(activity_rates.values())

    y_pos = np.arange(len(links))
    plt.barh(y_pos, rates)
    plt.yticks(y_pos, links)
    plt.xlabel('Activation Frequency')
    plt.title(f'Link Activation Frequency - {test_name}')

    plt.savefig(os.path.join(plots_dir, f"{test_name}_link_activation_frequency.png"))
    plt.close()


def plot_node_activity(scheduling_file, plots_dir, test_name):
    """Plot node activity based on scheduling decisions"""
    # Read data
    with open(scheduling_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Get header

        # Extract link labels
        link_labels = header[1:]  # Skip time step column

        # Extract node IDs from link labels
        node_ids = set()
        for link in link_labels:
            # Assume format is "Link X->Y"
            parts = link.split("->")
            if len(parts) == 2:
                source = parts[0].split()[-1]
                dest = parts[1]
                node_ids.add(source)
                node_ids.add(dest)

        node_ids = sorted(list(node_ids))

        # Initialize node activity data
        time_steps = []
        node_activity = {node: [] for node in node_ids}

        # Read scheduling decisions
        for row in reader:
            time_steps.append(int(row[0]))

            # Reset activity for this time step
            for node in node_ids:
                node_activity[node].append(0)

            # Mark nodes as active if any of their links are active
            for i, link in enumerate(link_labels):
                if int(row[i + 1]) == 1:  # If link is active
                    parts = link.split("->")
                    if len(parts) == 2:
                        source = parts[0].split()[-1]
                        dest = parts[1]
                        node_activity[source][-1] = 1
                        node_activity[dest][-1] = 1

    # Plot node activity over time
    plt.figure(figsize=(12, 8))

    # Prepare data for heatmap
    activity_matrix = np.zeros((len(node_ids), len(time_steps)))
    for i, node in enumerate(node_ids):
        activity_matrix[i] = node_activity[node]

    plt.imshow(activity_matrix, aspect='auto', cmap='plasma')
    plt.colorbar(label='Node Active')
    plt.yticks(range(len(node_ids)), [f"Node {node}" for node in node_ids])
    plt.xlabel('Time Step')
    plt.ylabel('Node')
    plt.title(f'Node Activity Over Time - {test_name}')

    plt.savefig(os.path.join(plots_dir, f"{test_name}_node_activity.png"))
    plt.close()

    # Plot node activation frequency
    node_frequencies = {}
    for node in node_ids:
        node_frequencies[node] = sum(node_activity[node]) / len(time_steps)

    plt.figure(figsize=(10, 6))
    nodes = [f"Node {node}" for node in node_frequencies.keys()]
    frequencies = list(node_frequencies.values())

    y_pos = np.arange(len(nodes))
    plt.barh(y_pos, frequencies)
    plt.yticks(y_pos, nodes)
    plt.xlabel('Activation Frequency')
    plt.title(f'Node Activation Frequency - {test_name}')

    plt.savefig(os.path.join(plots_dir, f"{test_name}_node_activation_frequency.png"))
    plt.close()


def plot_flow_rate_distribution(capacity_file, plots_dir, test_name):
    """Plot analysis of tested flow rates"""
    if not os.path.exists(capacity_file):
        print(f"Warning: {capacity_file} not found. Skipping flow rate analysis plots.")
        return

    # Read data
    stable_points = []
    unstable_points = []

    with open(capacity_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        # Determine number of flows
        num_flows = len(header) - 1  # Subtract point type column

        for row in reader:
            point_type = row[0]
            rates = [float(x) for x in row[1:]]

            if point_type == "Stable":
                stable_points.append(rates)
            else:
                unstable_points.append(rates)

    # Skip if no data
    if not stable_points and not unstable_points:
        print("Warning: No data points found for flow rate analysis.")
        return

    # Plot distribution of flow rates
    plt.figure(figsize=(10, 6))

    for i in range(num_flows):
        if stable_points:
            stable_rates = [point[i] for point in stable_points]
            plt.hist(stable_rates, alpha=0.5, label=f'Flow {i} (Stable)', bins=10)

        if unstable_points:
            unstable_rates = [point[i] for point in unstable_points]
            plt.hist(unstable_rates, alpha=0.5, label=f'Flow {i} (Unstable)', bins=10)

    plt.xlabel('Flow Rate')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Tested Flow Rates - {test_name}')
    plt.legend()

    plt.savefig(os.path.join(plots_dir, f"{test_name}_flow_rate_distribution.png"))
    plt.close()


def save_analysis_outputs(network, simulation, explorer, test_name, output_dir="output"):
    """Generate comprehensive analysis results, save to output directory"""
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save network topology info
    save_network_topology(network, output_dir, test_name)

    # Save independent sets info
    save_independent_sets(network, output_dir, test_name)

    # Save flow configuration
    save_flow_paths(network, output_dir, test_name)

    # Save capacity region data
    save_capacity_region_data(explorer, output_dir, test_name)

    print(f"All analysis results have been saved to directory: {output_dir}")


def visualize_network_topology(network, test_name, output_dir="output"):
    """创建并保存网络拓扑的可视化图"""
    import networkx as nx
    import matplotlib.pyplot as plt

    # 创建NetworkX图对象
    G = nx.DiGraph()

    # 添加节点
    for node in network.nodes:
        G.add_node(node.id)

    # 添加边（链路）
    for link in network.links:
        G.add_edge(link.source.id, link.destination.id, capacity=link.capacity)

    # 获取链路容量作为边的权重
    edge_capacities = [G[u][v]['capacity'] for u, v in G.edges()]

    # 创建布局
    if len(network.nodes) <= 20:  # 对于小型网络使用更好的布局
        pos = nx.spring_layout(G, seed=42)  # 使用弹簧布局，固定随机种子以获得一致的图像
    else:
        pos = nx.kamada_kawai_layout(G)  # 对于大型网络

    # 创建图形
    plt.figure(figsize=(12, 10))

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

    # 绘制边，颜色根据容量
    nx.draw_networkx_edges(G, pos, width=edge_capacities,
                           edge_color=edge_capacities, edge_cmap=plt.cm.Blues,
                           arrowsize=20, connectionstyle='arc3,rad=0.1')

    # 添加节点标签
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

    # 添加边标签（显示容量）
    edge_labels = {(u, v): f"{d['capacity']:.1f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # 添加流路径
    for flow in network.flows:
        if flow.path:
            flow_edges = [(flow.path[i].id, flow.path[i + 1].id) for i in range(len(flow.path) - 1)]
            # 绘制流路径
            nx.draw_networkx_edges(G, pos, edgelist=flow_edges, width=3,
                                   edge_color=f'C{flow.id}', style='dashed',
                                   alpha=0.7, arrowsize=25)

    # 添加图例说明
    plt.figtext(0.5, 0.02,
                f"Flow paths: {', '.join([f'Flow {flow.id}: {flow.source.id}->{flow.destination.id}' for flow in network.flows])}",
                ha='center', fontsize=12, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

    # 添加标题
    plt.title(f"Network Topology - {test_name}", fontsize=16)

    # 去除坐标轴
    plt.axis('off')

    # 保存图像
    output_file = os.path.join(output_dir, f"{test_name}_topology.png")
    plt.savefig(output_file)

    # 打印消息
    print(f"Network topology visualization saved to {output_file}")

    # 关闭图形
    plt.close()

    return output_file


def run_test(network, test_name, x_max=2.0, y_max=2.0, z_max=2.0, resolution=20, simulation_time=2000,
             stability_threshold=1000):
    """
    Runs a test on the given network topology.
    """
    print(f"Running test: {test_name}")
    print(f"Network has {len(network.nodes)} nodes and {len(network.links)} links")

    # Create output directories
    output_dir = f"output_{test_name}"
    plots_dir = f"plots_{test_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Generate network topology visualization
    visualize_network_topology(network, test_name, output_dir)

    # Setup flow paths
    network.setup_flows()
    print("Flow paths set up:")
    for flow in network.flows:
        path_str = " -> ".join([str(node.id) for node in flow.path])
        print(f"  Flow {flow.id}: {path_str}")

    # Update topology visualization with flow paths
    visualize_network_topology(network, f"{test_name}_with_flows", output_dir)

    # Generate independent sets
    network.generate_independent_sets()
    print(f"Found {len(network.independent_sets)} independent sets")

    # Run simulation to generate queue and scheduling data
    print("Running initial simulation to generate queue and scheduling data...")
    is_stable = run_simulation_and_save_data(network, output_dir, test_name, simulation_time)
    print(
        f"Initial simulation with rates {[flow.rate for flow in network.flows]}: {'Stable' if is_stable else 'Unstable'}")

    # Create capacity region explorer
    explorer = CapacityRegionExplorer(network, simulation_time=simulation_time)

    # Explore capacity region based on number of flows
    num_flows = len(network.flows)
    if num_flows == 2:
        print(f"Exploring 2D capacity region with resolution {resolution} up to rates ({x_max}, {y_max})...")
        explorer.explore_region_2d(x_max=x_max, y_max=y_max, resolution=resolution)
    elif num_flows == 3:
        print(f"Exploring 3D capacity region with resolution {resolution} up to rates ({x_max}, {y_max}, {z_max})...")
        explorer.explore_region_3d(x_max=x_max, y_max=y_max, z_max=z_max, resolution=resolution)
    else:
        print(f"Warning: Capacity region exploration supports 2 or 3 flows, but network has {num_flows} flows.")
        # Just explore first 2 or 3 flows depending on what's available
        if num_flows > 0:
            if num_flows >= 3:
                print(f"Exploring first 3 flows only...")
                explorer.explore_region_3d(x_max=x_max, y_max=y_max, z_max=z_max, resolution=resolution)
            else:
                print(f"Exploring first {num_flows} flows only...")
                explorer.explore_region_2d(x_max=x_max, y_max=y_max, resolution=resolution)

    # Count stable and unstable points
    stable_count = len(explorer.stable_points)
    unstable_count = len(explorer.unstable_points)
    total_points = stable_count + unstable_count

    print(f"Capacity region exploration complete.")
    print(f"Tested {total_points} points: {stable_count} stable, {unstable_count} unstable")

    # Save analysis data
    save_analysis_outputs(network, None, explorer, test_name, output_dir)

    # Generate analysis plots
    generate_analysis_plots(output_dir, plots_dir, test_name)

    # Plot flow rate distribution
    capacity_file = os.path.join(output_dir, f"{test_name}_capacity_region.csv")
    plot_flow_rate_distribution(capacity_file, plots_dir, test_name)

    print(f"Test complete. Results saved to {output_dir} and {plots_dir}")

    return explorer

def main():
    """
    Main function to parse command line arguments and run the selected test.
    """
    parser = argparse.ArgumentParser(description='Network Topology Tests for MaxWeight Scheduling')
    parser.add_argument('--topology', type=str,
                        choices=['line', 'ring', 'butterfly', 'random_conflict', 'dense_5node'],
                        default='line', help='Network topology to test')
    parser.add_argument('--xmax', type=float, default=2.0, help='Maximum rate for Flow 0')
    parser.add_argument('--ymax', type=float, default=2.0, help='Maximum rate for Flow 1')
    parser.add_argument('--zmax', type=float, default=2.0, help='Maximum rate for Flow 2 (if applicable)')
    parser.add_argument('--resolution', type=int, default=20, help='Number of points to test in each dimension')
    parser.add_argument('--time', type=int, default=2000, help='Simulation time for each test')
    parser.add_argument('--num_links', type=int, default=10, help='Number of links (for random topologies)')
    parser.add_argument('--num_flows', type=int, default=2, help='Number of flows (for random topologies)')
    parser.add_argument('--p_conflict', type=float, default=0.2, help='Conflict probability (for random topologies)')
    parser.add_argument('--threshold', type=int, default=1000, help='Stability threshold')

    args = parser.parse_args()

    # Select network topology
    if args.topology == 'line':
        network = setup_line_test()
        test_name = 'line'
    elif args.topology == 'ring':
        network = setup_ring_test()
        test_name = 'ring'
    elif args.topology == 'butterfly':
        network = setup_butterfly_test()
        test_name = 'butterfly'
    elif args.topology == 'random_conflict':
        network = setup_random_conflict_test(args.num_links, args.num_flows, args.p_conflict)
        test_name = 'random_conflict'
    elif args.topology == 'dense_5node':
        network = setup_dense_5node_test(args.p_conflict)
        test_name = 'dense_5node'
    else:
        print("Unknown topology. Using line network as default.")
        network = setup_line_test()
        test_name = 'line'

    # 传递包括z_max在内的所有参数
    explorer = run_test(network, test_name, args.xmax, args.ymax, args.zmax, args.resolution, args.time,
                        stability_threshold=args.threshold)

    # Return a success code
    return 0


if __name__ == "__main__":
    sys.exit(main())