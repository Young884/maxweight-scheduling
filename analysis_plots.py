# analysis_plots.py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def generate_analysis_plots(output_dir="output", plots_dir="plots"):
    """Generate analysis plots from output files"""
    # Create plots directory
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Generate all plots
    plot_capacity_region(output_dir, plots_dir)
    plot_queue_evolution(output_dir, plots_dir)
    plot_scheduling_decisions(output_dir, plots_dir)
    plot_node_activity(output_dir, plots_dir)
    plot_flow_rates(output_dir, plots_dir)

    print(f"All analysis plots have been saved to: {plots_dir}")


def plot_capacity_region(output_dir, plots_dir):
    """Plot capacity region from CSV file"""
    capacity_file = os.path.join(output_dir, "capacity_region.csv")

    if not os.path.exists(capacity_file):
        print(f"Warning: {capacity_file} not found. Skipping capacity region plot.")
        return

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
    stable_points = np.array(stable_points)
    unstable_points = np.array(unstable_points)

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
        plt.title('Capacity Region')
        plt.grid(True)
        plt.legend()

        plt.savefig(os.path.join(plots_dir, "capacity_region_2d.png"))
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
        ax.set_title('3D Capacity Region')
        ax.legend()

        plt.savefig(os.path.join(plots_dir, "capacity_region_3d.png"))
        plt.close()

    else:
        print(f"Warning: Cannot visualize capacity region with {dim} dimensions.")


def plot_queue_evolution(output_dir, plots_dir):
    """Plot queue evolution for each node and flow"""
    queue_file = os.path.join(output_dir, "queue_evolution.csv")

    if not os.path.exists(queue_file):
        print(f"Warning: {queue_file} not found. Skipping queue evolution plots.")
        return

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
    plt.title('Queue Evolution for All Nodes and Flows')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(plots_dir, "all_queues.png"))
    plt.close()

    # 2. Separate plots for each node-flow pair
    for pair, values in queue_data.items():
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, values)
        plt.xlabel('Time Step')
        plt.ylabel('Queue Length')
        plt.title(f'Queue Evolution for {pair}')
        plt.grid(True)

        # Save with sanitized filename
        filename = pair.replace(" ", "_").replace(":", "_")
        plt.savefig(os.path.join(plots_dir, f"queue_{filename}.png"))
        plt.close()


def plot_scheduling_decisions(output_dir, plots_dir):
    """Plot scheduling decisions over time"""
    scheduling_file = os.path.join(output_dir, "scheduling_decisions.csv")

    if not os.path.exists(scheduling_file):
        print(f"Warning: {scheduling_file} not found. Skipping scheduling decision plots.")
        return

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
    plt.title('Scheduling Decisions Over Time')

    plt.savefig(os.path.join(plots_dir, "scheduling_decisions.png"))
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
    plt.title('Link Activation Frequency')

    plt.savefig(os.path.join(plots_dir, "link_activation_frequency.png"))
    plt.close()


def plot_node_activity(output_dir, plots_dir):
    """Plot node activity based on scheduling decisions"""
    scheduling_file = os.path.join(output_dir, "scheduling_decisions.csv")

    if not os.path.exists(scheduling_file):
        print(f"Warning: {scheduling_file} not found. Skipping node activity plots.")
        return

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
    plt.title('Node Activity Over Time')

    plt.savefig(os.path.join(plots_dir, "node_activity.png"))
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
    plt.title('Node Activation Frequency')

    plt.savefig(os.path.join(plots_dir, "node_activation_frequency.png"))
    plt.close()


def plot_flow_rates(output_dir, plots_dir):
    """Plot analysis of tested flow rates"""
    capacity_file = os.path.join(output_dir, "capacity_region.csv")

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
    plt.title('Distribution of Tested Flow Rates')
    plt.legend()

    plt.savefig(os.path.join(plots_dir, "flow_rate_distribution.png"))
    plt.close()

    # For 2D case, plot boundary approximation
    if num_flows == 2 and stable_points and unstable_points:
        plt.figure(figsize=(10, 8))

        # Convert to numpy arrays
        stable = np.array(stable_points)
        unstable = np.array(unstable_points)

        # Plot points
        plt.scatter(stable[:, 0], stable[:, 1], c='green', marker='o', label='Stable')
        plt.scatter(unstable[:, 0], unstable[:, 1], c='red', marker='x', label='Unstable')

        # Try to approximate boundary
        if len(unstable) > 0:
            # Simple convex hull approximation (would need scipy for better boundary)
            max_stable_x = max([p[0] for p in stable_points]) if stable_points else 0
            max_stable_y = max([p[1] for p in stable_points]) if stable_points else 0

            min_unstable_x = min([p[0] for p in unstable_points])
            min_unstable_y = min([p[1] for p in unstable_points])

            # Draw approximate boundary lines
            plt.plot([0, max_stable_x, min_unstable_x], [max_stable_y, max_stable_y, 0],
                     'k--', label='Estimated Boundary')

        plt.xlabel('Flow 0 Rate')
        plt.ylabel('Flow 1 Rate')
        plt.title('Capacity Region with Estimated Boundary')
        plt.grid(True)
        plt.legend()

        plt.savefig(os.path.join(plots_dir, "capacity_boundary_estimate.png"))
        plt.close()


if __name__ == "__main__":
    generate_analysis_plots()