# maxweight_scheduling.py
# A framework for MaxWeight scheduling and capacity region exploration

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, chain
from collections import defaultdict
import random
import csv


# Network Components
class Node:
    def __init__(self, id):
        self.id = id
        self.queues = defaultdict(int)  # Flow id -> queue length

    def __repr__(self):
        return f"Node({self.id})"


class Link:
    def __init__(self, source, destination, capacity=1):
        self.source = source
        self.destination = destination
        self.capacity = capacity

    def __repr__(self):
        return f"Link({self.source.id}->{self.destination.id}, cap={self.capacity})"


class Flow:
    def __init__(self, id, source, destination, rate):
        self.id = id
        self.source = source
        self.destination = destination
        self.rate = rate
        # Create path
        self.path = []  # List of nodes in the path

    def set_path(self, path):
        self.path = path

    def __repr__(self):
        return f"Flow({self.id}, {self.source.id}->{self.destination.id}, rate={self.rate})"


class Network:
    def __init__(self):
        self.nodes = []
        self.links = []
        self.flows = []
        self.independent_sets = []

    def add_node(self, node):
        self.nodes.append(node)
        return node

    def add_link(self, source, destination, capacity=1):
        # Create the link
        link = Link(source, destination, capacity)
        self.links.append(link)
        return link

    def add_flow(self, id, source, destination, rate):
        flow = Flow(id, source, destination, rate)
        self.flows.append(flow)
        return flow

    def is_conflicting(self, link1, link2):
        """Check if two links conflict based on half-duplex constraint"""
        # In half-duplex, a node cannot be involved in more than one transmission
        return (link1.source == link2.source or
                link1.source == link2.destination or
                link1.destination == link2.source or
                link1.destination == link2.destination)

    def generate_independent_sets(self):
        """Generate all possible independent sets based on link conflicts"""
        # Start with empty set and sets with single links
        self.independent_sets = [[]]
        for link in self.links:
            self.independent_sets.append([link])

        # Try to add more links to existing independent sets
        for i in range(len(self.links)):
            current_sets = self.independent_sets.copy()
            for ind_set in current_sets:
                if len(ind_set) == 0 or self.links[i] in ind_set:
                    continue

                # Check if adding this link maintains independence
                is_independent = True
                for link in ind_set:
                    if self.is_conflicting(link, self.links[i]):
                        is_independent = False
                        break

                if is_independent:
                    new_set = ind_set.copy()
                    new_set.append(self.links[i])
                    self.independent_sets.append(new_set)

        # Remove duplicates and empty set
        unique_sets = []
        for ind_set in self.independent_sets:
            if ind_set and ind_set not in unique_sets:
                unique_sets.append(ind_set)

        self.independent_sets = unique_sets
        return self.independent_sets

    def find_path(self, source, destination):
        """Find a simple path from source to destination (BFS)"""
        visited = {node: False for node in self.nodes}
        queue = [(source, [source])]
        visited[source] = True

        while queue:
            (node, path) = queue.pop(0)

            # Find all outgoing links from this node
            for link in self.links:
                if link.source == node and not visited[link.destination]:
                    if link.destination == destination:
                        return path + [destination]

                    visited[link.destination] = True
                    queue.append((link.destination, path + [link.destination]))

        return None  # No path found

    def setup_flows(self):
        """Setup paths for all flows"""
        for flow in self.flows:
            path = self.find_path(flow.source, flow.destination)
            if path:
                flow.set_path(path)
            else:
                print(f"Warning: No path found for {flow}")


# MaxWeight Scheduler
class MaxWeightScheduler:
    def __init__(self, network):
        self.network = network

    def calculate_weight(self, independent_set):
        """Calculate weight of an independent set based on queue backlogs"""
        weight = 0
        for link in independent_set:
            source_node = link.source
            dest_node = link.destination

            # For each flow passing through this link
            for flow in self.network.flows:
                # Check if this flow passes through the link
                if source_node in flow.path and dest_node in flow.path:
                    # Get indices in the path
                    src_idx = flow.path.index(source_node)
                    dst_idx = flow.path.index(dest_node)

                    # Check if they're consecutive
                    if dst_idx == src_idx + 1:
                        # Calculate differential backlog
                        queue_diff = source_node.queues[flow.id] - dest_node.queues[flow.id]
                        if queue_diff > 0:
                            weight += queue_diff * link.capacity

        return weight

    def schedule(self):
        """Select the independent set with maximum weight"""
        max_weight = 0
        selected_set = None

        for ind_set in self.network.independent_sets:
            weight = self.calculate_weight(ind_set)
            if weight > max_weight:
                max_weight = weight
                selected_set = ind_set

        return selected_set


# Simulation
class Simulation:
    def __init__(self, network, max_time=1000, stability_threshold=5000):
        self.network = network
        self.max_time = max_time
        self.stability_threshold = stability_threshold
        self.scheduler = MaxWeightScheduler(network)
        self.current_time = 0
        self.queue_history = []
        self.is_stable = True

    def initialize(self):
        """Reset simulation state"""
        self.current_time = 0
        self.queue_history = []
        self.is_stable = True

        # Clear all queues
        for node in self.network.nodes:
            node.queues.clear()

    def generate_packets(self):
        """Generate new packets based on flow rates"""
        for flow in self.network.flows:
            # Bernoulli arrivals with probability equal to flow rate
            if random.random() < flow.rate:
                # Add packet to source node's queue
                flow.source.queues[flow.id] += 1

    def transmit_packets(self, independent_set):
        """Transmit packets according to the selected independent set"""
        if not independent_set:
            return

        for link in independent_set:
            source_node = link.source
            dest_node = link.destination

            # For each flow passing through this link
            for flow in self.network.flows:
                # Check if this flow passes through the link
                if source_node in flow.path and dest_node in flow.path:
                    # Get indices in the path
                    src_idx = flow.path.index(source_node)
                    dst_idx = flow.path.index(dest_node)

                    # Check if they're consecutive
                    if dst_idx == src_idx + 1 and source_node.queues[flow.id] > 0:
                        # Transmit packet (up to link capacity)
                        transmit_count = min(source_node.queues[flow.id], link.capacity)
                        source_node.queues[flow.id] -= transmit_count

                        # If not the destination, add to next node's queue
                        if dest_node != flow.destination:
                            dest_node.queues[flow.id] += transmit_count

    def check_stability(self):
        """Check if all queues are bounded"""
        for node in self.network.nodes:
            for flow_id, queue_length in node.queues.items():
                if queue_length > self.stability_threshold:
                    return False
        return True

    def record_state(self):
        """Record current queue states"""
        state = {}
        for node in self.network.nodes:
            state[node.id] = node.queues.copy()
        self.queue_history.append(state)

    def run(self):
        """Run the simulation for max_time steps"""
        self.initialize()

        for t in range(self.max_time):
            self.current_time = t

            # Generate new packets
            self.generate_packets()

            # Run scheduler
            selected_set = self.scheduler.schedule()

            # Transmit packets
            self.transmit_packets(selected_set)

            # Record state
            self.record_state()

            # Check stability
            if not self.check_stability():
                self.is_stable = False
                break

        return self.is_stable, self.queue_history


# Capacity Region Explorer
class CapacityRegionExplorer:
    def __init__(self, network, simulation_time=1000):
        self.network = network
        self.simulation_time = simulation_time
        self.stable_points = []
        self.unstable_points = []

    def test_rate_vector(self, rate_vector):
        """Test if a specific rate vector is stable"""
        # Set flow rates
        for i, flow in enumerate(self.network.flows):
            flow.rate = rate_vector[i]

        # Run simulation
        simulation = Simulation(self.network, self.simulation_time)
        is_stable, _ = simulation.run()

        # Record result
        if is_stable:
            self.stable_points.append(rate_vector)
        else:
            self.unstable_points.append(rate_vector)

        return is_stable

    def explore_line(self, direction, step=0.1, max_iterations=20):
        """Explore along a line to find boundary point"""
        current_vector = [0] * len(self.network.flows)

        for _ in range(max_iterations):
            # Increment rates
            for i in range(len(current_vector)):
                current_vector[i] += direction[i] * step

            # Test stability
            if not self.test_rate_vector(current_vector.copy()):
                # Found unstable point, return last stable point
                return current_vector

        return current_vector

    def explore_region_2d(self, x_max=1.0, y_max=1.0, resolution=10):
        """Explore 2D capacity region (for two flows)"""
        if len(self.network.flows) != 2:
            raise ValueError("This method works only for networks with exactly 2 flows")

        x_values = np.linspace(0, x_max, resolution)
        y_values = np.linspace(0, y_max, resolution)

        for x in x_values:
            for y in y_values:
                self.test_rate_vector([x, y])

    def explore_region_3d(self, x_max=1.0, y_max=1.0, z_max=1.0, resolution=10):
        """Explore 3D capacity region (for three flows)"""
        if len(self.network.flows) != 3:
            raise ValueError("This method works only for networks with exactly 3 flows")

        x_values = np.linspace(0, x_max, resolution)
        y_values = np.linspace(0, y_max, resolution)
        z_values = np.linspace(0, z_max, resolution)

        for x in x_values:
            for y in y_values:
                for z in z_values:
                    self.test_rate_vector([x, y, z])

    # def visualize_region_2d(self):
    #     """Visualize 2D capacity region"""
    #     if len(self.network.flows) != 2:
    #         raise ValueError("This method works only for networks with exactly 2 flows")
    #
    #     stable_x = [point[0] for point in self.stable_points]
    #     stable_y = [point[1] for point in self.stable_points]
    #
    #     unstable_x = [point[0] for point in self.unstable_points]
    #     unstable_y = [point[1] for point in self.unstable_points]
    #
    #     plt.figure(figsize=(10, 8))
    #     plt.scatter(stable_x, stable_y, c='green', marker='o', label='Stable')
    #     plt.scatter(unstable_x, unstable_y, c='red', marker='x', label='Unstable')
    #
    #     plt.xlabel(f'Flow {self.network.flows[0].id} Rate')
    #     plt.ylabel(f'Flow {self.network.flows[1].id} Rate')
    #     plt.title('Capacity Region Exploration')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig('capacity_region_2d.png')
    #     plt.show()


# Utility Functions
def create_line_network(n, capacity=1):
    """Create a line network with n nodes"""
    network = Network()

    # Create nodes
    nodes = []
    for i in range(n):
        nodes.append(network.add_node(Node(i)))

    # Create links
    for i in range(n - 1):
        network.add_link(nodes[i], nodes[i + 1], capacity)

    return network


def create_grid_network(n, m, capacity=1):
    """Create an n x m grid network"""
    network = Network()

    # Create nodes
    nodes = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(network.add_node(Node(i * m + j)))
        nodes.append(row)

    # Create horizontal links
    for i in range(n):
        for j in range(m - 1):
            network.add_link(nodes[i][j], nodes[i][j + 1], capacity)
            network.add_link(nodes[i][j + 1], nodes[i][j], capacity)  # Bidirectional

    # Create vertical links
    for i in range(n - 1):
        for j in range(m):
            network.add_link(nodes[i][j], nodes[i + 1][j], capacity)
            network.add_link(nodes[i + 1][j], nodes[i][j], capacity)  # Bidirectional

    return network

def save_independent_sets(network, filename):
    """保存独立集信息到文件"""
    with open(filename, 'w') as f:
        f.write("独立集列表\n")
        f.write("=========\n\n")

        for i, ind_set in enumerate(network.independent_sets):
            f.write(f"独立集 {i + 1}:\n")
            for link in ind_set:
                f.write(f"  Link {link.source.id} -> {link.destination.id}\n")
            f.write("\n")


def save_flow_paths(network, filename):
    """保存流路径信息到文件"""
    with open(filename, 'w') as f:
        f.write("流配置与路径\n")
        f.write("===========\n\n")

        for flow in network.flows:
            f.write(f"Flow {flow.id}:\n")
            f.write(f"  源节点: {flow.source.id}\n")
            f.write(f"  目标节点: {flow.destination.id}\n")
            f.write(f"  速率: {flow.rate}\n")

            f.write("  路径: ")
            if flow.path:
                path_str = " -> ".join([str(node.id) for node in flow.path])
                f.write(path_str + "\n")
            else:
                f.write("未设置\n")
            f.write("\n")


def save_capacity_region_data(explorer, filename):
    """保存容量区域数据到CSV文件"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # 写入表头
        header = ["Point Type"]
        for i in range(len(explorer.network.flows)):
            header.append(f"Flow {i} Rate")
        writer.writerow(header)

        # 写入稳定点
        for point in explorer.stable_points:
            row = ["Stable"] + list(point)
            writer.writerow(row)

        # 写入不稳定点
        for point in explorer.unstable_points:
            row = ["Unstable"] + list(point)
            writer.writerow(row)


def save_queue_evolution(simulation, filename):
    """保存队列长度演化到CSV文件"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # 创建表头
        header = ["Time Step"]
        for node in simulation.network.nodes:
            for flow in simulation.network.flows:
                header.append(f"Node {node.id} Flow {flow.id}")
        writer.writerow(header)

        # 写入每个时间步的队列长度
        for t, state in enumerate(simulation.queue_history):
            row = [t]
            for node in simulation.network.nodes:
                for flow in simulation.network.flows:
                    row.append(state[node.id].get(flow.id, 0))
            writer.writerow(row)


def save_scheduling_decisions(network, simulation, filename):
    """模拟并保存调度决策到CSV文件"""
    # 重置模拟
    simulation.initialize()
    scheduler = simulation.scheduler

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # 创建表头 - 所有链路
        header = ["Time Step"]
        link_labels = []
        for link in network.links:
            label = f"Link {link.source.id}->{link.destination.id}"
            link_labels.append(label)
            header.append(label)
        writer.writerow(header)

        # 运行模拟并记录每个时间步的调度决策
        for t in range(simulation.max_time):
            # 生成数据包
            simulation.generate_packets()

            # 获取调度决策
            selected_set = scheduler.schedule()

            # 准备行数据
            row = [t]
            for link in network.links:
                if selected_set and link in selected_set:
                    row.append(1)  # 激活
                else:
                    row.append(0)  # 未激活
            writer.writerow(row)

            # 传输数据包
            simulation.transmit_packets(selected_set)

            # 检查稳定性
            if not simulation.check_stability():
                break

# Example usage
def main():
    """
    Simple example of using the MaxWeight scheduling framework.
    For more comprehensive tests, use network_topology_tests.py
    """
    # Create a simple line network
    network = create_line_network(4)

    # Add two flows
    flow1 = network.add_flow(0, network.nodes[0], network.nodes[3], 0.3)
    flow2 = network.add_flow(1, network.nodes[0], network.nodes[2], 0.3)

    # Setup flow paths
    network.setup_flows()

    # Generate independent sets
    network.generate_independent_sets()
    print(f"Found {len(network.independent_sets)} independent sets")

    # Run a simple simulation
    simulation = Simulation(network)
    is_stable, _ = simulation.run()
    print(f"Network stability: {'Stable' if is_stable else 'Unstable'}")

    print("For more comprehensive tests, run network_topology_tests.py")


if __name__ == "__main__":
    main()