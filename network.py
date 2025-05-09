import numpy as np
import networkx as nx

"""
network.py
用于生成冲突图(conflict graph)及定义流集合
用于生成冲突图(conflict graph)及定义流集合
For generating conflict graphs and defining flow sets
"""


def build_conflict_graph(num_links, topology='random', p_conflict=0.2):
    """
    构造冲突图：节点代表链路编号 (0..num_links-1)，边代表两条链路冲突，不能同时激活。
    Construct conflict graph: nodes represent link IDs (0..num_links-1),
    edges represent conflicts between two links that cannot be active simultaneously.

    topology: 'chain', 'ring', 或 'random'
    topology: 'chain', 'ring', or 'random'
    p_conflict: 随机图时的连接概率
    p_conflict: connection probability for random graph
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_links))
    if topology == 'chain':
        # 线性冲突：i 与 i+1 冲突
        # Chain conflict: i conflicts with i+1
        G.add_edges_from([(i, i+1) for i in range(num_links-1)])
    elif topology == 'ring':
        # 环形冲突
        # Ring conflict
        G.add_edges_from([(i, (i+1) % num_links) for i in range(num_links)])
    else:
        # 随机冲突图
        # Random conflict graph
        for i in range(num_links):
            for j in range(i+1, num_links):
                if np.random.rand() < p_conflict:
                    G.add_edge(i, j)
    return G


def build_flows(G, num_flows, max_hops=None):
    """
    在冲突图节点上定义流，每条流从一个链路走到另一个链路，路径为简单最短路径。
    Define flows on conflict graph nodes; each flow goes from one link to another via the simple shortest path.
    返回 flows: List[List[int]]，每个子列表为链路 ID 列表。
    Returns flows: List[List[int]], each sublist is a list of link IDs.
    """
    nodes = list(G.nodes)
    flows = []
    attempts = 0
    while len(flows) < num_flows and attempts < num_flows * 5:
        src, dst = np.random.choice(nodes, 2, replace=False)
        try:
            path = nx.shortest_path(G, src, dst)
            if max_hops is None or len(path)-1 <= max_hops:
                flows.append(path)
        except nx.NetworkXNoPath:
            pass
        attempts += 1
    return flows
