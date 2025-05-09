import itertools
import random
import time

"""
independent_sets.py
提供三种独立集生成方法：ExactEnum、GreedyApprox、RandomSample
three ways of generating ISs
"""


def exact_enum_independent_sets(G):
    """
    精确枚举所有独立集（适用于小规模图）。
    精确枚举所有独立集（适用于小规模图）。
    accurately enumerate all independent sets (suitable for small graphs).
    返回列表，每个元素是顶点列表（链路集合）。
    Returns a list where each element is a list of vertices (link set).
    """
    nodes = list(G.nodes())
    indep_sets = []
    # 遍历所有子集
    # Iterate over all subsets
    for r in range(len(nodes)+1):
        for subset in itertools.combinations(nodes, r):
            valid = True
            for u, v in itertools.combinations(subset, 2):
                if G.has_edge(u, v):
                    valid = False
                    break
            if valid:
                indep_sets.append(list(subset))
    return indep_sets


def greedy_approx_independent_sets(G, queue_lengths):
    """
    贪心近似：按队列长度降序选择节点，若和已选集合无冲突则加入。
    greedy approximation: select nodes in descending queue length,
    add if no conflict with already chosen set.

    如果最终为空集，但存在正队列的链路，则兜底选取最长队列的链路。
    If result is empty but there are positive-queue links,
    pick the link with the longest queue as a fallback.
    queue_lengths: dict{node: length}
    返回单个独立集列表。
    Returns a single independent set as a list.
    """
    # DEBUG: 打印非零队列的条目
    # DEBUG: print non-zero queue entries
    # nonzeros = {u: L for u, L in queue_lengths.items() if L>0}
    # print(f"[DEBUG greedy] nonzero queue_lengths: {nonzeros}")
    # 按队列长度从大到小排序
    # Sort nodes by queue length descending
    sorted_nodes = sorted(queue_lengths.keys(),
                          key=lambda x: queue_lengths[x],
                          reverse=True)
    chosen = []
    for u in sorted_nodes:
        # 跳过空队列
        # Skip zero-length queues
        if queue_lengths[u] <= 0:
            continue

        # 检测冲突
        # Check for conflict
        conflict = False
        for v in chosen:
            if G.has_edge(u, v):
                conflict = True
                break

        if not conflict:
            chosen.append(u)

    # 兜底：如果没有选中任何链路，但确实有正队列的链路，则选最长队列的那个
    # Fallback: if nothing chosen yet but there are positive-queue links,
    # choose the one with the maximum queue length
    if not chosen:
        max_u, max_len = max(queue_lengths.items(),
                             key=lambda x: x[1])
        if max_len > 0:
            chosen = [max_u]

    return chosen


def random_sample_independent_sets(G, k, queue_lengths=None):
    """
    随机采样 k 个独立集：先随机打乱节点列表，再贪心生成。
    Randomly sample k independent sets: shuffle node list, then greedily generate.

    若给定 queue_lengths，会在采样时考虑长度优先或纯随机。
    If queue_lengths provided, sampling may consider lengths first or be purely random.
    返回独立集列表。
    Returns a list of independent sets.
    """
    nodes = list(G.nodes())
    samples = []
    for _ in range(k):
        random.shuffle(nodes)
        chosen = []
        for u in nodes:
            # 若考虑队列长度，可以按一定概率先选择大队列节点
            # If considering queue lengths, may prioritize longer queues with some probability
            conflict = any(G.has_edge(u, v) for v in chosen)
            if not conflict:
                chosen.append(u)
        samples.append(chosen)
    return samples


if __name__ == '__main__':
    # 简单的测试示例
    # Simple test example
    import networkx as nx
    G = nx.path_graph(5)
    print("Exact count:", len(exact_enum_independent_sets(G)))
    ql = {i: random.randint(0, 10) for i in G.nodes()}
    print("Greedy set:", greedy_approx_independent_sets(G, ql))
    print("Random samples:", random_sample_independent_sets(G, 3))

