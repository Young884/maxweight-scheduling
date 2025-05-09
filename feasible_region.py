import numpy as np
from scipy.spatial import ConvexHull
from network import build_conflict_graph
from independent_sets import exact_enum_independent_sets
import matplotlib.pyplot as plt
from config import RESULTS_DIR
import networkx as nx

def plot_feasible_region():
    # 1) 生成同一张图的所有独立集
    # generate all Iss for a topology
    L = 5
    G = build_conflict_graph(L, topology='random', p_conflict=0.3)
    Isets = exact_enum_independent_sets(G)

    plt.figure(figsize=(5,5))
    pos = nx.circular_layout(G)  # 或者 spring_layout、kamada_kawai_layout
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=800)
    plt.title("Conflict Graph (random, 5 links, p_conflict=0.3)")
    plt.show()

    # 2) 把每个独立集转换成一个 0/1 向量
    # turn every Iss into a 0/1 vector
    all_vectors = np.array([
        [1 if i in S else 0 for i in range(L)]
        for S in Isets
    ])

    # 3) 从向量里截取“第1维”和“第2维”——即链路1和链路2的激活指标

    pts12 = all_vectors[:, [0, 1]] 

    # 4) 计算凸包
    # convex hull
    hull = ConvexHull(pts12)

    # 5) 绘图
    plt.figure(figsize=(6,6))
    # 原始可行点
    plt.plot(pts12[:,0], pts12[:,1], 'o', label='Independent sets (0,1)')
    # 凸包边界
    for simp in hull.simplices:
        plt.plot(pts12[simp,0], pts12[simp,1], 'k-')

    plt.xlabel('Link 0 activation')
    plt.ylabel('Link 1 activation')
    plt.title('Feasible Region projected on (0,1)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out = f"{RESULTS_DIR}/feasible_region_01.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved to {out}")

if __name__ == "__main__":
    plot_feasible_region()