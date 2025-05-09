import networkx as nx
import matplotlib.pyplot as plt
from network import build_conflict_graph


# 5 flow, conflict rate =0.3
G = build_conflict_graph(num_links=5, topology='random', p_conflict=0.3)

plt.figure(figsize=(5,5))
pos = nx.circular_layout(G)  # 或者 spring_layout、kamada_kawai_layout
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=800)
plt.title("Conflict Graph (random, 5 links, p_conflict=0.3)")
plt.show()
