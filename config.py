import numpy as np

"""
config: global args
"""

# --------------------
# net and flow arg
# --------------------
# link number
NUM_LINKS = [5, 10, 20]
# flow number
NUM_FLOWS = [2, 5, 10]

# time slot
SIM_SLOTS = 50_000
# experiment time
NUM_RUNS = 8

# RandomSample k
RSAMPLE_K = 50

# --------------------
# arraval mode
# --------------------
# ARRIVAL_RATES as a dictionary, key is mode name, value is function of flow
# 为一个字典，键为模式名，值为根据流数 F 生成到达率列表的函数
ARRIVAL_RATES = {
    # 所有流到达率均为 0.1
    'low': lambda F: [0.1 for _ in range(F)],
    # 所有流到达率均为 0.3
    'medium': lambda F: [0.3 for _ in range(F)],
    # 随机均匀分布在 [0, 0.5)
    'random': lambda F: list(np.random.rand(F) * 0.5)
}

# --------------------
# independent set method
# 独立集生成方法
# --------------------
# 支持 ExactEnum、GreedyApprox、RandomSample
METHODS = ['ExactEnum', 'GreedyApprox', 'RandomSample']

# --------------------
# other args
# 其它全局参数
# --------------------
# 随机种子，保证可复现
SEED = 42
np.random.seed(SEED)

# 日志目录
# result dir
RESULTS_DIR = 'results'
