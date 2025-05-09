import random
import matplotlib.pyplot as plt
import numpy as np
'''
Only for testing
'''
# Simulation parameters
SIM_SLOTS = 10000
ARRIVAL_PROB1 = 0.3  # Flow1 arrival probability
ARRIVAL_PROB2 = 0.2  # Flow2 arrival probability
RUNS = 10

# Function to run the simulation once
def run_simulation():
    Q_AB_f1 = 0
    Q_BC_f1 = 0
    Q_BC_f2 = 0
    deliv_f1 = 0
    deliv_f2 = 0
    
    for _ in range(SIM_SLOTS):
        # Arrivals
        if random.random() < ARRIVAL_PROB1:
            Q_AB_f1 += 1
        if random.random() < ARRIVAL_PROB2:
            Q_BC_f2 += 1
            
        # Weights
        w_AB = Q_AB_f1
        w_BC = Q_BC_f1 + Q_BC_f2
        
        # Max-Weight decision
        if w_AB > w_BC:
            chosen = "AB"
        elif w_BC > w_AB:
            chosen = "BC"
        else:
            chosen = random.choice(["AB", "BC"])
        
        # Service
        if chosen == "AB" and Q_AB_f1 > 0:
            Q_AB_f1 -= 1
            Q_BC_f1 += 1
        elif chosen == "BC":
            if Q_BC_f1 >= Q_BC_f2 and Q_BC_f1 > 0:
                Q_BC_f1 -= 1
                deliv_f1 += 1
            elif Q_BC_f2 > 0:
                Q_BC_f2 -= 1
                deliv_f2 += 1
    
    return deliv_f1/SIM_SLOTS, deliv_f2/SIM_SLOTS

# Run multiple simulations
throughputs_f1 = []
throughputs_f2 = []

for i in range(RUNS):
    t1, t2 = run_simulation()
    throughputs_f1.append(t1)
    throughputs_f2.append(t2)

# Plotting
plt.figure()
plt.plot(range(1, RUNS+1), throughputs_f1, marker='o', label='Flow1 Throughput')
plt.plot(range(1, RUNS+1), throughputs_f2, marker='o', label='Flow2 Throughput')
plt.xlabel('Run Number')
plt.ylabel('Throughput')
plt.title('Throughput over Multiple Simulation Runs')
plt.legend()
plt.grid(True)
plt.show()

# 定义网格
# defining 
λ1 = np.linspace(0, 1, 200)
λ2 = np.linspace(0, 1, 200)
L1, L2 = np.meshgrid(λ1, λ2)

# 可行条件
feasible = (L1 >= 0) & (L2 >= 0) & (L1 + L2 <= 1)

# 绘制填充
plt.figure()
plt.contourf(L1, L2, feasible, levels=[-0.5, 0.5, 1.5], alpha=0.3)

# 绘制边界 λ1 + λ2 = 1
plt.plot([0, 1], [1, 0], linewidth=2)

plt.xlabel('Arrival rate λ1 (Flow1)')
plt.ylabel('Arrival rate λ2 (Flow2)')
plt.title('Feasible Region: λ1 + λ2 ≤ 1')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
