# point_ratios = np.arange(0.0006, 0.0101, 0.0004)
# 0.92363, 0.92585, 0.92588, 0.92600, 0.92351, 0.92401, 0.92347, 0.92350, 0.92394, 0.92067, 0.92110, 0.92095, 0.92079, 0.92099, 0.92077, 0.92066, 0.92088, 0.92106, 0.91931, 0.91629, 0.91626, 0.91623, 0.91665, 0.91640

import numpy as np
import matplotlib.pyplot as plt

def list_(start, end):
    point_ratios = []
    current = start
    while current <= end:
        point_ratios.append(current)
        current*=2
    return np.array(point_ratios)

# 横坐标：point_ratios
# point_ratios = list_(0.000256, 1.0)
point_ratios = np.arange(0.000128, 0.0501, 0.001)

# 纵坐标：对应的y值
y_values = [
    0.93097, 0.92894, 0.92932, 0.92931, 0.92788, 0.92785, 0.92786, 0.92785, 0.92775, 0.92781, 0.92787, 0.92769, 0.92780, 0.92778, 0.92782, 0.92782, 0.92605, 0.92607, 0.92598, 0.92605, 0.92590, 0.92601, 0.92604, 0.92596, 0.92597, 0.92599, 0.92601, 0.92601, 0.92604, 0.92602, 0.92594, 0.92524, 0.92505, 0.92495, 0.92504, 0.92507, 0.92497, 0.92495, 0.92496, 0.92504, 0.92504, 0.92504, 0.92499, 0.92500, 0.92500, 0.92502, 0.92504, 0.92506, 0.92509, 0.92499
]

# 检查一下两者长度是否匹配
assert len(point_ratios) == len(y_values), "横纵坐标数量不一致！"

# 开始画图
plt.figure(figsize=(8, 5))
plt.plot(point_ratios, y_values, marker='o', linestyle='-', label='Curve')

plt.xlabel('Point Ratios')
plt.ylabel('Y Values')
plt.title('Curve of Point Ratios vs Y Values')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('deep1M.png', dpi=300)