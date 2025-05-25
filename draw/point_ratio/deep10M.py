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
point_ratios = np.arange(0.0000512, 0.0501, 0.001)

# 纵坐标：对应的y值
y_values = [
    0.92442, 0.92085, 0.91994, 0.91993, 0.92005, 0.91966, 0.91992, 0.91974, 0.91804, 0.91814, 0.91830, 0.91794, 0.91805, 0.91796, 0.91812, 0.91798, 0.91583, 0.91606, 0.91595, 0.91570, 0.91592, 0.91598, 0.91583, 0.91608, 0.91577, 0.91608, 0.91566, 0.91580, 0.91586, 0.91590, 0.91599, 0.91648, 0.91373, 0.91376, 0.91371, 0.91379, 0.91378, 0.91370, 0.91394, 0.91383, 0.91377, 0.91364, 0.91366, 0.91386, 0.91379, 0.91364, 0.91379, 0.91389, 0.91378, 0.91374, 0.91376
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
plt.savefig('deep10M.png', dpi=300)