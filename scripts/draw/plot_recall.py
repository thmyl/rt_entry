import re
import matplotlib.pyplot as plt

# 1. 读取文件内容
with open("../all_output.txt", "r") as f:
    text = f.read()

# 2. 正则匹配 recall@10 = 0.981970 这种格式
pattern = r"recall@10\s*=\s*([0-9]*\.?[0-9]+)"
recalls = re.findall(pattern, text)

# 字符串转 float
recalls = [float(r) for r in recalls]

# 3. 检查是否数量正确
print(f"共找到 {len(recalls)} 个 recall@10 值")

if len(recalls) == 0:
    print("没有匹配到数据，请检查文件内容是否含有 'recall@10 = xxx' 格式")
    exit(0)

# 4. 绘图
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(recalls)+1), recalls, marker='o')
plt.xlabel("Run ID (1~100)")
plt.ylabel("recall@10")
plt.title("Recall@10 over 100 Runs")
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig("recall_curve.png", dpi=300)
