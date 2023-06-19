import seaborn as sns
import matplotlib.pyplot as plt

# 创建示例数据
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 使用Seaborn绘制曲线图
sns.lineplot(x=x, y=y)

# 设置图形标题和轴标签
plt.title('曲线图')
plt.xlabel('X轴')
plt.ylabel('Y轴')

# 显示图形
plt.savefig('loss.png')
