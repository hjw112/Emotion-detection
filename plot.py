import matplotlib.pyplot as plt

def plot_two_groups(data1, data2, label1='accused guilt', label2='biased', color1='blue', color2='red'):
    # 绘制第一组数据
    x = [1, 2, 5, 10, 15, 20, 23]
    plt.plot(x, data1, label=label1, color=color1)

    # 绘制第二组数据
    plt.plot(x, data2, label=label2, color=color2)

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()

# 示例数据
group1 = [0.5749, 0.69896, 0.77826, 0.69174, 0.59617, 0.51967, 0.48582]
group2 = [0.16194, 0.22145, 0.34644, 0.51351, 0.67305, 0.80054, 0.85696]

# 调用函数绘制图形
plot_two_groups(group1, group2)
