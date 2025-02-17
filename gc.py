import torch
import torchvision
from thop import profile

# Model
print('==> Building model..')
model = torchvision.models.alexnet(pretrained=False)

dummy_input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.patches import Patch
# import os
# # 数据
# n_values = [1, 5, 10, 15,
#             1, 5, 10, 15,
#             1, 5, 10, 15,
# m_values = [1, 1, 1, 1,
#             2, 2, 2, 2,
#             3, 3, 3, 3,]]
# R_L = [51.91, 51.90, 51.78, 52.02,
#        51.94, 52.27, 51.88, 52.13,
#        51.97, 52.11, 51.69, 51.56,]
# Cos = [65.28, 65.33, 65.04, 64.86,
#        65.20, 65.68, 65.23, 65.24,
#        65.12, 65.46, 64.91, 65.06,]
#
# # 转换为 numpy 数组
# m_values = np.array(m_values)
# n_values = np.array(n_values)
# R_L = np.array(R_L)
# Cos = np.array(Cos)
#
# # 计算柱状图的偏移量
# x = m_values
# y = n_values
# z = np.zeros_like(R_L)  # 起点 z = 0
#
# # 将 n=1 对应的 m=1, m=2, m=3 的柱子向 n=0 偏移
# #y[0:4:4] -= 1  # 将 m=1, n=1 的柱子向 n=0 方向偏移
# y[0:10:4] -= 1  # 将 m=2, n=1 的柱子向 n=0 方向偏移
#
#
# # 柱的宽度和深度（恢复正常柱宽，分别增加柱深）
# dx = np.ones_like(x) * 0.4  # 恢复柱宽
# dy_RL = np.ones_like(y) * 2  # 增加 R-L 柱的深度
# dy_Cos = np.ones_like(y) * 2  # 增加 Cos 柱的深度
#
# # 创建 3D 图
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # 将 Cos 的柱子稍微向右偏移，增大偏移量
# y_offset_Cos = y + 2  # 增加偏移量，增大两个柱子之间的间距
#
# # 设置 R-L 和 Cos 两组柱子的偏移
# bars1 = ax.bar3d(x, y, z, dx, dy_RL, R_L, color='#ADD8E6', alpha=0.7, label='R-L')  # 浅蓝色
# bars2 = ax.bar3d(x, y_offset_Cos, z, dx, dy_Cos, Cos, color='#FFB6C1', alpha=0.7, label='Cos')  # 粉红色
#
# # 设置坐标轴标签
# ax.set_xlabel("m'")
# ax.set_ylabel("n'")
# ax.set_zlabel("Value")
#
# # 设置 y 轴的刻度和标签
# ax.set_yticks([1, 6, 11, 16])  # 手动设置间隔
# ax.set_yticklabels(['1', '5', '10', '15'])  # 自定义刻度标签
#
# # 拉伸 y 轴范围
# ax.set_ylim(0, 20)  # 拉伸范围，增强间距感
#
# # 手动创建图例句柄
# legend_handles = [
#     Patch(color='#ADD8E6', label='R-L(%)'),
#     Patch(color='#FFB6C1', label='Cos(%)')
# ]
# ax.legend(handles=legend_handles, loc='upper left')
#
# # 设置视角，旋转到更左边并降低视角
# ax.view_init(elev=30, azim=5)  # elev 设为 30，降低视角
#
# # 显示图形
# plt.show()
#
# # 保存图形为 PDF 之前渲染
# output_folder = './saved_plots/'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # 保存图像（此时确保已经渲染）
# file_path = os.path.join(output_folder, '3d_bar_chart_final.pdf')
# plt.savefig(file_path, format='pdf')
#
# # 返回文件路径
# print(f"图表已保存为: {file_path}")
# # 显示图形
# plt.show()

