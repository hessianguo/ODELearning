#%%

import matplotlib.pyplot as plt
import numpy as np

def plot_lc(rho, eta, marker, reg_param):
    plt.plot(rho, eta, marker=marker, label="Data", color='b')
    plt.xlabel('rho')
    plt.ylabel('eta')
    plt.legend()

# 示例数据：
rho = np.logspace(-2, 2, 100)  # 示例rho数据
eta = np.logspace(-3, 3, 100)  # 示例eta数据
marker = 'o'
reg_param = 0.5
rho_c = 10
eta_c = 100
reg_c = 0.1

# 步骤1：调用plot_lc函数绘制初始数据
plot_lc(rho, eta, marker, reg_param)

# 步骤2：获取当前坐标轴，并绘制L-curve的拐角
ax = plt.gca()  # 获取当前坐标轴
plt.loglog([np.min(rho) / 100, rho_c], [eta_c, eta_c], ':r')  # L-curve 拐角（水平线）
plt.loglog([rho_c, rho_c], [np.min(eta) / 100, eta_c], ':r')  # L-curve 拐角（垂直线）

# 步骤3：调整标题和坐标轴范围
plt.title(f'L-curve corner at {reg_c**2}')
plt.axis([ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1]])

# 显示图形
plt.show()

# %%
