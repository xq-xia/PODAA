import matplotlib.pyplot as plt
import shap
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import matplotlib.cm as cm
from matplotlib.collections import PathCollection
import numpy as np


plt.rcParams['font.size'] = 24 # 基础字号
# # 设置中文字体为宋体，英文和数字使用Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

def draw_shap_interact(feature_name,cmap, shap_interaction_values, x_feature, cbar_feature, x_increment, y_increment):
    viridis_cmap = plt.get_cmap(cmap)
    shap.dependence_plot(
        (x_feature, cbar_feature),
        shap_interaction_values,
        feature_name,
        dot_size=80,  # 增大散点大小（默认16）
        alpha=0.7,  # 提高透明度（默认0.5）
        cmap=viridis_cmap,  # 更改色度带为viridis（默认coolwarm）
        x_jitter=0.3 , # 增加x轴抖动（默认0.3）
        show=False  # 禁止自动显示
    )

    # 获取当前坐标轴
    ax = plt.gca()

    for collection in ax.collections:
        if isinstance(collection, PathCollection):  # 使用直接导入的PathCollection
            # 方法1：直接设置大小（适用于所有点相同大小）
            collection.set_sizes([40])  # 简化为统一大小设置

            # 方法2：精确控制每个点的大小（如果需要）
            # sizes = np.array([40] * len(collection.get_sizes()))
            # collection.set_sizes(sizes)

            collection.set_cmap(viridis_cmap)
            collection.set_alpha(0.6)
    ax.set_xlabel(x_feature, fontsize=24,fontweight='bold')
    ax.set_ylabel('Shap interaction value', fontsize=24,fontweight='bold')
    # 3. 设置x轴刻度（增量为5）
    x_min, x_max = ax.get_xlim()  # 获取当前x轴范围
    ax.set_xticks(np.arange(np.floor(x_min / x_increment) * x_increment + x_increment, np.ceil(x_max / x_increment) * x_increment, x_increment))  # 从5的倍数开始，间隔5
    #
    # # 4. 设置y轴刻度（增量为0.25）
    y_min, y_max = ax.get_ylim()  # 获取当前y轴范围
    ax.set_yticks(np.arange(np.floor(y_min / y_increment) * y_increment, np.ceil(y_max / y_increment) * y_increment, y_increment))  # 从0.25的倍数开始，间隔0.25
    ax.tick_params(axis='both', direction='out', which='major', labelsize=18)  # 加大刻度数字

    ax.grid(
        True,
        which='major',
        linestyle=':',
        linewidth=0.8,
        color='gray',
        alpha=0.5
    )
    # 3. 添加加粗的y=0参考线（红色实线，线宽2）
    ax.axhline(
        y=0,
        color='red',
        linestyle='--',  # 实线
        linewidth=2,  # 加粗线宽
        alpha=0.8,  # 降低透明度
        zorder=3  # 确保在最上层
    )

    # 4. 强制显示所有四条坐标轴边框
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)  # 显示边框
        ax.spines[spine].set_linewidth(0.5)  # 边框粗细
        ax.spines[spine].set_color('black')  # 边框颜色

    # 获取色度条并设置字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)  # 设置色度条的字体大小为12
    cbar.set_label(cbar_feature, fontsize=24,fontweight='bold')  # 设置色度条的标题
    plt.tight_layout()
    plt.show()

def draw_shap_total(feature_name,cmap, shap_values, x_feature, cbar_feature, x_increment, y_increment):
    viridis_cmap = plt.get_cmap(cmap)
    shap.dependence_plot(x_feature, shap_values, feature_name, interaction_index=cbar_feature,dot_size=20,
        alpha=0.7,
        cmap=viridis_cmap,
        x_jitter=0.2,
        show=False)

    # 获取当前坐标轴
    ax = plt.gca()

    for collection in ax.collections:
        if isinstance(collection, PathCollection):  # 使用直接导入的PathCollection
            # 方法1：直接设置大小（适用于所有点相同大小）
            collection.set_sizes([40])  # 简化为统一大小设置

            # 方法2：精确控制每个点的大小（如果需要）
            # sizes = np.array([40] * len(collection.get_sizes()))
            # collection.set_sizes(sizes)

            collection.set_cmap(viridis_cmap)
            collection.set_alpha(0.6)
    ax.set_xlabel(x_feature, fontsize=24,fontweight='bold')
    ax.set_ylabel('Shap value', fontsize=24, fontweight='bold')
    # 3. 设置x轴刻度（增量为5）
    x_min, x_max = ax.get_xlim()  # 获取当前x轴范围
    ax.set_xticks(np.arange(np.floor(x_min / x_increment) * x_increment + x_increment, np.ceil(x_max / x_increment) * x_increment, x_increment))  # 从5的倍数开始，间隔5
    #
    # # 4. 设置y轴刻度（增量为0.25）
    y_min, y_max = ax.get_ylim()  # 获取当前y轴范围
    ax.set_yticks(np.arange(np.floor(y_min / y_increment) * y_increment, np.ceil(y_max / y_increment) * y_increment, y_increment))  # 从0.25的倍数开始，间隔0.25
    ax.tick_params(axis='both', direction='out', which='major', labelsize=18)  # 加大刻度数字

    ax.grid(
        True,
        which='major',
        linestyle=':',
        linewidth=0.8,
        color='gray',
        alpha=0.5
    )
    # 3. 添加加粗的y=0参考线（红色实线，线宽2）
    ax.axhline(
        y=0,
        color='red',
        linestyle='--',  # 实线
        linewidth=2,  # 加粗线宽
        alpha=0.8,  # 降低透明度
        zorder=3  # 确保在最上层
    )

    # 4. 强制显示所有四条坐标轴边框
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)  # 显示边框
        ax.spines[spine].set_linewidth(0.5)  # 边框粗细
        ax.spines[spine].set_color('black')  # 边框颜色

    # 获取色度条并设置字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.set_box_aspect(15)
    cbar.ax.tick_params(labelsize=18)  # 设置色度条的字体大小为12
    cbar.set_label(cbar_feature, fontsize=24,fontweight='bold')  # 设置色度条的标题
    plt.tight_layout()
    plt.show()