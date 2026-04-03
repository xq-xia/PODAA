#!/usr/bin/python3

# Made by xiaxingquan
# March 2025

# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import numpy as np
from featurewiz_polars import polars_train_test_split
import polars as pl
from featurewiz_polars import FeatureWiz
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
import shap
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import matplotlib.cm as cm
from matplotlib.collections import PathCollection
from draw import draw_shap_interact, draw_shap_total
from sklearn.utils import resample
from scipy.stats import chi2_contingency
import os

###   Constructing a Prediction Model for Anesthesia Recovery Delay   ###


# # 设置matplotlib支持中文显示
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# # 设置全局字体（Times New Roman）
# rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24  # 基础字号

# # 设置中文字体为宋体，英文和数字使用Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 为了支持中文显示，需要指定中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


class PODAA:
    def __init__(self, X_train, X_test, y_train, y_test, features):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.features = features

    # 特征选择 by Boruta
    def feature_select_by_Boruta(self):
        model = RandomForestClassifier(bootstrap=True, max_features='sqrt', min_samples_split=2,
                                       min_samples_leaf=1, class_weight='balanced',
                                       n_estimators=200, random_state=42, max_depth=30)
        feat_selector = BorutaPy(
            model,
            n_estimators='auto',
            verbose=2,
            random_state=42)
        feat_selector.fit(np.array(self.X_train), np.array(self.y_train))
        # 输出各个特征的重要性排名
        print('\n Feature ranking:')
        print(pd.DataFrame({"feature": self.features,
                            "feature_ranking": feat_selector.ranking_}))

        # 特征是否被选择
        print('\n Selected features:')
        print(pd.DataFrame({"feature": self.features,
              "Selected": feat_selector.support_}))

        # 被选中的特征（Selected=True）
        print('\nSelected features (True):')
        selected_df = pd.DataFrame({
            "feature": self.features,
            "Selected": feat_selector.support_
        })
        print(selected_df[selected_df['Selected']])  # 筛选Selected=True的行

        # 获取特征重要性
        feature_importance = feat_selector.ranking_

        # 创建特征重要性数据框
        feature_importance_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': feature_importance
        })

        # 按重要性排序
        feature_importance_df = feature_importance_df.sort_values(
            by='Importance')

        # 设置中文字体（如SimHei）或其他支持你字符的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
        # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # 绘制特征重要性排序图
        plt.figure(figsize=(10, 8))
        plt.barh(
            feature_importance_df['Feature'],
            feature_importance_df['Importance'])
        # 调整y轴标签样式
        plt.yticks(
            # rotation=30,  # 旋转30度
            fontsize=4,  # 缩小字体大小
            ha='right',  # 水平对齐方式（右对齐）
            va='center'  # 垂直对齐方式（居中）
        )
        plt.xlabel('Importance')
        plt.title('Boruta Feature Importance')
        plt.gca().invert_yaxis()  # 使最重要的特征在顶部
        # plt.savefig('Figure/Boruta_Feature_Importance.png', dpi=1080, bbox_inches='tight')
        plt.show()

    # 特征选择 by polars
    def feature_select_by_featurewiz_polars(self, file_path):
        df = pl.read_csv(file_path, null_values=['NULL', 'NA'], try_parse_dates=True,
                         infer_schema_length=10000, ignore_errors=True)
        # Before we do feature selection we always need to make sure we split t
        target = '苏醒延迟60'
        predictors = [x for x in df.columns if x not in ['苏醒延迟60', '转出延迟']]

        X = df[predictors]
        y = df[target]

        X_train, X_test, y_train, y_test = polars_train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Initialize FeatureWiz for classification
        wiz = FeatureWiz(model_type="Classification", estimator=None,
                         corr_limit=0.7, category_encoders='onehot', classic=True, verbose=0)

        return wiz.selected_features

    # # shapley 特征重要性排名
    # def shap_Importance_Features(self):
    #
    #     # # XGBoost\GBM shap
    #     # 训练一个XGBoost模型
    #     # xgb_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=100)
    #     # xgb_model.fit(self.X_train, self.y_train)
    #
    #     gbm_model = GradientBoostingClassifier(
    #         n_estimators=100,
    #         learning_rate=0.1,
    #         random_state=42
    #     )
    #     gbm_model.fit(self.X_train, self.y_train)
    #
    #     # 对模型文件model进行解释
    #     explainer = shap.TreeExplainer(gbm_model)
    #     # 传入特征矩阵X，计算shap值
    #     shap_values = explainer.shap_values(self.X_train)
    #     print("base value:", explainer.expected_value)
    #
    #     #### 这里我是要获取某个特征的所有shap值
    #     # 获取特征名称
    #     feature_names = self.X_train.columns
    #     # 假设你想获取的特征名称是 'feature_name'
    #     #nine_feature = ['舒芬太尼ugkg','麻醉时长','术中艾司氯胺酮1是','腔镜手术','麻醉上级工作时间几年多','术中输液量ml','上级开始工作年限','顺阿1维库0']
    #     nine_feature = ['氟马纳洛酮']
    #
    #     for elem in nine_feature:
    #         feature_name = elem
    #         feature_index = feature_names.get_loc(feature_name)  # 获取特征的索引
    #
    #         # 提取该特征的所有SHAP值
    #         feature_shap_values = shap_values[:, feature_index]
    #
    #         # 将每个样本的特征值和SHAP值写入txt文件
    #         output_file = r"C:\Users\xiaxq\Desktop\麻醉苏醒延迟预测模型\data\前9个shap数据\shap_values_for_" + elem + ".txt"
    #         with open(output_file, 'w') as f:
    #             f.write(f"SHAP values for feature '{feature_name}':\n")
    #             for i, (value, shap_value) in enumerate(zip(self.X_train[feature_name], feature_shap_values)):
    #                 f.write(f"{value} : {shap_value}\n")
    #         print("ok")
    #
    #     shap.summary_plot(shap_values, self.X_train, cmap="jet", max_display=46, plot_size=(9, 12), show=False)
    #     shap.dependence_plot('舒芬太尼ugkg', shap_values, self.X_train, interaction_index='舒芬太尼ugkg', show=False)
    #     ax = plt.gca()  # 获取当前坐标轴
    #     # 设置y轴刻度标签大小（特征名称）
    #     ax.tick_params(axis='y', labelsize=8)  # 调整14为所需字号
    #     plt.tight_layout()  # 优化布局
    #     plt.savefig('Figure/Shapley_GBM_Importance.png', dpi=540, bbox_inches='tight')
    #     plt.show()  # 显示图形
    #
    #     # RF shap
    #     '''
    #     rf_model = RandomForestClassifier()
    #     rf_model.fit(self.X_train, self.y_train)
    #
    #     # 对模型文件model进行解释
    #     explainer = shap.TreeExplainer(rf_model)
    #     # 传入特征矩阵X，计算shap值
    #     shap_values = explainer.shap_values(self.X_train)
    #     shap_values = shap_values[:, :, 1]  # 取每个子数组的第一个元素
    #     print("base value:", explainer.expected_value)
    #     shap.summary_plot(shap_values, self.X_train,cmap="jet", max_display=46,plot_size=(9, 12),show=False)
    #     ax = plt.gca()  # 获取当前坐标轴
    #     # 设置y轴刻度标签大小（特征名称）
    #     ax.tick_params(axis='y', labelsize=8)  # 调整14为所需字号
    #     plt.tight_layout()  # 优化布局
    #     plt.savefig('Figure/Shapley_RF_Importance.png', dpi=540, bbox_inches='tight')
    #     plt.show()  # 显示图形
    #     '''

    # def plot_interaction_heatmap(self, shap_interaction_values):
    #     # 1. 计算每个特征的全局绝对SHAP值总和（用于排序）
    #     total_abs_shap = np.sum(np.abs(shap_interaction_values), axis=(0, 1))  # 对所有样本求和
    #     top_20_indices = np.argsort(total_abs_shap)[-20:][::-1]  # 取前20个最重要的特征（降序）
    #
    #     # 2. 筛选前20个特征的名称和交互值总和（非平均）
    #     top_20_features = self.X_train.columns[top_20_indices]
    #     top_20_interaction_total = np.sum(shap_interaction_values, axis=0)[top_20_indices, :][:,
    #                                top_20_indices]  # 用sum代替mean
    #
    #     df = pd.DataFrame(top_20_interaction_total)
    #
    #     # 导出DataFrame为CSV文件
    #     df.to_csv(r'C:\Users\xiaxq\Desktop\tmp\shap_5.csv', index=False, header=False)
    #     # 设置绘图风格
    #     sns.set(style="white")
    #
    #     # 3. 绘制热图
    #     plt.figure(figsize=(12, 10))
    #
    #     # 设置中文字体（如SimHei）或其他支持你字符的字体
    #     plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
    #     # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
    #     plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    #
    #     sns.heatmap(top_20_interaction_total, cmap="seismic", xticklabels=top_20_features,fmt=".2f",
    #         yticklabels=top_20_features,cbar=True, annot=True, linewidths=.5,
    #                      cbar_kws={"shrink": 0.8, "label": "Total Interaction Value"},annot_kws={"size": 10})
    #     # sns.heatmap(
    #     #     top_20_interaction_total,
    #     #     annot=True,
    #     #     fmt=".2f",
    #     #     cmap="coolwarm",
    #     #     xticklabels=top_20_features,
    #     #     yticklabels=top_20_features,
    #     #     linewidths=0.5,
    #     #     cbar_kws={"shrink": 0.8, "label": "Total Interaction Value"}  # 修改颜色条标签
    #     # )
    #     plt.title('Total SHAP Interaction Values (Top 20 Features)', fontsize=16)
    #     plt.xlabel('Feature', fontsize=14)
    #     plt.ylabel('Feature', fontsize=14)
    #     plt.xticks(rotation=45, ha='right')
    #     plt.yticks(rotation=0)
    #     plt.tight_layout()
    #     plt.show()

    def plot_interaction_heatmap(self, shap_interaction_values):
        # 1. 计算特征重要性并排序
        total_abs_shap = np.sum(np.abs(shap_interaction_values), axis=(0, 1))
        top_20_indices = np.argsort(total_abs_shap)[-20:][::-1]
        top_20_features = self.X_train.columns[top_20_indices]
        interaction_matrix = np.sum(shap_interaction_values, axis=0)[
            top_20_indices, :][:, top_20_indices]

        # 2. 创建下三角掩码（保留对角线）
        mask = np.triu(
            np.ones_like(
                interaction_matrix,
                dtype=bool),
            k=1)  # 上三角（不包括对角线）

        plt.rcParams.update({
            'font.sans-serif': 'SimHei',
            'axes.unicode_minus': False,
            # 'font.family': 'Times New Roman',
            'font.size': 10,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })

        # 4. 创建画布和坐标轴
        fig, ax = plt.subplots(figsize=(12, 10))

        # 5. 绘制热图
        heatmap = sns.heatmap(
            interaction_matrix,
            mask=mask,
            cmap="rainbow",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor='lightgray',
            cbar=True,
            cbar_kws={
                'shrink': 0.8,
                'label': 'Interaction Strength',
                'ticks': np.linspace(-np.max(np.abs(interaction_matrix)),
                                     np.max(np.abs(interaction_matrix)), 5)
            },
            square=True,
            ax=ax
        )

        # 6. 美化坐标轴
        ax.set_xticks(np.arange(len(top_20_features)) + 0.5)
        ax.set_yticks(np.arange(len(top_20_features)) + 0.5)
        ax.set_xticklabels(top_20_features, rotation=45, ha='right')
        ax.set_yticklabels(top_20_features, rotation=0)

        # 显示所有四个边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')

        # 7. 添加标题和调整布局
        ax.set_title('SHAP Interaction Values (Lower Triangle)', pad=20)
        ax.set_xlabel('Feature', labelpad=15)
        ax.set_ylabel('Feature', labelpad=15)

        plt.tight_layout()
        plt.show()

    def plot_combined_dependence(
            self, feature_name, X, shap_values, total_interaction):
        """
            分开在两个子图中显示主SHAP效应和总SHAP交互效应
            """
        feature_idx = X.columns.get_loc(feature_name)
        feature_values = X.iloc[:, feature_idx]

        # 创建1行2列的子图布局
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 第一个子图：主SHAP效应
        ax1.scatter(
            feature_values,
            shap_values[:, feature_idx],
            c='blue',
            alpha=0.5,
            label='Main SHAP'
        )
        ax1.set_xlabel(feature_name)
        ax1.set_ylabel('SHAP Value')
        ax1.set_title(f'Main SHAP Effect: {feature_name}')
        ax1.legend()
        ax1.grid(True)

        # 第二个子图：总SHAP交互效应
        ax2.scatter(
            feature_values,
            total_interaction[:, feature_idx],
            c='red',
            alpha=0.5,
            label='Total Interaction SHAP'
        )
        ax2.set_xlabel(feature_name)
        ax2.set_ylabel('SHAP Interaction Value')
        ax2.set_title(f'Total SHAP Interaction: {feature_name}')
        ax2.legend()
        ax2.grid(True)

        # 调整子图间距
        plt.tight_layout()
        plt.show()
    # # shapley 特征重要性排名
    # def shap_Importance_Features(self):
    #
    #     # # XGBoost\GBM shap
    #     # 训练一个XGBoost模型
    #     # xgb_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=100)
    #     # xgb_model.fit(self.X_train, self.y_train)
    #
    #     gbm_model = GradientBoostingClassifier(
    #         n_estimators=100,
    #         learning_rate=0.1,
    #         random_state=42
    #     )
    #     gbm_model.fit(self.X_train, self.y_train)
    #
    #     # 对模型文件model进行解释
    #     explainer = shap.TreeExplainer(gbm_model)
    #     # 传入特征矩阵X，计算shap值
    #     shap_values = explainer.shap_values(self.X_train)
    #     print("base value:", explainer.expected_value)
    #
    #     #### 这里我是要获取某个特征的所有shap值
    #     # 获取特征名称
    #     feature_names = self.X_train.columns
    #     # 假设你想获取的特征名称是 'feature_name'
    #     # nine_feature = ['舒芬太尼ugkg','麻醉时长','术中艾司氯胺酮1是','腔镜手术','麻醉上级工作时间几年多','术中输液量ml','上级开始工作年限','顺阿1维库0']
    #     nine_feature = ['氟马纳洛酮']
    #
    #     for elem in nine_feature:
    #         feature_name = elem
    #         feature_index = feature_names.get_loc(feature_name)  # 获取特征的索引
    #
    #         # 提取该特征的所有SHAP值
    #         feature_shap_values = shap_values[:, feature_index]
    #
    #         # 将每个样本的特征值和SHAP值写入txt文件
    #         output_file = r"C:\Users\xiaxq\Desktop\麻醉苏醒延迟预测模型\data\前9个shap数据\shap_values_for_" + elem + ".txt"
    #         with open(output_file, 'w') as f:
    #             f.write(f"SHAP values for feature '{feature_name}':\n")
    #             for i, (value, shap_value) in enumerate(zip(self.X_train[feature_name], feature_shap_values)):
    #                 f.write(f"{value} : {shap_value}\n")
    #         print("ok")
    #
    #     shap.summary_plot(shap_values, self.X_train, cmap="jet", max_display=46, plot_size=(9, 12), show=False)
    #     shap.dependence_plot('舒芬太尼ugkg', shap_values, self.X_train, interaction_index='舒芬太尼ugkg',
    #                          show=False)
    #     ax = plt.gca()  # 获取当前坐标轴
    #     # 设置y轴刻度标签大小（特征名称）
    #     ax.tick_params(axis='y', labelsize=8)  # 调整14为所需字号
    #     plt.tight_layout()  # 优化布局
    #     plt.savefig('Figure/Shapley_GBM_Importance.png', dpi=540, bbox_inches='tight')
    #     plt.show()  # 显示图形
    #
    #     # RF shap
    #     '''
    #     rf_model = RandomForestClassifier()
    #     rf_model.fit(self.X_train, self.y_train)
    #
    #     # 对模型文件model进行解释
    #     explainer = shap.TreeExplainer(rf_model)
    #     # 传入特征矩阵X，计算shap值
    #     shap_values = explainer.shap_values(self.X_train)
    #     shap_values = shap_values[:, :, 1]  # 取每个子数组的第一个元素
    #     print("base value:", explainer.expected_value)
    #     shap.summary_plot(shap_values, self.X_train,cmap="jet", max_display=46,plot_size=(9, 12),show=False)
    #     ax = plt.gca()  # 获取当前坐标轴
    #     # 设置y轴刻度标签大小（特征名称）
    #     ax.tick_params(axis='y', labelsize=8)  # 调整14为所需字号
    #     plt.tight_layout()  # 优化布局
    #     plt.savefig('Figure/Shapley_RF_Importance.png', dpi=540, bbox_inches='tight')
    #     plt.show()  # 显示图形
    #     '''

    def shap_Importance_Features(self):
        # 训练GBM模型
        gbm_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        gbm_model.fit(self.X_train, self.y_train)

        explainer = shap.Explainer(gbm_model)  # 使用训练好的最佳模型创建一个SHAP解释器
        print("正在计算主效应SHAP值 (基于X_test)...")  # 打印开始计算主效应SHAP值的提示
        # 将测试集数据传入解释器，计算每个样本每个特征的SHAP值
        shap_values_obj = explainer(self.X_train)
        shap_values = shap_values_obj.values  # 从SHAP解释对象中提取SHAP值矩阵
        # shap.summary_plot(shap_values, self.X_train, cmap="jet", max_display=46, plot_size=(9, 12), show=False)
        df = pd.DataFrame(self.X_train)

        # 导出DataFrame为CSV文件
        df.to_csv(
            r'C:\Users\xiaxq\Desktop\tmp\train_add_suf_dosage.csv',
            index=False,
            header=False)

        # shap.dependence_plot('麻醉时长', shap_values, self.X_train, interaction_index='舒芬太尼ugkg')
        # shap.dependence_plot('麻醉时长', shap_values, self.X_train, interaction_index='麻醉输液mlkgh')
        # shap.dependence_plot('麻醉时长', shap_values, self.X_train, interaction_index='术中输液量ml')
        # shap.dependence_plot('舒芬太尼ugkg', shap_values, self.X_train, interaction_index='麻醉时长')
        # shap.dependence_plot('麻醉输液mlkgh', shap_values, self.X_train, interaction_index='麻醉时长')
        # shap.dependence_plot('术中输液量ml', shap_values, self.X_train, interaction_index='麻醉时长')

        # shap.dependence_plot('上级开始工作年限', shap_values, self.X_train, interaction_index='术中艾司氯胺酮1是')
        draw_shap_total(self.X_train, 'winter', shap_values, "Senior Anesthetist Start Year", "Intraoperative Esketamine", 5,
                        0.25)
        draw_shap_total(self.X_train, 'winter', shap_values, "Anesthesia Infusion", "Anesthesia Duration", 20,
                        0.5)
        draw_shap_total(self.X_train, 'winter', shap_values, "Anesthesia Duration", "Intraoperative Fluid Volume", 100,
                        0.2)

        # shap.dependence_plot('Anesthesia Duration', shap_values, self.X_train, interaction_index='Anesthesia Infusion _ml/kg/h')
        # # shap.dependence_plot('麻醉时长', shap_values, self.X_train, interaction_index='术中输液量ml')
        # # shap.dependence_plot('术中艾司氯胺酮1是', shap_values, self.X_train, interaction_index='上级开始工作年限')
        # # shap.dependence_plot('麻醉输液mlkgh', shap_values, self.X_train, interaction_index='麻醉时长')
        # shap.dependence_plot('Intraoperative Fluid Volume _ml', shap_values, self.X_train, interaction_index='Anesthesia Duration')
        print(shap_values.shape)
        print("主效应SHAP值计算完成。")  # 打印计算完成的提示
        print("\n正在计算SHAP交互效应值 (基于X_test)...")  # 打印开始计算交互效应SHAP值的提示
        shap_interaction_values = explainer.shap_interaction_values(
            self.X_train)
        self.plot_interaction_heatmap(shap_interaction_values)
        print(type(shap_interaction_values))
        ax = shap.plots.heatmap(shap_values_obj, show=False)

        # 获取当前图形并保存
        fig = ax.get_figure()  # 从Axes获取Figure对象
        fig.set_size_inches(12, 8)
        plt.tight_layout()
        plt.savefig(
            r'C:\Users\xiaxq\Desktop\tmp\SHAP_Heatmap.png',
            dpi=240,
            bbox_inches='tight')
        plt.show()  # 最后显示

        # shap.dependence_plot(("上级开始工作年限", "术中艾司氯胺酮1是"), shap_interaction_values, self.X_train, show=False)
        draw_shap_interact(
            self.X_train,
            'winter',
            shap_interaction_values,
            "Senior Anesthetist Start Year",
            "Intraoperative Esketamine",
            5,
            0.25)
        draw_shap_interact(
            self.X_train,
            'winter',
            shap_interaction_values,
            "Anesthesia Infusion",
            "Anesthesia Duration",
            20,
            0.25)
        draw_shap_interact(
            self.X_train,
            'winter',
            shap_interaction_values,
            "Anesthesia Duration",
            "Intraoperative Fluid Volume",
            100,
            0.2)

        # shap.dependence_plot(("麻醉时长", "麻醉输液mlkgh"), shap_interaction_values, self.X_train, show=False)
        # shap.dependence_plot(("术中输液量ml", "麻醉时长"), shap_interaction_values, self.X_train, show=False)
        # shap.dependence_plot(("术中艾司氯胺酮1是", "上级开始工作年限"), shap_interaction_values, self.X_train, show=False)

        # shap.dependence_plot(("麻醉输液mlkgh", "麻醉时长"), shap_interaction_values, self.X_train, show=False)
        # shap.dependence_plot(("麻醉时长", "术中输液量ml"), shap_interaction_values, self.X_train, show=False)
        # shap.dependence_plot(("舒芬太尼ugkg", "麻醉时长"), shap_interaction_values, self.X_train, show=False)
        # shap.dependence_plot(("麻醉时长", "麻醉输液mlkgh"), shap_interaction_values, self.X_train, show=False)
        # shap.dependence_plot(("术中输液量ml", "麻醉时长"), shap_interaction_values, self.X_train, show=False)
        # shap.dependence_plot(("麻醉时长", "舒芬太尼ugkg"), shap_interaction_values, self.X_train, show=False)
        # shap.dependence_plot(("麻醉输液mlkgh", "麻醉时长"), shap_interaction_values, self.X_train, show=False)
        # shap.dependence_plot(("麻醉时长", "术中输液量ml"), shap_interaction_values, self.X_train, show=False)
        shap_values_total = []
        for i in shap_interaction_values:
            shap_values_total.append([])
            for j in i:
                count = 0
                for k in j:
                    count += k
                shap_values_total[-1].append(count)
        shap_values_total = np.array(shap_values_total)

        # 现在可以调用 .shape 属性
        # print(shap_values_total.shape)
        df = pd.DataFrame(shap_values_total)

        # 导出DataFrame为CSV文件
        df.to_csv(
            r'C:\Users\xiaxq\Desktop\tmp\shap_3_add_suf_dosage.csv',
            index=False,
            header=False)

        shap_values_main = []
        for i in shap_interaction_values:
            shap_values_main.append([])
            for j in range(0, len(i)):
                for k in range(0, len(i[j])):
                    if j == k:
                        shap_values_main[-1].append(i[j][k])
        shap_values_main = np.array(shap_values_main)

        # 现在可以调用 .shape 属性
        # print(shap_values_total.shape)
        df = pd.DataFrame(shap_values_main)

        # 导出DataFrame为CSV文件
        df.to_csv(
            r'C:\Users\xiaxq\Desktop\tmp\shap_4_add_suf_dosage.csv',
            index=False,
            header=False)

    def scatter_plot(self):
        # 1. 读取Excel文件
        excel_file = r"C:\Users\xiaxq\Desktop\PODAA结果数据\PODAA结果数据\excel\前9个shap值_gbm.xlsx"

        # 2. 获取所有工作表名称
        xls = pd.ExcelFile(excel_file, engine='openpyxl')
        sheet_names = xls.sheet_names[:9]  # 只取前9个工作表

        y_increment_list = [0.5, 0.25, 0.5, 0.2, 0.2, 0.1, 0.2, 0.05, 0.1]

        # 创建输出目录
        output_dir = r"C:\Users\xiaxq\Desktop\tmp"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 循环绘制每个type（0和1）
        for type_idx in range(0, 2):
            # 创建3x3的大图
            fig, axes = plt.subplots(3, 3, figsize=(24, 18))  # 增大画布尺寸
            fig.subplots_adjust(wspace=0.3, hspace=0.4)  # 调整子图间距

            # 定义编号字母
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

            flag = 0
            for i, (sheet, ax) in enumerate(zip(sheet_names, axes.flat)):
                # 读取数据
                df = pd.read_excel(excel_file, sheet_name=sheet)

                # 在当前子图中绘制散点图
                scatter = ax.scatter(
                    df.iloc[:, 0],  # 第一列作为x轴
                    df.iloc[:, type_idx + 1],  # 第二列作为y轴
                    s=80,  # 点大小
                    alpha=0.7,  # 透明度
                    edgecolors='#163EE7',  # 点边缘颜色
                    linewidths=0.5,  # 边缘线宽
                    c='#163EE7'  # 点颜色
                )

                # 调整点的大小和透明度
                for collection in ax.collections:
                    if isinstance(collection, PathCollection):
                        collection.set_sizes([40])
                        collection.set_alpha(0.6)

                # 设置坐标轴刻度
                x_increment = 1
                if df.columns[0] == 'SII':
                    x_increment = 1000
                x_min, x_max = ax.get_xlim()
                ax.set_xticks(np.arange(np.floor(x_min / x_increment) * x_increment + x_increment,np.ceil(x_max / x_increment) * x_increment, x_increment))

                y_increment = y_increment_list[flag]
                y_min, y_max = ax.get_ylim()
                ax.set_yticks(np.arange(np.floor(y_min / y_increment) * y_increment,np.ceil(y_max / y_increment) * y_increment, y_increment))

                # 设置标签
                ax.set_xlabel(df.columns[0],fontsize=36,fontweight='bold')  # 稍微减小字体
                ax.set_ylabel(df.columns[type_idx + 1],fontsize=36, fontweight='bold')
                ax.tick_params(axis='both',direction='out',which='major',labelsize=24)  # 调整刻度字体

                # 添加网格和参考线
                ax.grid(
                    True,
                    which='major',
                    linestyle=':',
                    linewidth=0.8,
                    color='gray',
                    alpha=0.5)
                ax.axhline(
                    y=0,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.8,
                    zorder=3)

                # 设置坐标轴边框
                for spine in ['top', 'bottom', 'left', 'right']:
                    ax.spines[spine].set_visible(True)
                    ax.spines[spine].set_linewidth(0.5)
                    ax.spines[spine].set_color('black')

                # 在左上角添加编号（A-I）
                ax.text(-0.2, 1.1, labels[i],
                        transform=ax.transAxes,
                        fontsize=36,
                        fontweight='bold',
                        fontfamily='Times New Roman',
                        color='black',
                        verticalalignment='top')

                flag += 1

            # 保存组合图
            plt.tight_layout()
            output_path = os.path.join(
                output_dir, f'combined_plot_type_{
                    type_idx + 1}.png')
            plt.savefig(output_path, dpi=800, bbox_inches='tight')
            # print(f"组合图已保存至: {output_path}")
            plt.show()

    # 超参数网格搜索调优
    def h_parameters_tuning(self, model, param_grid):
        # 初始化 GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1)

        # 执行搜索
        grid_search.fit(self.X_train, self.y_train)

        # 输出最佳参数和最佳分数
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation score:", grid_search.best_score_)

        # 使用最佳参数训练模型
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)

        # 计算评估指标
        print('f1: ', f1_score(self.y_test, y_pred))
        print('acc：', accuracy_score(self.y_test, y_pred))
        print('pcc：', precision_score(self.y_test, y_pred))
        print('rcc：', recall_score(self.y_test, y_pred), '\n')

        self.Visualization(best_model,'h_parameters_tuning_best_model',y_pred)

    # LR模型预测
    def LR(self):
        # 初始化 LogisticRegression
        lr = LogisticRegression(
            C=100,
            max_iter=100,
            solver='liblinear',
            penalty='l1',
            random_state=42)  # 增加最大迭代次数以避免收敛警告
        lr.fit(self.X_train, self.y_train)  # 训练模型s
        lr_pres = lr.predict(self.X_test)  # 预测

        # 计算评估指标
        print('f1: ', f1_score(self.y_test, lr_pres))
        print('acc：', accuracy_score(self.y_test, lr_pres))
        print('pcc：', precision_score(self.y_test, lr_pres))
        print('rcc：', recall_score(self.y_test, lr_pres), '\n')

        self.Visualization(lr, 'LR', lr_pres)

    # RF模型预测
    def RF_Model(self):
        clf = RandomForestClassifier(bootstrap=True, max_features='sqrt', min_samples_split=2,
                                     min_samples_leaf=1, class_weight='balanced',
                                     n_estimators=200, random_state=42, max_depth=30)
        clf.fit(self.X_train, self.y_train)  # 训练模型s
        clf_pres = clf.predict(self.X_test)  # 预测

        # 计算评估指标
        print('f1: ', f1_score(self.y_test, clf_pres))
        print('acc：', accuracy_score(self.y_test, clf_pres))
        print('pcc：', precision_score(self.y_test, clf_pres))
        print('rcc：', recall_score(self.y_test, clf_pres), '\n')

        self.Visualization(clf, 'RF', clf_pres)

    # GBM模型预测
    def GBM_Model(self):
        class_weight = {1: 5, 0: 1}
        clf = GradientBoostingClassifier(subsample=0.8, n_estimators=200,
                                         learning_rate=0.1, max_depth=10,
                                         random_state=42, min_samples_split=10,
                                         min_samples_leaf=5)
        clf.fit(self.X_train, self.y_train)  # 训练模型
        clf_pres = clf.predict(self.X_test)  # 预测

        # 计算评估指标
        print('f1: ', f1_score(self.y_test, clf_pres))
        print('acc：', accuracy_score(self.y_test, clf_pres))
        print('pcc：', precision_score(self.y_test, clf_pres))
        print('rcc：', recall_score(self.y_test, clf_pres), '\n')

        self.Visualization(clf, 'GBM', clf_pres)

    # XGBoost模型预测
    def XGBoost_Model(self):
        clf = XGBClassifier(class_weight='balanced',
                            booster='dart',  # 给定模型求解方式 可选参数gbtree、gblinear、dart
                            objective='multi:softmax',
                            num_class=2,
                            gamma=0.1,  # 指定节点分裂所需的最小损失函数下降值
                            max_depth=15,  # 最大树深
                            reg_lambda=2,
                            # 权重的L2正则化项。(和Ridge
                            # regression类似)。这个参数是用来控制XGBoost的正则化部分的。这个参数在减少过拟合上很有帮助。
                            subsample=1,  # 对于每棵树，随机采样的比例。减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。
                            colsample_bytree=0.5,  # 控制每棵随机采样的列数的占比(每一列是一个特征)
                            min_child_weight=1,  # 用于控制子节点中样本的最小权重和
                            eta=0.1,  # 每一步迭代的步长
                            seed=1000,
                            nthread=4,  # 使用的线程数
                            # 设置 XGBoost
                            # 使用softmax目标函数做多分类，需要设置参数num_class（类别个数）
                            eval_metric=['mlogloss', 'merror'],
                            n_estimators=20,  # 弱分类器的数量
                            learning_rate=0.1,  # 学习率
                            reg_alpha=1,  # L1正则化权重项，增加此值将使模型更加保守
                            random_state=42
                            )

        clf.fit(self.X_train, self.y_train)  # 训练模型
        clf_pres = clf.predict(self.X_test)  # 预测

        # 计算评估指标
        print('f1: ', f1_score(self.y_test, clf_pres))
        print('acc：', accuracy_score(self.y_test, clf_pres))
        print('pcc：', precision_score(self.y_test, clf_pres))
        print('rcc：', recall_score(self.y_test, clf_pres))

        self.Visualization(clf, 'XGBoost', clf_pres)

    # AdaBoost模型预测
    def AdaBoost_Model(self):
        dt = tree.DecisionTreeClassifier(max_features='sqrt', min_samples_split=5,
                                         min_samples_leaf=4, class_weight='balanced',
                                         random_state=42, max_depth=25)
        clf = AdaBoostClassifier(estimator=dt, learning_rate=0.01,
                                 n_estimators=100, random_state=42)
        clf.fit(self.X_train, self.y_train)  # 训练模型
        clf_pres = clf.predict(self.X_test)  # 预测

        # 计算评估指标
        print('f1: ', f1_score(self.y_test, clf_pres))
        print('acc：', accuracy_score(self.y_test, clf_pres))
        print('pcc：', precision_score(self.y_test, clf_pres))
        print('rcc：', recall_score(self.y_test, clf_pres))

        self.Visualization(clf, 'AdaBoost', clf_pres)

    # KNN模型预测
    def KNN_Model(self):
        clf = KNeighborsClassifier(
            metric='manhattan',
            n_neighbors=1,
            p=1,
            weights='uniform')
        clf.fit(self.X_train, self.y_train)  # 训练模型
        clf_pres = clf.predict(self.X_test)  # 预测

        # 计算评估指标
        print('f1: ', f1_score(self.y_test, clf_pres))
        print('acc：', accuracy_score(self.y_test, clf_pres))
        print('pcc：', precision_score(self.y_test, clf_pres))
        print('rcc：', recall_score(self.y_test, clf_pres))

        self.Visualization(clf, 'KNN', clf_pres)

    # DT模型预测
    def DT_Model(self):
        clf = tree.DecisionTreeClassifier(max_features='sqrt', min_samples_split=5,
                                          min_samples_leaf=4, class_weight='balanced',
                                          random_state=42, max_depth=25)
        clf.fit(self.X_train, self.y_train)  # 训练模型
        clf_pres = clf.predict(self.X_test)  # 预测

        # 计算评估指标
        print('f1: ', f1_score(self.y_test, clf_pres))
        print('acc：', accuracy_score(self.y_test, clf_pres))
        print('pcc：', precision_score(self.y_test, clf_pres))
        print('rcc：', recall_score(self.y_test, clf_pres))

        self.Visualization(clf, 'DT', clf_pres)

    # Naive Bayes模型预测
    def NB_Model(self):
        clf = BernoulliNB(alpha=0.1, binarize=0.5, fit_prior=False)
        clf.fit(self.X_train, self.y_train)  # 训练模型
        clf_pres = clf.predict(self.X_test)  # 预测

        # 计算评估指标
        print('f1: ', f1_score(self.y_test, clf_pres))
        print('acc：', accuracy_score(self.y_test, clf_pres))
        print('pcc：', precision_score(self.y_test, clf_pres))
        print('rcc：', recall_score(self.y_test, clf_pres))

        self.Visualization(clf, 'NB', clf_pres)

    # LDA模型预测
    def LDA_Model(self):
        clf = LinearDiscriminantAnalysis(n_components=1)
        clf.fit(self.X_train, self.y_train)  # 训练模型
        clf_pres = clf.predict(self.X_test)  # 预测

        # 计算评估指标
        print('f1: ', f1_score(self.y_test, clf_pres))
        print('acc：', accuracy_score(self.y_test, clf_pres))
        print('pcc：', precision_score(self.y_test, clf_pres))
        print('rcc：', recall_score(self.y_test, clf_pres))

        self.Visualization(clf, 'LDA', clf_pres)

    def Visualization(self, model, model_name, lr_pres):
        # 混淆矩阵热点图
        labels = [0, 1]  # 指定类别标签
        # 确保参数顺序正确: (y_true, y_pred)
        cm = confusion_matrix(self.y_test, lr_pres, labels=labels)
        cm_int = cm.astype(int)

        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(cm_int, annot=True, fmt='d', annot_kws={'size': 20, 'weight': 'bold', 'color': 'blue'},
                         cmap='Blues', cbar=False)  # 使用蓝色调

        # 设置坐标轴标签和标题
        plt.title(
            f'Confusion Matrix of {model_name} Model',
            fontsize=20,
            pad=15)
        plt.xlabel('Predicted Label', fontsize=16, labelpad=10)
        plt.ylabel('True Label', fontsize=16, labelpad=10)

        # 获取当前坐标轴对象
        ax = plt.gca()

        # 设置X轴和Y轴的刻度标签
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_yticklabels(labels, fontsize=14, rotation=0)  # 确保Y轴标签水平显示

        # 调整Y轴标签顺序（关键修改）
        ax.set_ylim(len(labels), 0)  # 修正热力图的Y轴范围，解决Seaborn默认反转问题

        plt.tight_layout()
        plt.show()

        # ROC曲线和AUC（保持不变）
        lr_pres_proba = model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, lr_pres_proba)
        auc = roc_auc_score(self.y_test, lr_pres_proba)

        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", linewidth=2.5)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
        plt.legend(loc=4, fontsize=14)
        plt.title(f'ROC Curve of {model_name} Model', fontsize=20, pad=15)
        plt.xlabel('False Positive Rate (FPR)', fontsize=16, labelpad=10)
        plt.ylabel('True Positive Rate (TPR)', fontsize=16, labelpad=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # 绘制所有模型ROC曲线 （不带置信区间）
    def plot_all_roc_curves(self):
        # 初始化模型列表
        dt = tree.DecisionTreeClassifier(max_features='sqrt', min_samples_split=5,
                                         min_samples_leaf=4, class_weight='balanced',
                                         random_state=42, max_depth=25)
        models = [
            LogisticRegression(
                C=100,
                max_iter=100,
                solver='liblinear',
                penalty='l1',
                random_state=42),
            RandomForestClassifier(bootstrap=True, max_features='sqrt', min_samples_split=2,
                                   min_samples_leaf=1, class_weight='balanced',
                                   n_estimators=200, random_state=42, max_depth=30),
            GradientBoostingClassifier(subsample=0.8, n_estimators=200,
                                       learning_rate=0.1, max_depth=10,
                                       random_state=42, min_samples_split=10,
                                       min_samples_leaf=5),
            XGBClassifier(class_weight='balanced',
                          booster='dart',  # 给定模型求解方式 可选参数gbtree、gblinear、dart
                          objective='multi:softmax',
                          num_class=2,
                          gamma=0.1,  # 指定节点分裂所需的最小损失函数下降值
                          max_depth=15,  # 最大树深
                          reg_lambda=2,
                          # 权重的L2正则化项。(和Ridge
                          # regression类似)。这个参数是用来控制XGBoost的正则化部分的。这个参数在减少过拟合上很有帮助。
                          subsample=1,  # 对于每棵树，随机采样的比例。减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。
                          colsample_bytree=0.5,  # 控制每棵随机采样的列数的占比(每一列是一个特征)
                          min_child_weight=1,  # 用于控制子节点中样本的最小权重和
                          eta=0.1,  # 每一步迭代的步长
                          seed=1000,
                          nthread=4,  # 使用的线程数
                          # 设置 XGBoost 使用softmax目标函数做多分类，需要设置参数num_class（类别个数）
                          eval_metric=['mlogloss', 'merror'],
                          n_estimators=20,  # 弱分类器的数量
                          learning_rate=0.1,  # 学习率
                          reg_alpha=1,  # L1正则化权重项，增加此值将使模型更加保守
                          random_state=42
                          ),
            AdaBoostClassifier(estimator=dt, learning_rate=0.01,
                               n_estimators=100, random_state=42),
            KNeighborsClassifier(
                metric='manhattan',
                n_neighbors=1,
                p=1,
                weights='uniform'),
            tree.DecisionTreeClassifier(max_features='sqrt', min_samples_split=5,
                                        min_samples_leaf=4, class_weight='balanced',
                                        random_state=42, max_depth=25),
            BernoulliNB(alpha=0.1, binarize=0.5, fit_prior=False),
            LinearDiscriminantAnalysis(n_components=1)
        ]

        # 颜色列表，每个模型对应一个颜色
        colors = [
            'blue',
            'red',
            'green',
            'purple',
            'orange',
            'brown',
            'pink',
            'gray',
            'olive']
        model_name = [
            'LR',
            'RF',
            'GBM',
            'XGB',
            'ADA',
            'KNN',
            'DT',
            'NB',
            'LDA']

        # 计算每个模型的ROC曲线和AUC值
        fpr, tpr, roc_auc = [], [], []
        for i, model in enumerate(models):
            model.fit(self.X_train, self.y_train)
            y_score = model.predict_proba(self.X_test)[:, 1]
            fpr_, tpr_, _ = roc_curve(self.y_test, y_score)
            roc_auc_ = auc(fpr_, tpr_)
            fpr.append(fpr_)
            tpr.append(tpr_)
            roc_auc.append(roc_auc_)
            print(f'{model_name[i]} AUC: {roc_auc_:.3f}')

        # 绘制所有模型的ROC曲线
        plt.figure(figsize=(10, 8))
        for i, (fpr_, tpr_, roc_auc_) in enumerate(zip(fpr, tpr, roc_auc)):
            plt.plot(fpr_, tpr_, color=colors[i % len(colors)], label=f'{model_name[i]} (AUC = {roc_auc_:.3f})',
                     linewidth=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        # plt.savefig(r'C:\Users\xiaxq\Desktop\tmp\AUC.png', dpi=1024)
        plt.show()

    # 绘制所有模型ROC曲线(带置信区间)
    def plot_all_roc_curves_with_ci(self, n_bootstrap=1000):
        # 初始化模型列表（保持原样）
        dt = tree.DecisionTreeClassifier(max_features='sqrt', min_samples_split=5,
                                         min_samples_leaf=4, class_weight='balanced',
                                         random_state=42, max_depth=25)
        models = [
            LogisticRegression(
                C=100,
                max_iter=100,
                solver='liblinear',
                penalty='l1',
                random_state=42),
            RandomForestClassifier(bootstrap=True, max_features='sqrt', min_samples_split=2,
                                   min_samples_leaf=1, class_weight='balanced',
                                   n_estimators=200, random_state=42, max_depth=30),
            GradientBoostingClassifier(subsample=0.8, n_estimators=200,
                                       learning_rate=0.1, max_depth=10,
                                       random_state=42, min_samples_split=10,
                                       min_samples_leaf=5),
            XGBClassifier(class_weight='balanced',
                          booster='dart',
                          objective='multi:softmax',
                          num_class=2,
                          gamma=0.1,
                          max_depth=15,
                          reg_lambda=2,
                          subsample=1,
                          colsample_bytree=0.5,
                          min_child_weight=1,
                          eta=0.1,
                          seed=1000,
                          nthread=4,
                          eval_metric=['mlogloss', 'merror'],
                          n_estimators=20,
                          learning_rate=0.1,
                          reg_alpha=1,
                          random_state=42),
            AdaBoostClassifier(estimator=dt, learning_rate=0.01,
                               n_estimators=100, random_state=42),
            KNeighborsClassifier(
                metric='manhattan',
                n_neighbors=1,
                p=1,
                weights='uniform'),
            tree.DecisionTreeClassifier(max_features='sqrt', min_samples_split=5,
                                        min_samples_leaf=4, class_weight='balanced',
                                        random_state=42, max_depth=25),
            BernoulliNB(alpha=0.1, binarize=0.5, fit_prior=False),
            LinearDiscriminantAnalysis(n_components=1)
        ]

        # 颜色和模型名称
        colors = [
            'blue',
            'red',
            'green',
            'purple',
            'orange',
            'brown',
            'pink',
            'gray',
            'olive']
        model_name = [
            'LR',
            'RF',
            'GBM',
            'XGB',
            'ADA',
            'KNN',
            'DT',
            'NB',
            'LDA']

        # 存储结果
        results = {
            'fpr': [],
            'tpr': [],
            'roc_auc': [],
            'roc_auc_ci_lower': [],
            'roc_auc_ci_upper': []
        }

        # 对每个模型进行Bootstrap计算
        for i, model in enumerate(models):
            print(f"Processing model: {model_name[i]}")

            # 存储每次Bootstrap的AUC值
            bootstrap_aucs = []

            # 原始数据计算
            model.fit(self.X_train, self.y_train)
            y_score = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_score)
            roc_auc = auc(fpr, tpr)

            # Bootstrap循环
            for _ in range(n_bootstrap):
                # 有放回抽样
                indices = resample(np.arange(len(self.X_test)), replace=True)
                X_bootstrap = self.X_test.iloc[indices]
                y_bootstrap = self.y_test.iloc[indices]

                # 计算AUC
                try:
                    y_score_bootstrap = model.predict_proba(X_bootstrap)[:, 1]
                    fpr_b, tpr_b, _ = roc_curve(y_bootstrap, y_score_bootstrap)
                    bootstrap_aucs.append(auc(fpr_b, tpr_b))
                except BaseException:
                    continue  # 跳过可能出现的错误

            # 计算95% CI
            bootstrap_aucs = np.array(bootstrap_aucs)
            ci_lower = np.percentile(bootstrap_aucs, 2.5)
            ci_upper = np.percentile(bootstrap_aucs, 97.5)

            # 存储结果
            results['fpr'].append(fpr)
            results['tpr'].append(tpr)
            results['roc_auc'].append(roc_auc)
            results['roc_auc_ci_lower'].append(ci_lower)
            results['roc_auc_ci_upper'].append(ci_upper)

            print(f"{model_name[i]} AUC: {
                  roc_auc:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")

        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        for i in range(len(models)):
            # 绘制主ROC曲线
            plt.plot(results['fpr'][i], results['tpr'][i],
                     color=colors[i % len(colors)],
                     label=f'{model_name[i]} (AUC = {results["roc_auc"][i]:.3f} [{
                results["roc_auc_ci_lower"][i]:.3f}-{results["roc_auc_ci_upper"][i]:.3f}])',
                linewidth=2)
            # plt.legend(loc="lower right", fontsize=10)

            # 可选：绘制置信区间阴影
            plt.fill_between(results['fpr'][i],
                             results['tpr'][i] - 0.02,  # 简单示例，实际应计算真实CI
                             results['tpr'][i] + 0.02,
                             color=colors[i % len(colors)],
                             alpha=0.1)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic with 95% Confidence Intervals')
        plt.legend(loc="lower right")
        plt.show()

    # 统计分析 连续变量 Kruskal Wallis验证

    def Kruskal_Wallis(self, feature_name):
        df = pd.read_csv(
            r"C:\Users\xiaxq\Desktop\麻醉苏醒延迟预测模型\fin_data\怀化数据_完整_填充缺失值.csv")

        # 2. 提取两组数据
        delay_group = df[df['Delayed Awakening more than 60 mins']
                         == '1'][feature_name]  # 苏醒延迟组
        no_delay_group = df[df['Delayed Awakening more than 60 mins']
                            == '0'][feature_name]  # 苏醒不延迟组

        # 3. 执行Kruskal-Wallis检验
        print(delay_group)
        h_statistic, p_value = stats.kruskal(delay_group, no_delay_group)

        # 4. 输出结果
        print(f"Kruskal-Wallis检验结果：")
        print(f"H统计量 = {h_statistic:.4f}")
        print(f"P值 = {p_value:.4f}")

        # 5. 判断显著性
        alpha = 0.05
        if p_value < alpha:
            print(f"P值 < {alpha}，两组在{feature_name}特征上存在显著差异")
        else:
            print(f"P值 ≥ {alpha}，两组在{feature_name}特征上无显著差异")

        # 6. 可选：计算中位数差异（医学研究常用）
        median_diff = delay_group.median() - no_delay_group.median()
        print(f"\n中位数差异（延迟组-非延迟组）: {median_diff:.4f}")

    # 统计分析 分类变量 χ2验证
    def Kruskal_Wallis(self, feature_name):
        df = pd.read_csv(
            r"C:\Users\xiaxq\Desktop\麻醉苏醒延迟预测模型\fin_data\怀化数据_完整_填充缺失值.csv")

        # 4. 构建列联表（Contingency Table）
        contingency_table = pd.crosstab(
            index=df['Delayed Awakening more than 60 mins'],  # 行：分组变量
            columns=df[feature_name],  # 列：目标二分类特征
            margins=False  # 不显示总计行/列
        )

        # 5. 检查列联表（确保无空单元格）
        print("\n列联表：")
        print(contingency_table)

        # 6. 执行卡方检验
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # 7. 输出结果
        print("\n卡方检验结果：")
        print(f"卡方统计量 = {chi2:.4f}")
        print(f"P值 = {p_value:.4f}")
        print(f"自由度 = {dof}")

        # 8. 判断显著性
        alpha = 0.05
        if p_value < alpha:
            print(
                f"\n结论：P值 < {alpha}, 两组在 {feature_name}特征上分布差异显著（p={
                    p_value:.4f}）")
        else:
            print(
                f"\n结论：P值 ≥ {alpha}，两组在{feature_name}特征上无显著差异（p={
                    p_value:.4f}）")

        # 9. 计算效应量（Phi系数或Cramer's V）
        n = contingency_table.sum().sum()
        phi = np.sqrt(chi2 / n)  # 适用于2x2表
        print(f"\n效应量（Phi系数） = {phi:.4f}")
