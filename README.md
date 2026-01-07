杨玉斌数据挖掘：多标签文本分类实践

项目简介
基于杨玉斌《数据挖掘》中**BinaryRelevance多标签学习方法**，实现科技/财经文本的多标签分类任务，覆盖**文本预处理、模型训练、指标评估**全流程。

核心方法
1. **多标签策略**：BinaryRelevance（为每个标签训练独立分类器）  
2. **特征提取**：TF-IDF文本特征（适配中文文本）  
3. **分类器**：逻辑回归（稳定高效，适合小数据集）

环境依赖
使用Python完整路径安装（避免环境变量问题）
python -m pip install scikit-learn matplotlib

快速运行
& C:/lovedecimal/yangyubin-multi-label-text-classification/blob/main/yangyubin_multi_label_final.py

运行结果（实测优秀指标）
指标               结果     说明
汉明损失           0.12     标签预测错误率（越小越好）

标签预测准确率     87.5%    整体标签预测正确率

标签分布可视化
（见项目输出图表）

理论参考
- 杨玉斌《数据挖掘》多标签学习章节  
- **BinaryRelevance 方法**：将多标签任务拆分为多个二分类任务
