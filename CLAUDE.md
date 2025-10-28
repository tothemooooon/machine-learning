# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个数据挖掘课程的学习笔记和实战作业仓库，包含机器学习算法的理论实现和实际应用案例。项目会随着课程进度持续更新，涵盖从基础算法到高级应用的完整学习路径。

## 项目结构

```
machine_leaning_homework/
├── KNN/                    # KNN算法学习模块
│   ├── KNN.ipynb          # KNN算法原理与手搓实现
│   ├── final_demo.ipynb   # KNN在金融数据分类中的应用
│   ├── credict_prediction.ipynb  # 企业信用评级预测实战
│   ├── financial_analyse.ipynb   # S&P 500公司金融数据分析
│   ├── financials.csv     # S&P 500公司金融数据集
│   └── corporate_rating.csv # 企业信用评级数据集
├── 感知机/                 # 感知机算法学习模块
│   ├── 3-感知机.pdf       # 感知机理论讲义
│   └── 感知机与神经网络教程.md # 感知机和神经网络学习笔记
└── 神经网络数字识别/        # 神经网络数字识别实战项目
    ├── 手写神经网络实现.ipynb    # 从零实现的神经网络
    ├── 标准库对比实现.ipynb      # TensorFlow/PyTorch框架对比
    ├── 数据预处理.ipynb          # 数据加载和预处理流程
    ├── 模型评估可视化.ipynb      # 模型评估和可视化工具
    ├── README.md                 # 项目详细说明文档
    └── saved_models/             # 保存的模型文件
```

## 技术环境

- **Python**: 3.13.7
- **核心数据科学库**:
  - numpy (2.3.3) - 数值计算
  - pandas (2.3.2) - 数据处理和分析
  - scikit-learn - 机器学习算法库
  - matplotlib & seaborn - 数据可视化
  - jupyter - 交互式开发环境
  - imblearn - 不平衡数据处理

## 环境配置

### 创建虚拟环境
```bash
# 使用conda创建环境
conda create -n ml_homework python=3.13.7
conda activate ml_homework

# 或使用venv
python -m venv ml_homework
# Windows:
ml_homework\Scripts\activate
# Linux/Mac:
source ml_homework/bin/activate
```

### 安装依赖包
```bash
pip install numpy pandas scikit-learn matplotlib seaborn imblearn jupyter

# 或者使用requirements.txt (如果存在)
pip install -r requirements.txt
```

## 开发环境设置

### 启动开发环境
```bash
jupyter notebook
```

### 安装依赖包
```bash
pip install numpy pandas scikit-learn matplotlib seaborn imblearn jupyter
```

## 学习内容组织

### 1. 算法理论学习
- **手搓实现**: 从零开始编写算法代码，深入理解原理
- **参数调优**: 探索不同超参数对算法性能的影响
- **性能优化**: 学习KD树等数据结构优化算法效率

### 2. 标准库对比
- **sklearn实现**: 对比手搓代码与标准库的差异
- **性能评估**: 使用准确率、F1分数等指标评估模型
- **最佳实践**: 学习工业级机器学习代码规范

### 3. 实际应用案例
- **鸢尾花分类**: 经典的入门数据集
- **金融数据分析**: S&P 500公司行业分类
- **信用评级预测**: 企业财务风险评估
- **手写数字识别**: 神经网络图像识别实战

## 数据处理流程

### 标准数据预处理流程
1. **数据加载**: pandas读取CSV文件
2. **特征工程**: 选择相关特征，处理缺失值
3. **数据标准化**: StandardScaler标准化处理
4. **降维处理**: PCA保留主要信息
5. **数据分割**: 分层采样保证类别分布

### 常用技术
- 缺失值填充：均值/中位数填充
- 特征标准化：Z-score标准化
- 维度约简：PCA主成分分析
- 不平衡处理：SMOTE过采样技术
- 图像预处理：归一化、数据增强
- 深度学习：神经网络、反向传播、梯度下降

## 开发工作流

### Jupyter Notebook 最佳实践
- 交互式开发和实验
- 代码和文档结合
- 可视化结果展示
- 实验结果保存和分享

### 实验方法论
1. 理论学习 → 代码实现
2. 标准库对比 → 性能分析
3. 实际数据应用 → 结果评估
4. 参数优化 → 模型改进

## 课程作业特色

### 循序渐进的学习路径
- **基础算法**: KNN、感知机等经典算法
- **理论实践**: 数学原理与编程实现结合
- **真实数据**: 使用实际业务数据集
- **完整流程**: 从数据处理到模型部署
- **深度学习**: 神经网络、卷积神经网络等高级技术

### 技能培养目标
- 机器学习算法理解能力
- 数据分析和处理能力
- 代码实现和优化能力
- 实验设计和评估能力

## 常用命令和操作

### 启动开发环境
```bash
# 启动Jupyter Notebook
jupyter notebook

# 直接打开特定notebook
jupyter notebook KNN/KNN.ipynb

# 启动Jupyter Lab (更现代化的界面)
jupyter lab
```

### 数据处理命令
```bash
# 查看数据集基本信息
python -c "import pandas as pd; df=pd.read_csv('KNN/financials.csv'); print(df.shape, df.columns.tolist())"

# 快速统计缺失值
python -c "import pandas as pd; df=pd.read_csv('KNN/financials.csv'); print(df.isnull().sum())"
```

### 数据集信息
- **金融数据**: 505家S&P 500公司，14个财务指标
- **评级数据**: 2029家企业，25个财务比率，10个信用等级
- **MNIST数据**: 70,000个手写数字样本，28×28像素图像

## 持续更新计划

随着课程进度，仓库将陆续添加：
- 更多机器学习算法实现
- 不同领域的数据集应用
- 高级优化技术
- 项目实战案例
- 深度学习专题项目
- 模型部署和生产化

### 已完成项目
- ✅ KNN算法实现与应用
- ✅ 感知机理论教程
- ✅ 神经网络数字识别系统

### 进行中项目
- 🔄 卷积神经网络(CNN)实现
- 🔄 自然语言处理基础