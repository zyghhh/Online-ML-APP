# Online Machine Learning App

一个面向 **概念漂移数据流** 的在线机器学习实验平台。项目基于 Streamlit 构建交互式 Web 应用，基于 River 扩展在线学习能力，并实现了自研的 **WPWPE（Window Performance Weighted Probability Ensemble）窗口性能加权概率集成框架**，用于在非平稳数据流场景下动态融合多个在线基学习器。

项目来源于毕业论文研究，但不止是算法 demo。它把数据流建模、概念漂移适应、在线评估、资源消耗监控和可视化对比实验整合成一个可运行的平台，用于研究和比较不同在线学习算法在真实/合成流式数据上的表现。

## Why This Project Matters

传统批处理机器学习默认训练集和测试集来自相同分布，但很多真实业务数据不是静态的。例如网络入侵检测、用户行为建模、金融风控、设备监控和推荐系统中，数据分布会随时间变化，模型需要在持续到来的样本上边预测边学习。

这个项目关注的正是这类 **data stream + concept drift** 问题：

- 数据不是一次性训练完成，而是逐条到达。
- 模型在预测后才能看到真实标签并增量更新。
- 单个模型在不同漂移阶段表现不稳定，需要动态集成。
- 算法评估不能只看最终准确率，还要观察时间、内存和性能曲线。

## Highlights

### 算法层面

- **面向概念漂移的数据流学习**：支持 Elec2、Phishing、SMTP 等典型在线学习数据集，以及用户上传 CSV 数据，用于模拟真实业务中数据分布随时间变化的问题。
- **自研 WPWPE 集成策略**：在滑动窗口内统计各基学习器的近期性能，将实时错误率转换为动态权重，并对多个模型的 `predict_proba_one` 输出做概率级融合。
- **多漂移检测器组合**：集成 ADWIN、KSWIN、PageHinkley、DDM、EDDM、HDDM_A、HDDM_W 等漂移检测器，可组合 ARF、SRP、Hoeffding Tree、GaussianNB 等在线模型。
- **预顺序评估（Progressive Validation）**：每条样本先预测再学习，更符合数据流场景下模型无法提前看到未来样本的评估方式。
- **多维指标评估**：支持 Accuracy、Precision、Recall、F1、Cohen Kappa、ROC AUC，以及回归任务中的 MAE、MSE、RMSE、R2。

### 系统层面

- **多页 Streamlit 应用架构**：通过 `MultiPage` 封装页面注册与路由，将离线 ML、在线 ML、集成学习、对比实验拆分为独立功能页。
- **算法实验闭环**：提供数据上传、目标列选择、算法配置、超参数调整、模型运行、指标展示和曲线可视化。
- **资源消耗可观测**：在实验过程中记录并绘制模型 Accuracy、运行时间和内存占用，支持从效果与成本两个维度比较算法。
- **可扩展的模型工厂设计**：通过 `make_model`、`make_WPWPE`、`get_classifier` 等函数将 UI 参数映射到具体模型，便于继续接入新的在线学习算法和漂移检测策略。
- **教学与研究友好**：用户可以在 Web UI 中直观看到不同算法在数据流上的性能变化，适合用于在线学习、概念漂移和集成学习的实验演示。

## System Architecture

```text
app.py
  └── MultiPage router
      ├── Home
      ├── ML
      │   └── scikit-learn batch classification
      ├── Online ML
      │   └── River classifier/regressor + progressive validation
      ├── Ensemble learning
      │   └── LB / BOLE / WPWPE + uploaded CSV stream
      └── Contrast experiment
          └── multi-model comparison + metric/time/memory curves
```

The application is intentionally organized as an experiment workbench:

1. Load a built-in stream dataset or upload a CSV file.
2. Select target column, model family, base learners and drift detectors.
3. Run online evaluation sample by sample.
4. Track predictive metrics, elapsed time and memory usage.
5. Plot comparable curves for algorithm analysis.

## 核心算法：WPWPE

WPWPE 的核心思想是：在线数据流中，不同基学习器在不同阶段的表现会变化，因此集成模型不应使用固定权重。系统使用滑动窗口衡量每个基学习器的近期准确率，并根据错误率动态分配权重。

核心流程：

1. 对输入数据流逐条样本执行预测。
2. 四个在线基学习器分别输出类别概率。
3. 使用滑动窗口 Accuracy 计算每个模型的近期错误率。
4. 将错误率的倒数归一化为模型权重，错误率越低，权重越高。
5. 对各模型的类别概率做加权求和。
6. 选择概率最高的类别作为最终预测。
7. 使用真实标签更新基学习器与整体集成指标。
8. 按指定间隔记录 Accuracy、耗时和内存占用。

简化公式：

```text
e_i = 1 - rolling_accuracy_i
w_i = (1 / (e_i + eps)) / sum(1 / (e_j + eps))
P(y = c) = sum(w_i * P_i(y = c))
```

与普通投票集成相比，WPWPE 的优势在于：

- 使用概率级融合，而不是只看离散预测标签。
- 权重来自近期窗口表现，能够随数据流状态动态调整。
- 可以组合不同模型结构和不同漂移检测器，提高对多类型概念漂移的适应能力。
- 同时关注预测效果、时间成本和内存成本，便于评估实际部署价值。

## 功能模块

| 模块 | 文件 | 说明 |
| --- | --- | --- |
| 应用入口 | `app.py` | 设置页面配置，注册多页应用 |
| 页面路由 | `multipages.py` | Streamlit 多页面封装 |
| 首页 | `page/home.py` | 项目首页与说明展示 |
| 传统机器学习 | `page/machine_learning.py` | 基于 scikit-learn 的批处理分类实验 |
| 在线机器学习 | `page/online_machine_learning.py` | 基于 River 的分类/回归在线学习实验 |
| 集成学习 | `page/ensemble_learning.py` | LB、BOLE、WPWPE 等在线集成框架实验 |
| 对比实验 | `page/contrast_experiment.py` | 多模型、多指标并行对比分析 |
| 样式 | `style/style.css` | Streamlit 页面样式 |

## 支持的算法与数据集

### 离线学习

- KNN
- SVM
- Random Forest
- Iris、Wine、Breast Cancer 数据集
- PCA 二维可视化

### 在线分类

- Logistic Regression
- Naive Bayes
- Perceptron
- Hoeffding Tree Classifier
- Adaptive Random Forest Classifier
- Streaming Random Patches Classifier

### 在线回归

- Hoeffding Tree Regressor
- Adaptive Random Forest Regressor
- Streaming Random Patches Regressor

### 在线集成与漂移检测

- Leveraging Bagging（LB）
- BOLE
- WPWPE
- ADWIN、KSWIN、PageHinkley
- DDM、EDDM、HDDM_A、HDDM_W

### 数据集

- Elec2
- Phishing
- SMTP
- ChickWeights
- TrumpApproval
- Friedman synthetic stream
- 用户上传 CSV 数据

## 技术栈

- Python 3.9
- Streamlit 1.20.0
- River 0.15.0
- scikit-learn
- pandas / numpy
- matplotlib / seaborn
- memory-profiler
- LightGBM
- streamlit-lottie

## 快速开始

安装依赖：

```bash
pip install streamlit==1.20.0 river==0.15.0 scikit-learn pandas numpy matplotlib seaborn memory-profiler lightgbm streamlit-lottie pillow requests
```

启动应用：

```bash
streamlit run app.py
```

启动后在浏览器中进入 Streamlit 页面，通过左侧导航选择不同实验模块。

## 使用方式

### 1. 传统机器学习实验

进入 `ML` 页面，选择内置数据集和分类器，调整超参数后运行模型。系统会输出测试集 Accuracy，并通过 PCA 展示二维数据分布。

### 2. 在线机器学习实验

进入 `Online ML` 页面，选择分类或回归任务，再选择数据集、模型、指标和样本数量。系统使用 River 的 progressive validation 对数据流进行逐样本评估，并绘制指标、时间和内存曲线。

### 3. 在线集成学习实验

进入 `Ensemble learning` 页面，上传 CSV 数据或使用已有数据，选择目标列后运行 LB、BOLE 或 WPWPE。WPWPE 支持手动配置四个基学习器和滑动窗口大小。

### 4. 对比实验

进入 `Contrast experiment` 页面，选择多个在线模型或模型-漂移检测器组合，系统会在同一数据流上并行评估并输出对比曲线。

## What This Demonstrates

This repository demonstrates the ability to connect algorithm research with an executable system:

- Understanding of online learning, incremental model updates and progressive validation.
- Practical handling of concept drift through detectors and adaptive ensembles.
- Design of a custom ensemble method instead of only using off-the-shelf models.
- Engineering of an interactive experiment platform around model configuration, data upload, evaluation and visualization.
- Awareness of production constraints by measuring not only accuracy, but also runtime and memory consumption.

## 可继续优化方向

- 将算法逻辑从 Streamlit 页面中拆分为独立 Python 包，提升可测试性和复用性。
- 增加 `requirements.txt` 或 `pyproject.toml`，固定依赖版本。
- 为 WPWPE、模型工厂和数据加载逻辑补充单元测试。
- 增加实验结果导出功能，支持 CSV、图片和报告生成。
- 增加 Dockerfile，降低部署和复现实验环境的成本。

## License

See [LICENSE](LICENSE).
