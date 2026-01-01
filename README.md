# ModernNCA (M-NCA) 复现指南

这份文档为您整理了复现 ICLR 2025 论文 **"Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later"** 中提出的 **ModernNCA (M-NCA)** 模型所需的架构细节与核心实现要点。

## 1. 模型架构 (Model Architecture)

模型的整体流程分为特征编码、非线性映射（Backbone）、以及基于距离的预测头。

### 1.1 输入特征编码 (Input Encoding)
*   **类别特征 (Categorical):** 使用 **One-hot Encoding**。
*   **数值特征 (Numerical):** 使用 **PLR (Periodic Linear Embeddings) - Lite 版本** (参考 TabR 论文)。
    *   PLR 将标量数值投影到高维空间，引入高频分量，增强模型捕捉细节的能力。
    *   *Lite版特点:* 线性层参数在所有特征间共享，以平衡复杂度与效率。

### 1.2 主干网络 $\phi(x)$ (Backbone / Embedding Network)
ModernNCA 使用一个多层感知机（MLP）作为映射函数 $\phi$，将编码后的输入 $x$ 映射到嵌入空间。

*   **基础模块 Block $g(x)$:**
    论文公式 (5) 及描述指出，由以下层顺序堆叠而成：
    1.  **Batch Normalization (1D)**
    2.  **Linear Layer**
    3.  **ReLU Activation**
    4.  **Dropout**
    5.  **Linear Layer**
    
    *结构公式:* 
    $$
    g(x) = \text{Linear}(\text{Dropout}(\text{ReLU}(\text{Linear}(\text{BatchNorm}(x)))))
    $$

*   **网络堆叠:**
    *   在初始线性层之上，堆叠一个或多个上述 $g(x)$ 模块。
    *   **Output Normalization:** 在整个网络的最后输出端，必须加一个额外的 **Batch Normalization** 来校准输出嵌入的分布。
    *   *注意:* 论文实验表明 Batch Normalization 优于 Layer Normalization。

*   **输出维度:** 不强制进行降维（Dimensionality Reduction），可以映射到比输入更高的维度以增强表达能力。

## 2. 核心逻辑与前向传播 (Forward Logic)

### 2.1 距离度量 (Distance Function)
虽然经典 NCA 使用欧氏距离平方，但 ModernNCA 实证发现直接使用 **欧氏距离 (Euclidean Distance)** 效果更好。

$$
\text{dist}(\phi(x_i), \phi(x_j)) = \|\phi(x_i) - \phi(x_j)\|_2
$$

### 2.2 随机邻域采样 (Stochastic Neighborhood Sampling, SNS) - **训练关键**
为了解决计算全量训练集距离的巨大开销，训练时采用 SNS 策略：
*   **操作:** 在每个 Mini-batch 训练中，随机采样训练集的一个子集 $\hat{\mathcal{D}}$ (Candidate Neighbors)。
*   **计算:** 仅计算当前 Batch 中的样本与子集 $\hat{\mathcal{D}}$ 中样本的距离。
*   **比例:** 采样率通常在 **30% - 50%** 之间效果最佳（既提升效率又增强泛化性，引入了额外的随机性）。

### 2.3 Soft-KNN 预测规则 (Prediction Rule)
无论是分类还是回归，都使用 Soft-KNN 形式，而非多数投票。

对于目标样本 $x_i$，其预测值 $\hat{y}_i$ 为邻居标签的加权和：

$$
\hat{y}_i = \sum_{(x_j, y_j) \in \hat{\mathcal{D}}, j \neq i} \alpha_{ij} \cdot y_j
$$

其中权重 $\alpha_{ij}$ 由 Softmax 计算得出：

$$
\alpha_{ij} = \frac{\exp\left(-\text{dist}(\phi(x_i), \phi(x_j))\right)}{\sum_{k \neq i} \exp\left(-\text{dist}(\phi(x_i), \phi(x_k))\right)}
$$

*   **分类任务:** $\hat{y}_i$ 是一个概率向量（Probability Vector）。
*   **回归任务:** $\hat{y}_i$ 是标量数值的加权平均。

## 3. 损失函数 (Loss Function)

ModernNCA 放弃了原始 NCA 的 "Sum of Probabilities" 目标，转而最小化 **负对数似然 (Negative Log-Likelihood)** 或 **均方误差 (MSE)**。

*   **分类 (Classification):** 最小化目标类别的负对数概率 (Soft-NN Loss)。
    $$
    \mathcal{L} = -\log P(\hat{y}_i = y_{\text{true}} \mid x_i)
    $$
*   **回归 (Regression):** 最小化 MSE。
    $$
    \mathcal{L} = (\hat{y}_i - y_{\text{true}})^2
    $$

## 4. 训练与推理细节 (Training & Inference Details)

| 环节         | 实现要点                                                     |
| :----------- | :----------------------------------------------------------- |
| **优化器**   | 使用 **SGD** (Stochastic Gradient Descent)，这比原始 NCA 使用的 L-BFGS 更适合深度学习框架且效果更好。 |
| **推理阶段** | **不再采样**。推理时，需使用**完整**的训练集 $\mathcal{D}$ 作为邻居库进行检索和加权预测。 |
| **自身掩码** | 在训练计算 Softmax 时，必须将样本自身排除（mask out self），即 $j \neq i$。 |
| **超参数**   | Batch Size 建议设为 1024。SNS 采样率建议在 [0.05, 0.6] 范围内搜索。 |

## 总结：从原始 NCA 到 ModernNCA 的关键改动表

| 组件                | 原始 NCA (2004)          | ModernNCA (2025)                                  |
| :------------------ | :----------------------- | :------------------------------------------------ |
| **映射函数 $\phi$** | 线性变换 (Linear Matrix) | **非线性深度神经网络 (MLP + ResNet-like blocks)** |
| **特征处理**        | 原始特征                 | **PLR (数值) + One-hot (类别)**                   |
| **维度变化**        | 降维 ($d' < d$)          | **升维或保持** (为了更好的表达能力)               |
| **优化器**          | L-BFGS                   | **SGD**                                           |
| **损失函数**        | Sum of Probabilities     | **Log-Likelihood / MSE**                          |
| **邻居选择**        | 全量数据                 | **训练时 SNS (随机采样)，推理时全量**             |
| **预测模式**        | Hard KNN                 | **Soft KNN (Weighted Average)**                   |
| **距离度量**        | 欧氏距离平方             | **欧氏距离 ($L_2$ Norm)**                         |