# 项目名称

本项目用于处理时间序列数据并进行相关分析和建模。

## 环境设置

1. 创建并激活 Conda 环境：
    ```bash
    conda create -n timeser python=3.8
    conda activate timeser
    ```

2. 安装依赖包：
    ```bash
    pip install -r requirements.txt
    pip install openpyxl
    pip install imbalanced-learn
    ```

## 生成数据集

1. 进入 `process_data` 目录：
    ```bash
    cd process_data
    ```

2. 新建 `outputs` 文件夹，并将各个类别的数据文件放入该文件夹：
    ```bash
    mkdir outputs
    # 将各个类别的数据文件放入 outputs 文件夹
    ```

3. 运行数据集生成脚本：
    ```bash
    python generate_dataset.py
    ```

4. 将生成的 `ProcessedData` 文件夹复制到 `Dataset/UEA/Multivariate_ts` 目录下：
    ```bash
    cp -r ProcessedData ../Dataset/UEA/Multivariate_ts
    ```

## 运行主程序

进入项目根目录，运行主程序：
```bash
python main.py
```

## 建立模型

本文提出了一种新的多变量时间序列分类模型，结合了绝对位置编码（tAPE）和相对位置编码（eRPE）以及基于卷积的输入编码（ConvTran），以改进时间序列数据的位置和数据嵌入。以下是模型的详细介绍，包括每个层的概念、实现和层间连接。

### 模型总览

ConvTran模型的架构如下图所示，主要包括以下部分：

1. **嵌入层**：使用卷积网络对输入数据进行初步特征提取。
2. **绝对位置编码层**：对时间序列数据进行绝对位置编码（tAPE）。
3. **相对位置编码层**：对时间序列数据进行相对位置编码（eRPE）。
4. **多头自注意力层**：利用多头自注意力机制捕捉长距离依赖关系。
5. **前馈神经网络层**：进一步处理特征向量。
6. **输出层**：通过全连接层输出分类结果。

每一层的具体实现和层间连接如下所述。

### 数据处理

数据处理部分包括对多变量时间序列数据进行标准化，以确保每个变量的数据分布在相同的尺度上。这一步有助于提升模型的训练效果和稳定性。

### 嵌入层

嵌入层使用卷积神经网络对输入数据进行特征提取。具体实现如下：

```python
self.embed_layer = nn.Sequential(
    nn.Conv2d(1, emb_size*4, kernel_size=[1, 8], padding='same'),
    nn.BatchNorm2d(emb_size*4),
    nn.GELU()
)

self.embed_layer2 = nn.Sequential(
    nn.Conv2d(emb_size*4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
    nn.BatchNorm2d(emb_size),
    nn.GELU()
)
```

- **卷积层（Conv2d）**：使用二维卷积核进行特征提取。公式如下：
  ```markdown
  Conv2d(x) = W * x + b
  ```
  其中，W 为卷积核，* 表示卷积操作，b 为偏置。
- **批量归一化（BatchNorm2d）**：对卷积结果进行归一化处理，加速模型训练并提高稳定性。
- **激活函数（GELU）**：使用GELU激活函数，公式为：
  ```markdown
  GELU(x) = x * Φ(x)
  ```
  其中，Φ(x) 为标准正态分布的累积分布函数。

嵌入层的结构如下图所示：

（此处可有图）

### 绝对位置编码层

绝对位置编码层使用时间绝对位置编码（tAPE）对时间序列数据进行编码。具体实现如下：

```python
if self.Fix_pos_encode == 'tAPE':
    self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
elif self.Fix_pos_encode == 'Sin':
    self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
elif config['Fix_pos_encode'] == 'Learn':
    self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
```

- **时间绝对位置编码（tAPE）**：结合序列长度和输入嵌入维度，使编码更适合时间序列数据。
此部分参考原文Section 4.1

### 相对位置编码层

相对位置编码层使用相对位置编码（eRPE）对时间序列数据进行编码。具体实现如下：

```python
if self.Rel_pos_encode == 'eRPE':
    self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
elif self.Rel_pos_encode == 'Vector':
    self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
else:
    self.attention_layer = Attention(emb_size, num_heads, config['dropout'])
```

- **相对位置编码（eRPE）**：提高时间序列数据的泛化能力。
此部分参考原文Section 4.2

### 多头自注意力层

多头自注意力层利用多头自注意力机制捕捉长距离依赖关系。具体实现如下：

```python
self.attention_layer = nn.MultiheadAttention(emb_size, num_heads, dropout=config['dropout'])
self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)
```

- **多头自注意力（Multi-Head Attention, MHA）**：将输入序列映射为查询（Q）、键（K）和值（V）向量，通过点积计算注意力分数，并将结果加权求和。公式如下：
  ```markdown
  Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
  ```
  其中，d_k 为键向量的维度。

多头自注意力层的结构如下图所示：

（此处可有图）

### 前馈神经网络层

前馈神经网络用于进一步处理特征向量。具体实现如下：

```python
self.FeedForward = nn.Sequential(
    nn.Linear(emb_size, dim_ff),
    nn.ReLU(),
    nn.Dropout(config['dropout']),
    nn.Linear(dim_ff, emb_size),
    nn.Dropout(config['dropout'])
)
```

- **线性层（Linear）**：通过全连接层进行特征变换。
  ```markdown
  Linear(x) = W x + b
  ```
  其中，W 为权重矩阵，b 为偏置向量。
- **激活函数（ReLU）**：使用ReLU激活函数，公式为：
  ```markdown
  ReLU(x) = max(0, x)
  ```
- **Dropout**：使用Dropout防止过拟合。

### 输出层

输出层通过全连接层输出分类结果。具体实现如下：

```python
self.out = nn.Linear(emb_size, num_classes)
```

- **全连接层（Linear）**：将特征向量映射为分类结果。

### 前后连接

模型的层间连接如下：

1. 输入数据经过嵌入层，生成初步特征。
2. 初步特征通过绝对位置编码层进行位置编码。
3. 位置编码后的特征通过相对位置编码层进一步编码。
4. 编码后的特征输入到多头自注意力层，捕捉长距离依赖关系。
5. 自注意力后的特征通过前馈神经网络层进行处理。
6. 最终特征通过输出层生成分类结果。


# 后附原文Section 4的翻译，便于理解原理：
（机翻，注意自己核对）
### 4. 适用于MTSC的Transformer位置编码

我们设计了新的位置编码方法，以研究先前基于Transformer的时间序列分类工作中尚未充分研究的几个方面（参见第5.4节的分析）。

首先，我们提出了一种新的专用于时间序列数据的绝对位置编码方法，称为时间绝对位置编码（tAPE）。tAPE结合了序列长度和输入嵌入维度在绝对位置编码中的应用。接下来，我们引入了高效的相对位置编码（eRPE），以探索位置与输入编码的独立编码。然后，为了研究eRPE在Transformer模型中的集成，我们比较了将位置信息集成到注意力矩阵中的不同方法；最后，我们为我们的方法提供了高效的实现。

#### 4.1 时间绝对位置编码（tAPE）

绝对位置编码最初是为语言建模提出的，在这种场景中，通常使用512或1024的高嵌入维度来对长度为512的输入进行位置编码（Vaswani等，2017）。图1a显示了使用公式 (5) 计算的两个正弦位置嵌入之间的点积，其距离为K，具有不同的嵌入维度。显然，较高的嵌入维度（如512，红色粗线）可以更好地反映不同位置之间的相似性。

然而，高嵌入维度并不适合时间序列数据集。原因是大多数时间序列数据集的输入维度相对较低（例如，32个数据集中的28个的输入维度小于64），并且较高的嵌入维度可能会由于额外的参数而降低模型吞吐量（增加模型过拟合的可能性）。

另一方面，在低嵌入维度下，两个随机嵌入向量之间的相似度值很高，使得嵌入向量彼此非常相似。换句话说，我们不能充分利用嵌入向量空间来区分两个位置。图1b展示了嵌入维度等于128且长度等于30的情况下，第一个和最后一个位置嵌入的向量。在该图中，几乎一半的嵌入向量是相同的。这被称为各向异性现象（Liang等，2021）。各向异性现象使得位置编码在低嵌入维度下失效，因为嵌入向量变得彼此相似，如图1a（蓝线）所示。

因此，我们需要一种时间序列的位置嵌入，具有距离感知的同时也是各向同性的。为了纳入距离感知，我们建议在公式 (5) 中使用时间序列长度。在该公式中， \(\omega_k\) 指的是生成嵌入向量的正弦和余弦函数的频率。没有我们的修改，随着序列长度 \(L\) 的增加，位置的点积变得越来越不规律，导致距离感知的丧失。通过在公式 (5) 中的正弦和余弦函数的频率项中纳入长度参数，点积保持平滑的单调趋势。

```markdown
\[
\omega_{new} = \omega_k \times \frac{d_{model}}{L}
\]
```

我们的新tAPE位置编码与传统正弦位置编码进行了比较，以进一步说明。使用嵌入维度为128的向量，图2a-b分别显示了长度为1000和30的序列中距离为K的两个位置的点积（相似性）。如图2a所示，在传统的APE中，只有序列中最接近的位置具有单调递减趋势，大约从距离50开始（|K| > 50），随着时间序列中两个位置之间的距离增加，递减相似性趋势变得不那么明显。然而，tAPE具有更稳定的递减趋势，并且更稳定地反映了两个位置之间的距离。同时，图2b显示了tAPE的嵌入向量彼此不那么相似，这得益于更好地利用嵌入向量空间来区分两个位置。

#### 4.2 高效的相对位置编码（eRPE）

在机器翻译和计算机视觉中，有多种上述第3.3.2节相对位置嵌入的扩展（Huang等，2020；Wu等，2021；Dufter等，2022）。然而，输入嵌入是所有先前相对位置编码方法的基础（如图3a所示，将位置矩阵添加或乘以查询、键和值矩阵）。在本研究中，我们引入了一种独立于输入嵌入的高效相对位置编码模型（图3b）。

特别是，我们提出了以下公式：

```markdown
\[
\alpha_i = \sum_{j \in L} \left( \frac{\exp(e_{i,j})}{\sum_{k \in L} \exp(e_{i,k})} + w_{i-j} \right) x_j
\]
```

其中， \(L\) 是序列长度， \(A_{i,j}\) 是注意力权重， \(w_{i-j}\) 是一个可学习的标量（即 \(w \in \mathbb{R}^{O(L)}\) ），表示位置 \(i\) 和 \(j\) 之间的相对位置权重。

值得比较的是相对位置编码和注意力的优缺点，以确定哪种属性对于时间序列数据的相对位置编码更为理想。首先，相对位置嵌入 \(w_{i-j}\) 是具有静态值的输入无关参数，而注意力权重 \(A_{i,j}\) 是由输入序列的表示动态确定的。换句话说，注意力通过加权策略适应输入序列（输入自适应加权（Vaswani等，2017））。输入自适应加权使得模型能够捕捉不同时间点之间复杂的关系，这是我们在提取时间序列中的高层概念时最期望的属性。例如，这是时间序列中的季节性成分。然而，当我们有有限的数据时，使用注意力时更容易过拟合。

#### 4.2.1 高效实现：索引

为了实现公式 (14) 中的eRPE的高效版本，对于长度为 \(L\) 的输入时间序列，对于每个头，我们创建一个大小为 \(2L - 1\) 的可训练参数 \(w\) ，因为最大距离为 \(2L - 1\) 。然后，对于两个位置索引 \(i\) 和 \(j\) ，相应的相对标量是 \(w_{i-j+L}\) ，其中索引从1开始而不是0（1基索引）。因此，我们需要从 \(2L - 1\) 向量中索引 \(L^2\) 个元素。

在GPU上，更高效的索引方法是使用gather，这只需要内存访问。在推理时，可以预先计算并缓存从 \(2L - 1\) 向量中索引的 \(L^2\) 个元素，以进一步提高处理速度。如表1所示，我们提出的eRPE在内存和时间复杂度方面都比文献中的现有相对位置编码方法更高效。

#### 4.3 ConvTran

现在我们来看看如何利用我们的新位置编码方法构建时间序列分类网络。根据之前的讨论，全局注意力相对于序列长度具有二次复杂度。这意味着如果我们直接将公式 (14) 中提出的注意力应用于原始时间序列，对于长时间序列计算将非常慢。因此，我们首先使用卷积来减少序列长度，然后在特征图减少到计算量较小的大小后应用我们提出的位置编码。见图4，卷积块作为第一个组件，后面是注意力块。

使用卷积的另一个好处是卷积操作非常适合捕捉局部模式。通过将卷积作为我们架构的第一个组件，我们可以捕捉到原始时间序列中存在的任何区分性局部信息。

如图4所示，在卷积层的第一步中，应用M个时间滤波器到输入数据。在这一步中，模型提取输入序列中的时间模式。接下来，时间滤波后的输出与形状为 \(d_{model} \times dx \times M\) 的空间滤波器卷积，以捕捉多变量时间序列中变量之间的相关性，并构建 \(d_{model}\) 大小的输入嵌入。这种分离的时间和空间卷积类似于“Sandler等（2018）”中的“倒瓶颈”。它首先扩展输入通道的数量，然后压缩它们。选择这个架构的关键原因是Transformer中的前馈网络（FFN）也扩展了输入大小，然后将扩展的隐藏状态投射回原始大小，以捕捉空间交互。

# 优化点
## 1. 数据重采样形成非平衡数据集
原先的正常数据集较少，不符合正常分布，因此使用Smote重采样形成非平衡数据集

## 2. Focal Loss
原先的交叉熵损失函数对于非平衡数据集的分类效果不佳，因此使用Focal Loss进行改进

## 3. 调整学习率
使用ReduceLROnPlateau，相比较原先的固定学习率，可以更好地调整学习率，提高模型的收敛速度和性能

## 4. 更多的正则化
在模型中加入更多的正则化项，如Dropout等，以防止过拟合

## 5. 混合精度
使用混合精度训练，可以加速训练过程，提高模型性能