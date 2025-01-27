--------------------------------------------------------------------------------------------------
1、Introduction of this course
01_introduction.pptx

参考书

“Machine Learning” (ML) 和 “Machine Learning and having it Deep and Structured” (MLDS) 有何不同？

Speech Recognition 语音识别
Image Recognition 图像识别
Playing Go 下围棋
Dialogue System 对话系统

Goodness of function f 函数优度 f

Learning Theory 学习理论
scenario 设想
Supervised Learning 监督学习
Semi-supervised Learning 半监督学习
Transfer Learning 迁移学习
Unsupervised Learning 无监督学习
Reinforcement Learning 强化学习
task
Regression 回归
Classification 分类
Structured Learning 结构化学习
method
Linear Model 线性模型
Non-linear Model
Deep Learning 深度学习
SVM, decision tree, K-NN … SVM、决策树、K-NN……

Regression
The output of the target function 𝑓 is “scalar”.

Binary Classification 二元分类
Multi-class Classification 多类分类

Spam filtering 垃圾邮件过滤

Document Classification 文档分类

Hierarchical Structure 层次结构

需要注意下围棋的训练数据的组织形式

半监督学习和迁移学习的区别是同样是猫狗分类的任务，
半监督学习使用的全部是猫和狗的图片，只不过有一部分图片没有标签，
迁移学习除了使用猫和狗的图片，还使用了其他图片，这些图片和猫狗无关

Data not related to the task considered (can be either labeled or unlabeled) 与考虑的任务无关的数据（可以标记或不标记）

Machine Reading: Machine learns the meaning of words from reading a lot of documents without supervision 机器阅读：机器无需监督，通过阅读大量文档来学习单词的含义

Beyond Classification 超越分类
Speech Recognition 语音识别—没有办法穷举声音可能得所有输出
Machine Translation 机器翻译—没有办法穷举所有可能翻译的结构

Learning from critics 向批评家学习

Convolutional Neural Network (CNN) 卷积神经网络（CNN）


--------------------------------------------------------------------------------------------------
2、Regression
02_Regression.pptx

Regression 回归
Case Study 案例研究

Stock Market Forecast 股市预测
Dow Jones Industrial Average at tomorrow 明天的道琼斯工业平均指数
Self-driving Car
Recommendation

Estimating the Combat Power (CP) of a pokemon after evolution 评估宝可梦进化后的战斗力 (CP)

feature 特征
weight 权重
bias 偏差
线性模型和多个特征值之前的关系：𝑦=𝑏+ ∑▒〖𝑤_𝑖 𝑥_𝑖 〗

Goodness of Function 函数优度
Training Data: 10 pokemons 训练数据：10 只 Pokemon

Estimated y based on input function 根据输入函数估计 y
Estimation error 估计误差

The color represents L(𝑤,𝑏). 颜色代表 L(𝑤,𝑏)。

Gradient Descent 梯度下降

Negative 负
Positive 正
Increase w 增加
Decrease w 减少

η is called learning rate

Local optimal 局部最优
not global optimal 不是全局最优

Is this statement correct? 这种说法正确吗？

Stuck at local minima 卡在局部极小值
Stuck at saddle point 卡在鞍点
Very slow at the plateau 在高原上非常缓慢

Worry 担心
Don’t worry. In linear regression, the loss function L is convex. 别担心。在线性回归中，损失函数 L 是凸函数。
Formulation of 𝜕𝐿∕𝜕𝑤 and 𝜕𝐿∕𝜕𝑏 𝜕𝐿∕𝜕𝑤 和 𝜕𝐿∕𝜕𝑏 的公式

Generalization 概括
What we really care about is the error on new data (testing data) 我们真正关心的是新数据（测试数据）上的错误

Better! Could it be even better? 更好！还能更好吗？

Slightly better. 稍微好一点。
How about more complex model? 更复杂的模型怎么样？

The results become worse ... 结果变得更糟......

A more complex model yields lower error on training data. 更复杂的模型在训练数据上产生的错误更少。
If we can truly find the best function 如果我们真的能找到最好的函数

A more complex model does not always lead to better performance on testing data. 更复杂的模型并不总是能在测试数据上带来更好的表现。
This is Overfitting. 这就是过度拟合。
Select suitable model 选择合适的模型

There is some hidden factors not considered in the previous model …… 其中存在一些先前模型未考虑到的隐藏因素……

Pidgey 波波
Eevee 伊布
Weedle 威德尔
Caterpie 卡特皮

Regularization 正则化
The functions with smaller 𝑤_𝑖 are better 𝑤_𝑖 越小的函数越好
We believe smoother function is more likely to be correct 我们相信平滑的函数更可能是正确的
smoother 平滑
Why smooth functions are preferred? 为什么更倾向于平滑函数？
If some noises corrupt input xi when testing 如果测试时某些噪声破坏了输入 xi
A smoother function has less influence. 更平滑的函数影响较小。

Training error: larger𝜆, considering the training error less 训练误差：𝜆越大，考虑到训练误差越小
We prefer smooth function, but don’t be too smooth. 我们更喜欢平滑的函数，但不要太平滑。

Pokemon: Original CP and species almost decide the CP after evolution (there are probably other hidden factors) 口袋妖怪：原始 CP 和物种几乎决定了进化后的 CP（可能还有其他隐藏因素）
Gradient descent 梯度下降
Following lectures: theory and tips 接下来的讲座：理论和技巧
Overfitting and Regularization 过度拟合和正则化
Following lectures: more theory behind these 接下来的讲座：这些背后的更多理论
We finally get average error = 11.1 on the testing data 我们最终在测试数据上得到平均误差 = 11.1
How about another set of new data? Underestimate? Overestimate? 另一组新数据怎么样？低估？高估？
Following lectures: validation 接下来的讲座：验证


--------------------------------------------------------------------------------------------------
3、Where does the error come from?
03_Bias and Variance.pptx

bias 偏见，偏差
variance 方差

Estimator 估计量

Estimate the mean of a variable x 估计变量x的均值
assume the mean of x is 𝜇 假设均值
assume the variance of x is 𝜎^2 假设方差
m 表示算术平均值
𝐸[𝑚] 表示期望值
unbiased 无偏见的

V𝑎𝑟[𝑚] 表示 m 的方差

s 表示方差
𝐸[s] 表示 s 的期望值

If we can do the experiments several times

Parallel Universes 平行宇宙
In all the universes, we are collecting (catching) 10 Pokémons as training data to find 𝑓^∗  在所有的宇宙中，我们收集（捕捉）10个poksamons作为训练数据来找到𝑓^ *

Simpler model is less influenced by the sampled data 简单的模型受采样数据的影响较小
Consider the extreme case f(x) = c 考虑极端情况f(x) = c

We don’t really know the F^

Overfitting过度拟合
Underfitting 欠拟合

Diagnosis: 诊断
If your model cannot even fit the training examples, then you have large bias 如果您的模型甚至无法拟合训练样本，那么您存在较大的偏差。
If you can fit the training data, but large error on testing data, then you probably have large variance 如果您能够拟合训练数据，但在测试数据上存在较大误差，那么您可能具有较大的方差。
For bias, redesign your model: 对于偏差，重新设计您的模型：
Add more features as input 增加更多的输入特征
A more complex model 使用更复杂的模型

Very effective, but not always practical 非常有效，但并不总是实用

Regularization 正则化

There is usually a trade-off between bias and variance. 偏差和方差之间通常存在权衡。
Select a model that balances two kinds of error to minimize total error 选择一个平衡两种误差的模型，以最小化总误差
What you should NOT do: 你不应该做的事情：


--------------------------------------------------------------------------------------------------
4、Gradient Descent
04_Gradient Descent.pptx

Gradient Descent 梯度下降法
Stochastic 随机
Data normalization 数据归一化

Tuning your learning rates 调整学习率

Learning Rate 如果刚刚好，Learning Rate 顺着红色箭头走到最低点
Learning Rate 如果太小，步伐太小，如蓝色箭头所示，虽然也会走到最低点，但是速度会很慢
Learning Rate 如果太大，步伐太大，如绿色箭头所示，那么就走不到最低点，它永远在最低点两边震荡
Learning Rate 如果特别大，步伐特别大，如黄色箭头所示，那么参数就会越来越大，永远不能到达最低点

Adaptive Learning Rates 自适应学习率
Popular & Simple Idea: Reduce the learning rate by some factor every few epochs. 流行和简单的想法：每隔几个epoch就减少一些学习率。
  At the beginning, we are far from the destination, so we use larger learning rate 一开始，我们离目的地很远，所以我们使用较大的学习率
  After several epochs, we are close to the destination, so we reduce the learning rate 经过几个epoch之后，我们接近了目的地，所以我们降低了学习率
  E.g. 1/t decay: 𝜂^𝑡=𝜂∕√(𝑡+1) 1/t衰变
Learning rate cannot be one-size-fits-all 学习率不可能是放之四海而皆准的
Giving different parameters different learning rates 给出不同的参数不同的学习率

Adagrad
Stochastic Gradient descent 随机
Vanilla Gradient descent 一般的 Gradient descent

Contradiction 矛盾

Intuitive Reason 直观的原因

http://seed.ucsd.edu/mediawiki/images/6/6a/Adagrad.pdf
http://courses.cs.washington.edu/courses/cse547/15sp/slides/adagrad.pdf

Larger 1st order derivative means far from the minima 一阶导数越大，表示距离最小值越远

Some features can be extremely useful and informative to an optimization problem but they may not show up in most of the training instances or data. If, when they do show up, they are weighted equally in terms of learning rate as a feature that has shown up hundreds of times we are practically saying that the influence of such features means nothing in the overall optimization (it's impact per step in the stochastic gradient descent will be so small that it can practically be discounted). To counter this, AdaGrad makes it such that features that are more sparse in the data have a higher learning rate which translates into a larger update for that feature (i.e. in logistic regression that feature's regression coefficient will be increased/decreased more than a coefficient of a feature that is seen very often). 有些特征可能对优化问题非常有用，但它们可能不会出现在大多数训练实例或数据中。如果，当它们确实出现时，它们在学习率方面的权重是相等的，作为一个出现了数百次的特征，我们实际上是在说，这些特征的影响在整体优化中没有任何意义（它在随机梯度下降中的每一步的影响将非常小，几乎可以被贴现）。为了解决这个问题，AdaGrad使数据中更稀疏的特征具有更高的学习率，从而转化为该特征的更大更新（例如，在逻辑回归中，特征的回归系数将比经常看到的特征的系数增加/减少更多）。

Simply put, sparse features can be very useful. I don't have an example of application in neural network training. Different adaptive learning algorithms are useful with different data (it would really depend on what your data is and how much importance you place on sparse features). 简单地说，稀疏特征非常有用。我没有一个应用在神经网络训练中的例子。不同的自适应学习算法对不同的数据有用（这实际上取决于你的数据是什么以及你对稀疏特征的重视程度）。

Comparison between different parameters 不同参数比较

Second Derivative二阶导数

Stochastic Gradient descent 随机

Two approaches update the parameters towards the same direction, but stochastic is faster!

Feature Scaling 特征缩放
https://standardfrancis.wordpress.com/2015/04/16/batch-normalization/

Theory 理论
Stuck at local minima 卡在局部极小值
Stuck at saddle point 卡在鞍点
Very slow at the plateau 在高原上非常缓慢


--------------------------------------------------------------------------------------------------
5、Classification: Probabilistic Generative Model
05_Classification (v2).pptx

Classification: Probabilistic Generative Model 分类：概率生成模型

Credit Scoring 信用评分
  Input: income, savings, profession, age, past financial history …… 输入：收入，储蓄，职业，年龄，过去的财务历史......
  Output: accept or refuse 输出：accept或refuse
Medical Diagnosis 医学诊断
  Input: current symptoms, age, gender, past medical history ……  输入：当前症状，年龄，性别，既往病史......
  Output: which kind of diseases 输出：哪一类疾病
Handwritten character recognition 手写字符识别
Face recognition 人脸识别
Input: image of a face, output: person 输入：人脸图像，输出：人

Total: sum of all stats that come after this, a general guide to how strong a pokemon is 总值：在此之后的所有属性的总和，即关于pokemon的强大程度的一般指南
HP: hit points, or health, defines how much damage a pokemon can withstand before fainting HP：生命值或生命值决定了pokemon在昏厥前能够承受多少伤害
Attack: the base modifier for normal attacks (eg. Scratch, Punch) 攻击:普通攻击的基础修正值。划痕,打孔)
Defense: the base damage resistance against normal attacks 防御:抵抗普通攻击的基础伤害
SP Atk: special attack, the base modifier for special attacks (e.g. fire blast, bubble beam) SP攻击：特殊攻击，特殊攻击的基础修饰符（如火焰爆炸，气泡束）
SP Def: the base damage resistance against special attacks SP防御:对特殊攻击的基础伤害抵抗
Speed: determines which pokemon attacks first each round 速度：决定每个回合哪个口袋妖怪先攻击

Can we predict the “type” of pokemon based on the information? 我们能否根据这些信息预测pokemon的“类型”？

Ideal Alternatives 理想的替代品
Perceptron 感知器

Testing: closer to 1 → class 1; closer to -1 → class 2 
接近 1 为第一类，接近 2 为第二类

to decrease error 为了减少误差
Penalize to the examples that are “too correct” … 惩罚那些“太正确”的例子……

Estimating the Probabilities From training data 从训练数据估计概率

Prior 先前的，事先的

Gaussian Distribution 高斯分布
Ref: https://www.cs.cmu.edu/~epxing/Class/10701-08s/recitation/gaussian.pdf

Maximum Likelihood 最大似然


--------------------------------------------------------------------------------------------------
6、Classification: Logistic Regression
06_Logistic Regression (v4).pptx

Good ref:
http://www.win-vector.com/blog/2011/09/the-simpler-derivation-of-logistic-regression/
http://www.cs.columbia.edu/~smaskey/CS6998/slides/statnlp_week6.pdf
http://www.win-vector.com/dfiles/LogisticRegressionMaxEnt.pdf

Classification: Logistic Regression 分类：逻辑回归

Cross entropy between two Bernoulli distribution 两个伯努利分布之间的交叉熵

cross entropy 代表两个 Distribution 有多接近，如果 cross entropy 算出来为 0，代表两个 Distribution 一模一样

Discriminative 判别性
Generative 生成性

Will we obtain the same set of w and b? 我们会得到相同的一组 w 和 b 吗？

The same model (function set), but different function may be selected by the same training data. 相同的模型（函数集），但是相同的训练数据可能会选取不同的函数。


--------------------------------------------------------------------------------------------------
7、Introduction of Deep Learning
07_DL.pptx

Deep learning 深度学习
attracts lots of attention 吸引了大量关注
I believe you have seen lots of exciting results before. 我相信您之前已经看到过很多令人兴奋的结果。
Deep learning trends at Google. Source: SIGMOD/Jeff Dean 谷歌的深度学习趋势。资料来源：SIGMOD/Jeff Dean

1958: Perceptron (linear model) 1958 年：感知器（线性模型）
1969: Perceptron has limitation 1969 年：感知器存在局限性
1980s: Multi-layer perceptron 1980 年代：多层感知器
Do not have significant difference from DNN today 与今天的 DNN 没有显著差异
1986: Backpropagation 1986 年：反向传播
Usually more than 3 hidden layers is not helpful 通常超过 3 个隐藏层没有帮助
1989: 1 hidden layer is “good enough”, why deep? 1989 年：1 个隐藏层“足够好”，为什么要深？
2006: RBM initialization (breakthrough) 2006 年：RBM 初始化（突破）
2009: GPU
2011: Start to be popular in speech recognition 2011 年：开始在语音识别中流行
2012: win ILSVRC image competition 2012 年：赢得 ILSVRC 图像竞赛

Neural Network 神经网络
Different connection leads to different network structures 不同的连接方式导致不同的网络结构
Network parameter 𝜃: all the weights and biases in the “neurons” 网络参数𝜃：“神经元”中的所有权重和偏差

neuron 神经元

Special structure 特殊结构

Using parallel computing techniques to speed up matrix operation 利用并行计算技术加速矩阵运算

Feature extractor replacing feature engineering 特征提取器取代特征工程

Each dimension represents the confidence of a digit. 每个维度代表一个数字的置信度。

A function set containing the candidates for Handwriting Digit Recognition 包含手写数字识别候选函数集
You need to decide the network structure to let a good function in your function set. 您需要确定网络结构，以便在函数集中实现良好的函数

Trial and Error 反复试验
Intuition 直觉

Evolutionary Artificial Neural Networks 进化人工神经网络

Convolutional Neural Network (CNN) 卷积神经网络（CNN）

Cross Entropy 交叉熵

Gradient Descent 梯度下降

I hope you are not too disappointed 我希望你不会太失望

Backpropagation 反向传播

Backpropagation: an efficient way to compute 𝜕𝐿∕𝜕𝑤 in neural network 反向传播：在神经网络中计算𝜕𝐿∕𝜕𝑤的有效方法

Concluding Remarks 结束语

What are the benefits of deep architecture? 深度架构有什么好处？

Not surprised, more parameters, better performance 不意外，参数更多，性能更好

Universality Theorem 普遍性定理

Any continuous function f 任何连续函数 f
Can be realized by a network with one hidden layer 都可以通过具有一个隐藏层的网络实现
(given enough hidden neurons) （给定足够的隐藏神经元）
Why “Deep” neural network not “Fat” neural network? 为什么是“深”神经网络而不是“胖”神经网络？


--------------------------------------------------------------------------------------------------
8、Backpropagation
08_BP (v2).pptx

Forward pass 前传
Backward pass 后传


--------------------------------------------------------------------------------------------------
9、“Hello world” of Deep Learning
09_Keras.pptx

常见的激活函数：softplus, softsign, relu, tanh, hard_sigmoid, linear
https://keras.io/api/layers/activations/#available-activations

常见损失函数：https://keras.io/api/losses/

常见优化参数方法：SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
https://keras.io/api/optimizers/

Mini-batch

epoch

Backpropagation 反向传播

To compute the gradients efficiently, we use backpropagation. 为了有效地计算梯度，我们使用反向传播。

Very flexible 非常灵活
Need some effort to learn 需要花些功夫去学习
Interface of TensorFlow or Theano TensorFlow 或 Theano 的接口

x_train：(Number of training examples, 28, 20)
y_train：(Number of training examples, 10)

Batch size influences both speed and performance. You have to tune it. 批量大小会影响速度和性能。您必须对其进行调整。

Shuffle the training examples for each epoch 每个时期都对训练样本进行洗牌。


--------------------------------------------------------------------------------------------------
10、Tips for Deep Learning
10_DNN tip.pptx

CNN 中遗留的问题：
1、CNN 中有 max pooling 架构，但是 max pooling 不能微分，这个在 Gradient descent 要怎么处理
2、L1 的 Regularization 是什么

Recipe of Deep Learning 深度学习的流程

k nearest neighbor k最近邻居
decision tree 决策树
k nearest neighbor 和 decision tree 这些方法在 training data 上的正确率肯定是 100%

Do not always blame Overfitting 不要总是责怪过度拟合

Early Stopping
Regularization 正则化
Dropout
New activation function
Adaptive Learning Rate


New activation function

Deeper usually does not imply better. 更深通常并不意味着更好。
为什么

accuracy 准确性

Vanilla Gradient descent 一般的 Gradient descent

Vanishing Gradient Problem 梯度消失问题
Already converge 已经收敛

Intuitive way to compute the derivatives … 计算导数的直观方法。

sigmoid function 的问题
sigmoid function 会造成梯度消失的问题

ReLU
Fast to compute 计算速度快
Biological reason 生物的原因
Infinite sigmoid with different biases 无穷多个 sigmoid 叠加形成 ReLU
Vanishing gradient problem 梯度消失问题

为什么 ReLU 可以解决梯度消失的问题
A Thinner linear network 更细的线性网络
Thinner linear network 相当于是 linear 的，不会出现梯度逐渐减小的问题

使用 ReLU 会使整个 network 变成 linear，但我们需要的是 deep network，这是矛盾的吗？

ReLU 不能微分，这个要怎么处理
ReLU 不能微分只是在输入为 0 的时候，其他地方都是可以微分的

ReLU - variant
𝐿𝑒𝑎𝑘𝑦 𝑅𝑒𝐿𝑈
𝑃𝑎𝑟𝑎𝑚𝑒𝑡𝑟𝑖𝑐 𝑅𝑒𝐿𝑈

Maxout network
Maxout network 就是使用训练数据自动学习 activation function
ReLU is a special cases of Maxout ReLU是Maxout的一种特殊情况

Maxout network 有办法做到和 ReLU 一模一样的事情，当然它也可以做到其他 activation function 一样的事情

Maxout network 要怎么 training


Adaptive Learning Rate

RMSProp Adagrad 的进阶版

Hard to find optimal network parameters 难以找到最佳的网络参数
Stuck at local minima 卡在局部极小值
Stuck at saddle point 卡在鞍点
Very slow at the plateau 在高原上非常缓慢
local minima 很难出现，从概率的角度分析，假设一个参数的 local minima 出现的概率是 p，那么 1000 个参数同时出现 local minima 的概率就是 p 的 1000 次方，p 的 1000 次方是一个很小的值，所以 local minima 很难出现
Momentum 可以一定程度上处理 local minima 的问题

Momentum 动量
How about put this phenomenon in gradient descent? 如何将这种现象放入梯度下降中？

Adam RMSProp + Momentum 


--------------------------------------------------------------------------------------------------
11、Convolutional Neural Network
11_CNN.pptx

卷积神经网络

Max Pooling 不能微分要怎么解决

暂时忽略


--------------------------------------------------------------------------------------------------
12、Why Deep?
12_Why.pptx

Modularization 模块化

暂时忽略


--------------------------------------------------------------------------------------------------
13、Semi-supervised Learning
13_semi.pptx

Semi-supervised Learning 半监督学习

Transductive learning 传导学习
Inductive learning 归纳学习

Usually with some assumptions 通常有一些假设

Outline 概要
Semi-supervised Learning for Generative Model 生成模型的半监督学习
Low-density Separation Assumption 低密度分离假设
Smoothness Assumption 平滑度假设
Better Representation 更好的表示

Semi-supervised Learning for Generative Model 生成模型的半监督学习

Decision Boundary 决策边界

The algorithm converges eventually, but the initialization influences the results. 算法最终收敛，但初始化会影响结果。

Solved iteratively 迭代求解

Low-density Separation Assumption 低密度分离假设

Self-training

How to choose the data set remains open 如何选择数据集仍未确定
You can also provide a weight to each data. 您还可以为每个数据提供权重。

Similar to semi-supervised learning for generative model 类似于生成模型的半监督学习

Entropy-based Regularization 基于熵的正则化

Smoothness Assumption 平滑度假设

More precisely: 更准确地说应该是:
x is not uniform. X不是均匀的。
a high density region 高密度区域

Represented the data points as a graph 将数据点表示为图形
Graph representation is nature sometimes. 图形表示有时是自然的。
E.g. Hyperlink of webpages, citation of papers 例如：网页的超链接，论文的引用
Sometimes you have to construct the graph yourself. 有时候你必须自己构造这个图。

K Nearest Neighbor K近邻
e-Neighborhood e附近


--------------------------------------------------------------------------------------------------
14、Unsupervised Learning: Principle Component Analysis
14_PCA (v3).pptx
使用 2016 年的 ppt，因为所有的视频都是使用的 2016 年的 PPT
14_dim reduction (v5).pptx

Unsupervised Learning: Linear Dimension Reduction 无监督学习：线性降维

Clustering & Dimension Reduction 聚类和降维
Generation 生成

聚类方法：
K-means K 均值
Hierarchical Agglomerative Clustering (HAC) 层次凝聚聚类 (HAC)

聚类有一个问题，就是聚类会强制把每个事物都归结到一类中，而现实情况是大部分事物既具有A类的特征，也有B类的特征，单纯归结到一类不是很合理，所以需要降维，通过降维表示事物在每一类的特征

Distributed Representation 分布式表示
Dimension Reduction 降维

Dimension Reduction方法：
Feature selection
Principle component analysis (PCA)

Reduce to 1-D 说明 z 是一个 scale

PCA – Another Point of View PCA——另一种观点

Symmetric 对称
positive-semidefinite 半正定
(non-negative eigenvalues) （非负特征值）

𝑤^1 is the eigenvector of the covariance matrix S 𝑤^1 是协方差矩阵 S 的特征向量
Corresponding to the largest eigenvalue 𝜆_1 对应于最大特征值 𝜆_1

PCA - decorrelation PCA-去相关

principle components 主成分
ratio 比率

Non-negative matrix factorization (NMF)

Weakness of PCA PCA 的弱点

Matrix Factorization 矩阵分解

Latent semantic analysis (LSA)


--------------------------------------------------------------------------------------------------
15、Unsupervised Learning: Neighbor Embedding
15_TSNE.pptx

Unsupervised Learning: Neighbor Embedding 无监督学习：邻域嵌入 非线性降维

Manifold Learning 流形学习

Locally Linear Embedding (LLE) 局部线性嵌入 (LLE)
Laplacian Eigenmaps 拉普拉斯特征图
T-distributed Stochastic Neighbor Embedding(t-SNE) T 分布随机邻域嵌入 (t-SNE)

暂时忽略


--------------------------------------------------------------------------------------------------
16、Unsupervised Learning: Deep Auto-encoder
16_auto.pptx

Unsupervised Learning: Deep Auto-encoder 无监督学习：深度自动编码器

Compact representation of the input object 输入对象的紧凑表示
Can reconstruct the original object 能重建原来的物体吗
Learn together 一起学习

为什么同时需要编码器和解码器？

Recap: PCA 回顾:主成分分析
As close as possible 尽可能接近
Bottleneck later 瓶颈之后

Of course, the auto-encoder can be deep 当然，自动编码器可以是深度的
Initialize by RBM layer-by-layer 通过RBM逐层初始化
Symmetric is not necessary. 对称不是必须的。

De-noising auto-encoder 去噪auto-encoder
Contractive auto-encoder 收缩auto-encoder

Auto-encoder 应用：
Auto-encoder – Text Retrieval 自动编码器-文本检索
Vector Space Model 向量空间模型
Bag-of-word 
Semantics are not considered. 语义不被考虑。

The documents talking about the same thing will have close code. 讨论同一事物的文档将具有相近的代码。
LSA: project documents to 2 latent topics LSA: 2个潜在主题的项目文件

Auto-encoder – Similar Image Search 自动编码器-类似的图像搜索
Retrieved using Euclidean distance in pixel intensity space 使用欧几里得距离在像素强度空间检索

Auto-encoder for CNN
Convolution
Deconvolution
Pooling
Unpooling 的实现方法
1、记住原来的位置，在原来的位置上填充值，其他地方填充 0
2、直接在所有的地方都填充相同的值

Alternative: simply repeat the values 可选：简单地重复这些值
Actually, deconvolution is convolution. 实际上，反卷积就是卷积。

Auto-encoder – Pre-training DNN
Greedy Layer-wise Pre-training again 贪婪的分层预训练
DNN（深度神经网络）
RNN（递归神经网络）
CNN（卷积神经网络）

Find-tune by backpropagation 通过反向传播找到调谐


--------------------------------------------------------------------------------------------------
17、Unsupervised Learning: Word Embedding
17_word2vec (v2).pptx

Word Embedding 是 Dimension Reduction 降维的一个应用

为什么需要 Word Embedding

怎么做 Word Embedding

Word Embedding 能不能用 Auto-encoder？
不能，因为词汇的含义和词汇的上下文有关，而 Auto-encoder 其实是不考虑上下文的，图片的识别就不需要考虑上下文


--------------------------------------------------------------------------------------------------
18、Unsupervised Learning: Deep Generative Model
18_GAN (v3).pptx

太复杂了，暂时忽略


--------------------------------------------------------------------------------------------------
19、Transfer Learning
19_transfer.pptx

--------------------------------------------------------------------------------------------------
20、Recurrent Neural Network
20_RNN.pptx

--------------------------------------------------------------------------------------------------
21、Matrix Factorization
21_MF.pptx

Matrix Factorization 矩阵分解


--------------------------------------------------------------------------------------------------
22、Ensemble
22_Ensemble.pptx

--------------------------------------------------------------------------------------------------
23、Introduction of Structured Learning
23_Structured Introduction.pptx

--------------------------------------------------------------------------------------------------
24、Introduction of Reinforcement Learning
24_RL (v4).pptx

--------------------------------------------------------------------------------------------------

