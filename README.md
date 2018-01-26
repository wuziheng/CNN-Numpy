# 	CNN Inplementation With Numpy

​	复习深度学习相关知识。打算用numpy实现一个简单的可以训练，测试的cnn框架（model-free，model-based），包含一些主流的层和设计，以便复习与巩固基础。



**2018.01.22**

---

*Target1*:  （model-free）实现mnist的训练与测试：

* layer: Conv2D, FullyConnect, MaxPooling, Softmax 
* activation: Relu
* method: Mini-batch Gradient Descent(SGD)，learning_rate = 1e-5




<img src="fig/iteration.jpg" style="zoom:60%"/>

| version      | validation_acc | train_acc | inferencetime(ms/pf) |
| :----------- | :------------: | :-------: | :------------------: |
| **baseline** |     96.75%     |  97.15%   |       2(ms/pf)       |



**2018.01.24**

------

*Target2*: 　(model-based)实现Variable与Operator分离设计：

* 完成Variable与Operator 类的设计与graph的注册功能，GLOBAL_VARIABLE_SCOPE作为全局所有Variable,Operator的索引(graph)，Operator,Variable类自己维护自己的child,parent列表。（感觉有点像tf）
* 完成Conv2D类的设计，对比上一版本进行测试通过。



**2018.01.25**

------

* 完成其他基本组件的Operator改写。新版本支持隐式构建graph，调用converge（汇） Variable.eval()自动完成前向传播计算；调用对应的source(源)Variable.diff_eval()自动完成反向传播与导数计算；对于learnable的Variable，手动调用Variable.apply_gradient()完成梯度下降。（未来目标把上述操作封转到显示的graph 或者session类中）



**2018.01.26**

------

* 给train_epoch读入图片添加了shuffle
* 完成了不同的激活函数relu,leaky-relu,sigmoid,tanh,elu, prelu
* 完成了对激活函数的grad_check,实际上sigmoid确实容易出现gradient-vanish,所以一开始用1e-5学习率基本收敛的特别慢，所以实际测试里面调整到了1e-3
* 其实可比较性不强～不要当真，默认的init=MSRA(暂时就实现了这一种)

| version                    | validation_acc | train_acc | learning_rate | best_epoch |
| :------------------------- | :------------: | :-------: | :-----------: | :--------: |
| **SGD_RELU** (alpha=0)     |     96.42%     |  96.85%   |     1e-5      |     11     |
| **SGD_LRELU**(alpha=0.01)  |     97.46%     |  97.08%   |     1e-5      |     4      |
| **SGD_LRELU**(alpha=0.001) |     97.33%     |  96.40%   |     1e-5      |     1      |
| **SGD_SIGMOID**            |                |           |     1e-3      |            |
| **SGD_TANH**               |     96.09%     |  91.07%   |   1e-3~1e-4   |     1      |
| **SGD_ELU**                |                |           |     1e-5      |            |

