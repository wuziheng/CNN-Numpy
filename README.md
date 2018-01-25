# 	CNN Inplementation With Numpy

​	近期准备找工作，感觉太久没有涉及深度学习相关的知识，面试起来十分生疏。准备简单的复习一下。所以我打算用numpy实现一个简单的可以训练，测试的cnn框架（model-free），包含一些主流的层和设计，以便复习与巩固基础。



**2018.01.22**

---

*Target1*:  实现mnist的训练与测试：

* layer: Conv2D, FullyConnect, MaxPooling, Softmax 
* activation: Relu
* method: Mini-batch Gradient Descent(SGD)，learning_rate = 1e-5




<img src="fig/iteration.jpg" style="zoom:60%"/>

|   version    | validation_acc | train_acc | inferencetime(ms/pf) |
| :----------: | :------------: | :-------: | :------------------: |
| **baseline** |     96.75%     |  97.15%   |       2(ms/pf)       |



**2018.01.24**

------

*Target2*: 　实现Variable与Operator分离设计：

* 完成Variable与Operator 类的设计与graph的注册功能，GLOBAL_VARIABLE_SCOPE作为全局所有Variable,Operator的索引(graph)，Operator,Variable类自己维护自己的child,parent列表。
* 完成Conv2D类的设计，对比上一版本进行测试通过。



**2018.01.25**

------

* 继续完成其他基本组件的Operator改写。

