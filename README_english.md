## Pruning Tools by Moffett AI
**<font size='3'>This repo provides a wide spectrum of tools for neural network pruning in major deep learning platforms, including tensorflow, pytorch and mxnet.**


**<font size='3'>Two modes for using the pruning tools by Moffett AI:<font>**
1. [x] In order to prune their neural networks, users only need to replace the optimizer in their codes by the one provided by Moffett AI. No change in other codes is needed! For example: 
    ```key
    # import Moffett optimizer in pytorch training script 
    from optimizers import pytorch_pruning as pruning
    ......
    # replace SGD optimizer by Moffett AI's optimizer  
    # optimizer = torch.optim.SGD(...)
    optimizer = pruning.SGDSparse(...)
    ......
    ```

2. [x] Users can also use the sparse networks provided by Moffett AI to finetune on their own dataset, while the sparsity is kept. 
---

**<font size='3'>This repo includes the following contents:</font>**

**<font size='3'>1. pruning optimizers using in tensorflow, pytorch and mxnet：</font>**
 * [x] mxnet_pruning.NAGSparse
 * [x] mxnet_pruning.NadamSparse
 * [x] pytorch_pruning.SGDSparse
 * [x] pytorch_pruning.AdamSparse
 * [x] tensorflow1_pruning.MomentumOptimizerSparse (not support multiple GPUs so far)
 * [ ] tensorflow1_pruning.AdamOptimizerSparse

Detailed documents for pruning optimizers:
* [pytorch pruning optimizer](./docs/pytorch_parameters.md)
* [mxnet pruning optimizer](./docs/mxnet_parameters.md)
* [tensorflow v1 pruning optimizer](./docs/tensorflow1_parameters.md)

---

**<font size='3'>2. Examples of using pruning optimizers in mnist dataset:</font>**

<font size='3'>2.1. prune dense network:</font>

 * [x] mxnet version：
    `python3.7 examples/mnist/mxnet_pruning_simple.py`

 * [x] pytorch version:
    `python3.7 examples/mnist/pytorch_pruning_simple.py`

  * [x] tensorflow version:
    `python3.7 examples/mnist/tensorflow_pruning_simple.py`

   <font size='3'>2.2. continue pruning/training from a sparse network：</font> ([results](./docs/results.md#pruning-and-finetune-results))

   * [x] mxnet version:
    `python3.7 examples/mnist/mxnet_pruning_simple_resume.py`

   * [x] pytorch version:
    `python3.7 examples/mnist/pytorch_pruning_simple_resume.py`

   * [ ] tensorflow version:
    `python3.7 examples/mnist/tensorflow1_pruning_simple_resume.py`

*notes: current optimizers only support python3.7，more versions will be released soon. [mxnet installation](https://mxnet.apache.org/get_started/?platform=macos&language=python&), [pytorch installation](https://pytorch.org/)*

---
**<font size='3'>3. finetune sparse network in users' datasets </font>**

**<font size='3'>3.1.  imagenet数据集压缩后的模型在cifar10数据集上finetune:</font>**
* [x] pytorch version: ([results](./docs/results.md#task-2-finetune-sparse-network-on-classification-dataset))

    `python3.7 examples/cifar10/torch_res50_cifar10_224input.py`

    `python3.7 examples/cifar10/torch_res50_cifar10_32input.py`
* [ ] tensorflow 1.14 version
* [ ] mxnet version

**<font size='3'>3.2.  coco数据集压缩后的模型在全目标检测数据集上finetune:</font>** : 
* [x] pytorch版本: ([实验结果](./docs/results.md#task-3-finetune-sparse-network-on-detection-dataset))

    `python3.7 examples/cocoSubset/torch_res50_cocoSubset.py`

* [ ] tensorflow1.14版本
* [ ] mxnet版本

---
**<font size='3'>4. Moffett AI model zoo </font>**

我们同时提供一些已经稀疏化的预训练模型供使用，模型的稀疏率和性能见下表。模型数量会逐渐增加。目前仅提供模型，训练代码稍后也会提供。

**<font size='3'>分类模型 </font>**

|模型|框架|训练数据集|稀疏率|准确率|说明|
|-|-|-|-|-|-|
|resnet50v1d|mxnet|imagenet|95%|74.6%|同上|
|resnet50v1d|torch|imagenet|95%|74.6%|[Moffett IR生成<sup>1</sup>](./examples/cifar10/resnet50v1d_graph.png)|

**<font size='3'>检测模型 </font>**

|模型|框架|训练数据集|稀疏率|MAP|说明|
|-|-|-|-|-|-|
|resnet50v1d+centernet|mxnet|coco|80%|31.0%|基于gluoncv的预训练模型进行剪枝||
|resnet50v1d+centernet|torch|coco|80%|31.0%|[Moffett IR生成<sup>1</sup>](./examples/cocoSubset/resnet50v1d_centernet.png)|

[点击此连接Google drive下载模型](https://drive.google.com/open?id=1xZ-lDh1CGnaFMpsQft37kyfocPf16KuR)

[点击此连接Baidu drive下载](https://pan.baidu.com/s/1fL0WYtDJohzujl9AeZbY3w)，提取码：9irl。

<sup>1</sup>**<font size='3'>[Moffett IR转换不同深度学习框架模型的使用说明](./docs/reconstruct_network.md) </font>**
