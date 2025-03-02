# VGG16
该系统是一个围绕图像数据进行准备、处理，并基于处理后的数据构建和训练图像分类模型，进而评估模型性能的完整系统。它涵盖了从原始图像数据集出发，经过数据筛选与缩减、数据增强、模型构建与训练，到最后模型评估及训练过程可视化等多个核心环节，旨在打造一个能够准确对动物图像进行多类别分类的应用系统，同时保障整个流程的稳定性、数据的高质量以及模型的良好泛化能力。

# 动物图像分类

本仓库包含了一个基于 VGG16 架构的深度学习模型，用于分类 10 种不同动物类别的图像。

数据集位于main分支下的以下 10 个文件夹，每个文件夹对应一个不同的动物类别：
![微信图片_20250302134020](https://github.com/user-attachments/assets/9a40b2fb-d356-4de4-9c61-b8e843d2841d)

每个文件夹包含三种格式的图片：每个动物类别共有 300 张图片，其中包括 100 张原始图像、100 张水平镜像图像和 100 张灰度图像。通过数据增强，这个数据集帮助模型更好地进行泛化。

原始图像：100 张图片
![微信图片_20250302134043](https://github.com/user-attachments/assets/a62022ee-253a-46d0-8847-c17501611387)

水平镜像图像：100 张图片
![微信截图_20250302134125](https://github.com/user-attachments/assets/ead5f325-21e3-448d-8441-b3c65ea91045)

灰度图像：100 张图片
![微信图片_20250302134140](https://github.com/user-attachments/assets/62c79be5-55eb-448f-a318-ac9fbf107e8f)



# 目录结构
数据集的组织结构如下所示：

    butterfly/
    cat/
    chicken/
    cow/
    dog/
    elephant/
    horse/
    sheep/
    spider/
    squirrel/
训练集和测试集用于训练和评估模型，测试集每个类别中包含 50 张随机选择的图片。

# 环境要求
要运行此代码，您需要以下 Python 库：

tensorflow

numpy

scikit-learn

matplotlib


您可以使用 pip 安装它们：pip install tensorflow numpy scikit-learn matplotlib


# 代码概述：该代码使用了 VGG16 预训练模型进行迁移学习，并应用了数据增强技术来提升训练数据集的效果。


# 模型架构：
使用 VGG16 作为基础模型，并在其基础上添加自定义的卷积层、批归一化层、Dropout 层和全连接层。

# 训练：
使用增强后的数据集训练模型。
使用 EarlyStopping 来防止过拟合。

# 评估：
在测试集上评估模型，输出分类报告、F1 值和召回率。
可视化：
绘制训练过程中的准确率和损失曲线，帮助分析模型的学习过程。
如何运行
准备数据集：

确保数据集被正确放置在适当的目录下。
运行代码：

直接运行包含上述代码的 Python 脚本。
nginx
复制
编辑
python your_script_name.py
结果
训练完成后，模型将在测试集上进行评估，并显示准确率、F1 值和召回率等结果。您还可以查看训练过程中的准确率和损失曲线图，分析模型的学习进展。

这个 README.md 文件提供了对数据集和代码结构的全面概述，并解释了如何有效使用此仓库。如果有更多问题，随时可以提问！
