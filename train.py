import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import classification_report, f1_score, recall_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16  # 导入VGG16预训练模型
import shutil
import random

# 设置matplotlib使用中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者 'SimHei'
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 设置数据路径
train_dir = r'D:\imgae\animal10_augmented'  # 数据增强后的路径
test_dir = r'D:/imgae/animal10_test'  # 新建的测试集目录

# 创建新的测试集目录，并从每个类别中选取50张图片
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

class_names = os.listdir(train_dir)

# 为每个类别创建一个子文件夹来存放测试集图片
for class_name in class_names:
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):  # 确保是文件夹
        image_paths = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
        selected_images = random.sample(image_paths, 50)  # 随机选择50张图片
        class_test_dir = os.path.join(test_dir, class_name)
        os.makedirs(class_test_dir, exist_ok=True)
        for image_path in selected_images:
            shutil.copy(image_path, class_test_dir)

print("已成功从每个类别中选取50张图片并放入测试集。")

# 数据增强（训练集）
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 图片归一化到[0, 1]
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.5, 1.5],
    channel_shift_range=20.0
)

# 创建训练数据生成器
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 获取类别数量
num_classes = len(train_generator.class_indices)

# 学习率衰减
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)

# 编译优化器
optimizer = Adam(learning_rate=lr_schedule)

# 使用预训练模型 VGG16 进行迁移学习
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # 冻结预训练模型

# 构建CNN模型
model = Sequential([
    base_model,
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2, padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2, padding='same'),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 编译模型
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 打印模型结构
model.summary()

# 使用EarlyStopping来防止过拟合
early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

# 训练模型
history = model.fit(
    train_generator,
    epochs=20,
    verbose=1,
    callbacks=[early_stopping]
)

# 数据增强（测试数据集通常只进行归一化）
test_datagen = ImageDataGenerator(rescale=1./255)

# 创建测试数据生成器
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 评估模型
y_true = test_generator.classes  # 获取真实标签
y_pred = np.argmax(model.predict(test_generator), axis=-1)  # 获取预测标签

# 输出分类报告
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# 计算F1值和召回率
f1 = f1_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
print(f"F1 Score: {f1:.4f}")
print(f"Recall Score: {recall:.4f}")

# 可视化训练过程
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='训练准确率')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('训练准确率')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='训练损失')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('训练损失')
plt.show()
