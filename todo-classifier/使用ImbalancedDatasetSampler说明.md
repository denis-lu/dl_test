# ImbalancedDatasetSampler 使用说明

## 概述
我们已经成功将 `ImbalancedDatasetSampler` 集成到深度学习分类器中，用于处理数据不平衡问题。

## 主要修改

### 1. dl_models.py 文件修改
- **添加了自定义Dataset类**: `CustomDataset` 类支持 `ImbalancedDatasetSampler` 的要求
- **智能导入**: 使用 try-except 块安全导入 `torchsampler`，如果未安装会给出提示
- **修改Data_processor类**: 
  - 添加 `use_imbalanced_sampler` 和 `is_training` 参数
  - 训练时使用不平衡采样器，测试时使用标准加载器
  - 自动打印类别分布信息

### 2. dl_classifiers.py 文件修改
- **更新函数签名**: `dl_container` 和 `main` 函数都添加了不平衡采样器的控制参数
- **数据统计**: 显示正负样本数量和比例
- **配置选项**: 在主程序中可以轻松启用/禁用不平衡采样器

## 安装要求

首先需要安装 `torchsampler`:

```bash
pip install torchsampler
```

## 使用方法

### 1. 启用不平衡采样器（推荐）
```python
# 在 dl_classifiers.py 中设置
USE_IMBALANCED_SAMPLER = True
```

### 2. 禁用不平衡采样器
```python
# 在 dl_classifiers.py 中设置
USE_IMBALANCED_SAMPLER = False
```

### 3. 编程方式调用
```python
# 直接调用 main 函数
main(file_path, model_type, use_imbalanced_sampler=True)
```

## 工作原理

1. **训练阶段**: 
   - 使用 `ImbalancedDatasetSampler` 根据类别频率的倒数为样本分配采样权重
   - 每个epoch都会重新采样，确保各类别平衡
   - 自动打印类别分布信息

2. **测试阶段**: 
   - 使用标准的顺序数据加载器
   - 不进行采样，保持原始数据分布

## 输出信息

运行时会看到以下信息：
```
ImbalancedDatasetSampler 可用
数据集统计: 正样本=100, 负样本=300, 比例=0.333
将使用 ImbalancedDatasetSampler 处理数据不平衡问题
类别分布: {0: 300, 1: 100}
使用 ImbalancedDatasetSampler 处理数据不平衡
```

## 兼容性

- **向后兼容**: 如果没有安装 `torchsampler`，会自动回退到标准的数据加载方式
- **灵活配置**: 可以轻松启用或禁用不平衡采样器
- **多模型支持**: 支持 bi-lstm、textcnn、transformer 等所有现有模型

## 注意事项

1. **仅在训练时使用**: 不平衡采样器只在训练数据上使用，测试数据保持原始分布
2. **批次大小**: 建议根据类别数量调整批次大小，确保每个批次都包含不同类别的样本
3. **随机种子**: 采样器会影响数据顺序，但不会影响模型的可重现性（已设置随机种子）

## 性能预期

使用 `ImbalancedDatasetSampler` 后，您可以期待：
- 更平衡的训练批次
- 减少模型对多数类的偏向
- 提高少数类的召回率
- 整体分类性能的改善，特别是在不平衡数据集上