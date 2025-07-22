# 如何使用您自己的数据集运行深度学习模型

本指南将帮助您使用自己的数据集在"what-makes-a-good-TODO-comment"项目中运行深度学习模型。

## 步骤1：准备数据集

您需要准备以下文件来替换原始数据集：

1. **标记数据Excel文件**：替换`./data/todo_data_labeled-f.xlsx`
   - 这个Excel文件应包含以下列：
     - `Form`列：标记TODO注释的类型（'task'或其他）
     - `Quality`列：标记TODO注释的质量（1表示好，0表示不好）
     - 第3列（索引为2）应包含TODO注释文本

2. **代码差异文件**：替换`./data/diff_java`
   - 这是一个包含代码差异的文本文件
   - 每行应对应到标记数据中的一个TODO注释

请确保这两个文件之间的行数匹配，并且每行对应的是相同的TODO注释。

## 步骤2：修改CodeBERT路径

在使用模型前，您需要下载CodeBERT预训练模型并设置正确的路径：

1. 下载[CodeBERT预训练模型](https://huggingface.co/microsoft/codebert-base)

2. 在以下两个文件中修改模型路径：
   - `todo-classifier/dl_models.py`中的第21行和第91行
   ```python
   model_path = '/your/path/to/codebert-base'  # 修改为您的CodeBERT路径
   ```

## 步骤3：运行模型

1. 安装必要的依赖（确保您已安装PyTorch、transformers、sklearn等必要库）：
```
pip install torch transformers pandas scikit-learn tqdm openpyxl imblearn
```

2. 运行深度学习模型：
```
cd what-makes-a-good-TODO-comment/package
python todo-classifier/dl_classifiers.py
```

默认情况下，脚本将运行两种模型（"bi-lstm"和"textcnn"）。如果您想添加transformer模型，您可以修改`dl_classifiers.py`文件末尾的`model_list`：

```python
model_list = ["bi-lstm", "textcnn", "transformer"]  # 添加transformer
```

## 步骤4：自定义数据处理（可选）

如果您的数据格式与原始数据不同，您可能需要修改以下部分：

1. 修改`load_all_data`和`tmp_label_process`函数以适应您的数据格式：

```python
def load_all_data(file_path):
    data = pd.read_excel(file_path, engine='openpyxl')
    # 调整下面这行以匹配您的数据大小和格式
    data = data.iloc[0:您的数据行数, :您的数据列数]  
    return data

def tmp_label_process(labeled_data):
    # 根据您的数据格式修改这些行
    comment_labels = labeled_data['您的形式标签列名'].apply(lambda x: 1 if x.lower().strip()=='task' else 0)
    good_bad_labels = labeled_data['您的质量标签列名'].apply(lambda x: 1 if x == 1 else 0)
    return comment_labels, good_bad_labels
```

2. 如果您的TODO注释格式不同，可能需要修改`clean_todo_comment`函数中的正则表达式。

## 结果输出

模型训练完成后，结果将保存在以下位置：

1. 最佳模型权重将保存在`./todo_{category/score}_{model_name}/epoch_{best_epoch}/model.ckpt`
2. 训练统计信息将保存在`./todo_{category/score}_{model_name}/epoch_{best_epoch}/training_stats.json`
3. 评估指标将输出到控制台和日志文件

## 问题排查

1. **内存不足**：如果遇到内存不足，可以减小`Config`类中的`batch_size`
2. **CUDA错误**：如果遇到CUDA相关错误，检查GPU是否可用，或将`device`设置为'cpu'
3. **文件路径错误**：确保所有文件路径都是相对于运行脚本的位置正确的 