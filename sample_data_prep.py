#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例脚本：演示如何准备数据集用于TODO注释分类
"""

import pandas as pd
import os
import re
import random

# 创建示例数据集的函数
def create_sample_dataset(output_excel_path, output_diff_path, sample_size=100):
    """
    创建一个示例数据集用于TODO注释分类
    
    参数:
    output_excel_path: 标记数据的输出Excel文件路径
    output_diff_path: 代码差异的输出文件路径
    sample_size: 示例数据集的大小
    """
    # 创建示例TODO注释
    todo_comments = [
        "TODO: 实现用户登录功能",
        "TODO: 修复页面布局问题",
        "TODO: 添加错误处理",
        "TODO: 优化数据库查询",
        "TODO: 实现缓存机制",
        "TODO: 添加测试用例",
        "TODO: 重构此方法",
        "TODO: 处理边缘情况",
        "TODO: 更新文档",
        "TODO: 添加日志记录"
    ]
    
    # 创建示例代码差异
    diff_templates = [
        "--- a/src/main.py\n+++ b/src/main.py\n@@ -10,6 +10,7 @@\n def process_data():\n     data = load_data()\n     # {}\n+    result = transform_data(data)\n     return result",
        "--- a/src/utils.py\n+++ b/src/utils.py\n@@ -25,6 +25,7 @@\n class Helper:\n     def __init__(self):\n         # {}\n+        self.config = load_config()\n         self.initialized = True",
        "--- a/src/api.py\n+++ b/src/api.py\n@@ -42,6 +42,7 @@\n @app.route('/users')\n def get_users():\n     # {}\n+    users = db.query(User).all()\n     return jsonify(users)"
    ]
    
    # 生成随机数据
    data = []
    diff_content = []
    
    for i in range(sample_size):
        # 随机选择一个TODO注释和一个差异模板
        todo = random.choice(todo_comments)
        diff = random.choice(diff_templates).format(todo)
        
        # 随机决定是否为task类型和是否为好的质量
        is_task = random.choice([True, False])
        is_good_quality = random.choice([1, 0])
        
        # 添加到数据列表
        data.append({
            'ID': i+1,
            'Link': f"https://github.com/example/repo/pull/{i+1}",
            'TODO Comment': todo,
            'Form': 'task' if is_task else 'notice',
            'Quality': is_good_quality,
            'Additional Info': f"Sample {i+1}"
        })
        
        # 添加到差异内容
        diff_content.append(diff)
    
    # 创建DataFrame并保存到Excel
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
    df.to_excel(output_excel_path, index=False, engine='openpyxl')
    
    # 保存差异内容到文件
    os.makedirs(os.path.dirname(output_diff_path), exist_ok=True)
    with open(output_diff_path, 'w') as f:
        f.write('\n'.join(diff_content))
    
    print(f"示例数据集已创建:")
    print(f" - 标记数据: {output_excel_path}")
    print(f" - 代码差异: {output_diff_path}")
    
    return df

def main():
    # 创建自定义数据集的示例
    output_dir = "./my_custom_data"
    excel_path = os.path.join(output_dir, "my_todo_data.xlsx")
    diff_path = os.path.join(output_dir, "my_diff_java")
    
    # 创建示例数据集
    df = create_sample_dataset(excel_path, diff_path, sample_size=50)
    
    print("\n数据集预览:")
    print(df[['TODO Comment', 'Form', 'Quality']].head())
    
    print("\n接下来的步骤:")
    print(" 1. 修改todo-classifier/dl_classifiers.py中的file_path变量:")
    print(f"    file_path = \"{excel_path}\"")
    print(" 2. 确保读取差异文件的路径也已更新")
    print(" 3. 按照run_model_guide.md中的说明运行模型")

if __name__ == "__main__":
    main() 