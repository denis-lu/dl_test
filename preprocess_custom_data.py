#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预处理脚本：帮助用户准备自己的数据集用于TODO注释分类
"""

import pandas as pd
import os
import re
import argparse
import sys

def preprocess_todo_comments(comments):
    """
    预处理TODO注释，规范格式
    
    参数:
    comments: TODO注释列表
    
    返回:
    处理后的TODO注释列表
    """
    processed = []
    for comment in comments:
        # 确保是字符串
        if not isinstance(comment, str):
            comment = str(comment)
            
        # 删除多余的空白
        comment = comment.strip()
        
        # 确保TODO格式一致
        comment = re.sub(r'(?i)todo\s*:', 'TODO:', comment)
        
        processed.append(comment)
    
    return processed

def preprocess_excel_file(input_file, output_file, comment_col, form_col, quality_col):
    """
    预处理Excel数据文件
    
    参数:
    input_file: 输入Excel文件路径
    output_file: 输出Excel文件路径
    comment_col: 包含TODO注释的列名
    form_col: 包含Form标签的列名
    quality_col: 包含质量标签的列名
    """
    try:
        # 读取Excel文件
        print(f"正在读取 {input_file}...")
        df = pd.read_excel(input_file, engine='openpyxl')
        
        # 检查所需列是否存在
        required_cols = [comment_col, form_col, quality_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"错误: 输入文件缺少以下列: {', '.join(missing_cols)}")
            sys.exit(1)
        
        # 预处理TODO注释
        print("正在预处理TODO注释...")
        df[comment_col] = preprocess_todo_comments(df[comment_col])
        
        # 标准化Form列 (task或notice)
        print("正在标准化Form列...")
        df[form_col] = df[form_col].apply(lambda x: 'task' if str(x).lower().strip() == 'task' else 'notice')
        
        # 标准化Quality列 (1表示好, 0表示不好)
        print("正在标准化Quality列...")
        df[quality_col] = df[quality_col].apply(lambda x: 1 if int(x) == 1 else 0)
        
        # 保存处理后的数据
        print(f"正在保存到 {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_excel(output_file, index=False, engine='openpyxl')
        
        print(f"成功! 预处理后的文件已保存到 {output_file}")
        return df
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

def preprocess_diff_file(input_file, output_file):
    """
    预处理代码差异文件
    
    参数:
    input_file: 输入差异文件路径
    output_file: 输出差异文件路径
    """
    try:
        # 读取差异文件
        print(f"正在读取 {input_file}...")
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # 预处理差异内容
        print("正在预处理差异内容...")
        processed_lines = []
        for line in lines:
            # 确保每个差异项有正确的格式
            processed_lines.append(line.rstrip('\n'))
        
        # 保存处理后的数据
        print(f"正在保存到 {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines))
        
        print(f"成功! 预处理后的差异文件已保存到 {output_file}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

def create_config_file(excel_file, diff_file, comment_col, form_col, quality_col):
    """
    创建配置文件，以便脚本知道如何处理自定义数据
    
    参数:
    excel_file: Excel文件路径
    diff_file: 差异文件路径
    comment_col: 注释列名
    form_col: Form列名
    quality_col: 质量列名
    """
    config = {
        'excel_file': excel_file,
        'diff_file': diff_file,
        'comment_column': comment_col,
        'form_column': form_col,
        'quality_column': quality_col
    }
    
    config_file = './my_custom_data/config.py'
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, 'w') as f:
        f.write(f"# 自动生成的配置文件\n\n")
        f.write(f"EXCEL_FILE = '{excel_file}'\n")
        f.write(f"DIFF_FILE = '{diff_file}'\n")
        f.write(f"COMMENT_COLUMN = '{comment_col}'\n")
        f.write(f"FORM_COLUMN = '{form_col}'\n")
        f.write(f"QUALITY_COLUMN = '{quality_col}'\n")
    
    print(f"配置文件已创建: {config_file}")

def main():
    parser = argparse.ArgumentParser(description="预处理TODO注释分类的数据集")
    
    parser.add_argument('--excel', required=True, help='输入的Excel文件路径')
    parser.add_argument('--diff', required=True, help='输入的代码差异文件路径')
    parser.add_argument('--comment-col', default='TODO Comment', help='Excel中包含TODO注释的列名 (默认: "TODO Comment")')
    parser.add_argument('--form-col', default='Form', help='Excel中包含形式标签的列名 (默认: "Form")')
    parser.add_argument('--quality-col', default='Quality', help='Excel中包含质量标签的列名 (默认: "Quality")')
    parser.add_argument('--output-dir', default='./my_custom_data', help='输出目录 (默认: "./my_custom_data")')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构造输出文件路径
    output_excel = os.path.join(args.output_dir, 'processed_todo_data.xlsx')
    output_diff = os.path.join(args.output_dir, 'processed_diff')
    
    # 预处理文件
    preprocess_excel_file(args.excel, output_excel, args.comment_col, args.form_col, args.quality_col)
    preprocess_diff_file(args.diff, output_diff)
    
    # 创建配置文件
    create_config_file(output_excel, output_diff, args.comment_col, args.form_col, args.quality_col)
    
    print("\n处理完成! 接下来的步骤:")
    print(" 1. 修改todo-classifier/dl_classifiers.py中的file_path变量:")
    print(f"    file_path = \"{output_excel}\"")
    print(" 2. 修改todo-classifier/dl_classifiers.py中的diff_data路径:")
    print(f"    diff_data = read_file('{output_diff}')")
    print(" 3. 按照run_model_guide.md中的说明运行模型")

if __name__ == "__main__":
    main() 