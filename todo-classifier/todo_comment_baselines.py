# encoding=utf-8
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import pandas as pd
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
import torch
from cb_embedding import code_bert_embedding
from ml_classifiers import *
import logger
import glo
import re
warnings.filterwarnings("ignore")


def load_all_data(file_path):
    data = pd.read_excel(file_path, engine='openpyxl')
    data = data.iloc[0:2863,:10]
    return data

def tmp_label_process(labeled_data):
    comment_labels = labeled_data['Form'].apply(lambda x:1 if x.lower().strip()=='task' else 0)
    good_bad_labels = labeled_data['Quality'].apply(lambda x:1 if x == 1 else 0)
    return comment_labels, good_bad_labels


def clean_todo_comment(todo_data):
    re1 = r"#\s*\d+"
    re2 = r"todo\s*\(\s*[^()]+\s*\)"
    data_processed = []
    for todo in todo_data:
        new = re.sub(re2, "todo <info_tag>", todo)
        new = re.sub(re1, "<link_id>", new)
        data_processed.append(new)
    return data_processed


def main(file_path):
    mylogger = logger.mylog("ml")
    labeled_data = load_all_data(file_path)
    comment_labels, good_bad_labels = tmp_label_process(labeled_data)
    pos_num_1 = np.sum(comment_labels == 1)
    pos_num_2 = np.sum(good_bad_labels == 1)
    print(pos_num_1, pos_num_2)
    glo._init()

    kf = KFold(n_splits=10, random_state=3408, shuffle=True)
    for train_index, test_index in kf.split(labeled_data):
        train_x, train_y_1, train_y_2 = np.array(labeled_data.iloc[train_index, 2]), np.array(comment_labels.iloc[train_index]), np.array(good_bad_labels.iloc[train_index])
        test_x, test_y_1, test_y_2= np.array(labeled_data.iloc[test_index, 2]), np.array(comment_labels.iloc[test_index]), np.array(good_bad_labels.iloc[test_index])
        # clean data
        train_x = clean_todo_comment(train_x)
        test_x = clean_todo_comment(test_x)
        train_vec_list = code_bert_embedding(list(train_x)).to(torch.device('cpu'))
        test_vec_list = code_bert_embedding(list(test_x)).to(torch.device('cpu'))
        mylogger.info("Embeddings finished!")
        # Logistic Regression, Random Forest, Gradient Boosting, KNN, Decision Tree, MLP, MultinomialNB
        mylogger.info("==================Todo comment category classifers=============")
        methods_container(train_vec_list, train_y_1, test_vec_list, test_y_1, mylogger, "category")
        mylogger.info("==================Todo comment score classifers=============")
        methods_container(train_vec_list, train_y_2, test_vec_list, test_y_2, mylogger, "score")


if __name__ == '__main__':
    file_path = "../data/todo_data_labeled-f.xlsx"
    main(file_path)
