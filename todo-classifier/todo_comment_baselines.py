# encoding=utf-8
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from ml_classifiers import *
import logger
import glo
warnings.filterwarnings("ignore")


def load_all_data(file_path):
    data = pd.read_json(file_path, orient='records', lines=True)
    data = data.iloc[:758, :31]
    return data

def tmp_label_process(labeled_data):
    high_low_labels = labeled_data['Quality'].apply(lambda x:1 if x == 1 else 0)
    return high_low_labels


def main(file_path):
    mylogger = logger.mylog("ml")
    labeled_data = load_all_data(file_path)
    high_low_labels = tmp_label_process(labeled_data)
    labeled_data = labeled_data.drop(columns=['Quality', 'modelId'])
    pos_num = np.sum(high_low_labels == 1)
    print(pos_num)
    glo._init()

    # kf = KFold(n_splits=10, random_state=3408, shuffle=True)
    kf = StratifiedKFold(n_splits=10, random_state=3408, shuffle=True)
    for train_index, test_index in kf.split(labeled_data):
        train_x, train_y = np.array(labeled_data.iloc[train_index]), np.array(high_low_labels.iloc[train_index])
        test_x, test_y = np.array(labeled_data.iloc[test_index]), np.array(high_low_labels.iloc[test_index])
        # Logistic Regression, Random Forest, Gradient Boosting, KNN, Decision Tree, MLP, MultinomialNB
        mylogger.info("==================modelcard high_low classifers=============")
        methods_container(train_x, train_y, test_x, test_y, mylogger, "score")


if __name__ == '__main__':
    file_path = "D:/Research/RQ2/classifier/data.json"
    main(file_path)
