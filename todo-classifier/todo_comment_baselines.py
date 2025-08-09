# import os
# import sys

# current_dir = os.getcwd()
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# sys.path.insert(0, parent_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from ml_classifiers import *
import logger
import glo
warnings.filterwarnings("ignore")


def load_all_data(file_path):
    # data = pd.read_json(file_path, orient='records', lines=True)
    # data = data.iloc[:758, :31]

    data = pd.read_csv(file_path)
    data = data.iloc[:758, :31]

    return data

def tmp_label_process(labeled_data):
    high_low_labels = labeled_data['Quality'].apply(lambda x:1 if x == 1 else 0)
    return high_low_labels


def preprocessing(train_data, test_data, categorical_columns):
    train_processed = train_data.copy()
    test_processed = test_data.copy()
    
    for col in categorical_columns:
        train_processed[col] = train_processed[col].fillna("unknown").astype(str)
        test_processed[col] = test_processed[col].fillna("unknown").astype(str)

    label_encoders = {}
    for column in categorical_columns:
        if column in train_processed.columns:
            le = LabelEncoder()
            le.fit(train_processed[column])
            label_encoders[column] = le

            train_processed[column] = le.transform(train_processed[column])
            
            test_processed[column] = test_processed[column].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else len(le.classes_)
            )
    
    num_cols = train_processed.select_dtypes(include=['number']).columns
    train_processed[num_cols] = train_processed[num_cols].fillna(train_processed[num_cols].median())
    test_processed[num_cols] = test_processed[num_cols].fillna(train_processed[num_cols].median())
    
    return train_processed, test_processed


def main(file_path):
    mylogger = logger.mylog("ml")
    labeled_data = load_all_data(file_path)
    high_low_labels = tmp_label_process(labeled_data)
    labeled_data = labeled_data.drop(columns=['Quality', 'modelId'])
    pos_num = np.sum(high_low_labels == 1)
    print(pos_num)
    glo._init()

    # Categorical feature
    categorical_columns = ['what-license', 'what-library', 'what-task']

    # kf = KFold(n_splits=10, random_state=3408, shuffle=True)
    kf = StratifiedKFold(n_splits=10, random_state=3408, shuffle=True)

    for train_index, test_index in kf.split(labeled_data, high_low_labels):
        
        train_data_raw = labeled_data.iloc[train_index]
        test_data_raw = labeled_data.iloc[test_index]
        train_data_processed, test_data_processed = preprocessing(
            train_data_raw, test_data_raw, categorical_columns
        )
        train_x = np.array(train_data_processed)
        test_x = np.array(test_data_processed)
        train_y = np.array(high_low_labels.iloc[train_index])
        test_y = np.array(high_low_labels.iloc[test_index])

        # train_x, train_y = np.array(labeled_data.iloc[train_index]), np.array(high_low_labels.iloc[train_index])
        # test_x, test_y = np.array(labeled_data.iloc[test_index]), np.array(high_low_labels.iloc[test_index])
        
        mylogger.info("==================modelcard high_low classifers=============")
        methods_container(train_x, train_y, test_x, test_y, mylogger, "score")


if __name__ == '__main__':
    # file_path = "what-makes-a-good-TODO-comment/package/data.json"
    # file_path = "D:/Research/RQ2/classifier/data.json"
    file_path = "D:/Research/RQ2/classifier/data(test).csv"
    main(file_path)
