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
    
    label_encoders = {}
    for column in categorical_columns:
        if column in train_processed.columns:
            label_encoders[column] = LabelEncoder()
            train_processed[column] = label_encoders[column].fit_transform(
                train_processed[column].astype(str)
            )
            
            test_categories = test_processed[column].astype(str)
            processed_test = []
            
            for cat in test_categories:
                if cat in label_encoders[column].classes_:
                    processed_test.append(label_encoders[column].transform([cat])[0])
                else:
                    processed_test.append(-1)
            
            test_processed[column] = processed_test
    
    train_processed = train_processed.fillna(-1)
    test_processed = test_processed.fillna(-1)
    
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

    # kf = KFold(n_splits=10, random_state=3407, shuffle=True)
    kf = StratifiedKFold(n_splits=10, random_state=3407, shuffle=True)

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
