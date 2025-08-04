import datetime
import warnings
from collections import Counter

import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, roc_curve, auc, \
    roc_auc_score
import glo

from imbens.ensemble import SelfPacedEnsembleClassifier

SCORING = {'accuracy': 'accuracy', 'precision': make_scorer(precision_score), 'recall': make_scorer(recall_score),
           'f1': make_scorer(f1_score),
           'AUC': make_scorer(roc_auc_score)}
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def calculate_scores(logger, method_name, method_info):
    tmp_key = method_info + "_" + method_name
    temp_predict = glo.get_val(tmp_key)
    res = "\n fold " + str(temp_predict[6]) + "\tAccuracyMean: " + str(temp_predict[5] / temp_predict[6]) \
          + "\tWeightedPrecisionMean: " + str(temp_predict[2] / temp_predict[6]) + "\tWeightedRecallMean: " \
          + str(temp_predict[3] / temp_predict[6]) + "\t WeightedF1Mean: " + str(temp_predict[4] / temp_predict[6]) \
          + "\tPrecisionMean: " + str(temp_predict[0] / temp_predict[6]) + "\t NegativePrecisionMean: " + str(
        temp_predict[1] / temp_predict[6]) \
          + "\tRecallMean: " + str(temp_predict[7] / temp_predict[6]) + "\t NegativeRecallMean: " + str(temp_predict[8] / temp_predict[6]) \
          + "\tNegativeF1Mean: " + str(temp_predict[9] / temp_predict[6]) + "\t F1Mean: " + str(temp_predict[10] / temp_predict[6]) \
          + "\tAUCMean: " + str(temp_predict[11] / temp_predict[6])
    print(res)
    logger.info(res)


def evaluate_method(model, method_name, logger, test_x, test_y, method_info):
    logger.info("test data distribution: " + str(sorted(Counter(test_y).items())))

    # # 假设 test_x 是你的测试集
    # y_proba = model.predict_proba(test_x)[:, 1]  # 取正类（类别1）的概率

    # # 手动设置阈值
    # threshold = 0.13  # 举例：你想更敏感，降低阈值以提高召回率

    # # 手动判断：概率 >= 阈值 → 预测为 1，否则为 0
    # y_pred = (y_proba >= threshold).astype(int)

    y_pred = model.predict(test_x)
    y_score = model.predict_proba(test_x)[:, 1]

    acc_res = accuracy_score(test_y, y_pred)
    tn, fp, fn, tp = confusion_matrix(test_y, y_pred).ravel()
    recall_res = recall_score(test_y, y_pred, average=None)
    precision_res = precision_score(test_y, y_pred, average=None)
    f1_res = f1_score(test_y, y_pred, average=None)
    weighted_precision = precision_score(test_y, y_pred, average="weighted")
    weighted_recall = recall_score(test_y, y_pred, average="weighted")
    weighted_f1 = f1_score(test_y, y_pred, average="weighted")
    # AUC
    aucScore = roc_auc_score(y_true=test_y, y_score=y_score)
    
    tmp_key = method_info + "_" + method_name
    if glo.get_val(tmp_key) is None:
        glo.set_val(tmp_key, [precision_res[1], precision_res[0], weighted_precision, weighted_recall,
                                      weighted_f1, acc_res, 1, recall_res[1], recall_res[0], f1_res[0],
                                      f1_res[1], aucScore])
    else:
        temp_predict = glo.get_val(tmp_key)
        temp_predict[0] += precision_res[1]
        temp_predict[1] += precision_res[0]
        temp_predict[2] += weighted_precision
        temp_predict[3] += weighted_recall
        temp_predict[4] += weighted_f1
        temp_predict[5] += acc_res
        temp_predict[6] += 1
        temp_predict[7] += recall_res[1]
        temp_predict[8] += recall_res[0]
        temp_predict[9] += f1_res[0]
        temp_predict[10] += f1_res[1]
        temp_predict[11] += aucScore
        glo.set_val(tmp_key, temp_predict)


# Logistic Regression
def lr_classifier(train_x, train_y, logger):

    # 参数搜索
    '''
    parameters = {
        'C': np.linspace(0.0001, 20, 20),
        'solver': ["newton-cg", "lbfgs", "liblinear", "sag"],
        'dual': [False],
        'verbose': [False],
        'max_iter': [500]
    }
    model_g = LogisticRegression(random_state=3407)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("lr_para: %s", grid.best_params_)
    best_model = grid.best_estimator_

    return best_model
    '''

    model_g = LogisticRegression(random_state=3407)
    clf = SelfPacedEnsembleClassifier(random_state=3407, estimator=model_g)
    clf.fit(train_x, train_y)

    return clf


# Random Forest
def rf_classifier(train_x, train_y, logger):
    '''
    parameters = {
        'n_estimators': list(range(10, 110, 10))
    }
    model_g = RandomForestClassifier(random_state=3407)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("rf_para: %s" % (grid.best_params_))
    best_model = grid.best_estimator_
    return best_model
    '''

    '''
    # 调参
    parameters = {
        'n_estimators': [10, 20, 30, 50, 100],
        'estimator__n_estimators': list(range(10, 110, 10)),      # RandomForest树的数量
    }
    model_g = RandomForestClassifier(random_state=3407)
    clf = SelfPacedEnsembleClassifier(random_state=3407, estimator=model_g)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=5, random_state=5, shuffle=True)
    grid = GridSearchCV(clf, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("rf_para: %s" % (grid.best_params_))
    best_model = grid.best_estimator_
    return best_model
    '''

    model_g = RandomForestClassifier(random_state=3407)
    clf = SelfPacedEnsembleClassifier(random_state=3407, estimator=model_g, n_estimators=60)
    # clf = RandomForestClassifier(random_state=3407)
    clf.fit(train_x, train_y)

    return clf


# KNN
def knn_classifier(train_x, train_y, logger):
    # parameters = {
    #     'n_neighbors': np.arange(1, 11, 1),
    #     'algorithm': ['auto']
    # }
    # model_g = KNeighborsClassifier()
    # # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    # fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    # grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    # grid.fit(train_x, train_y)
    # logger.info("knn_para: %s" % (grid.best_params_))
    # best_model = grid.best_estimator_
    # return best_model

    model_g = KNeighborsClassifier()
    clf = SelfPacedEnsembleClassifier(random_state=3407, estimator=model_g)
    clf.fit(train_x, train_y)

    return clf

# Gradient Boosting
def gb_classifier(train_x, train_y, logger):
    parameters = {"loss": ["log_loss"],
                 "learning_rate": [0.1],
                 "min_samples_split": np.linspace(0.1, 0.4, 3),
                 "min_samples_leaf": np.linspace(0.1, 0.3, 3),
                 "max_depth": [3, 8],
                 "max_features": ["sqrt"],
                 "criterion": ["friedman_mse"],
                 "subsample": [0.95],
                 "n_estimators": [50, 100, 150]
                 }
    model_g = GradientBoostingClassifier(random_state=3407)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("gb_para: %s" % (grid.best_params_))
    best_model = grid.best_estimator_
    return best_model


# MLP
def mlp_classifier(train_x, train_y, logger):
    parameters = {
        "hidden_layer_sizes": [(50,), (50, 25), (25,)],
        "activation": ["relu", "logistic"],
        "solver": ['adam']
    }
    model_g = MLPClassifier(random_state=3407)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("mlp_para: %s" % (grid.best_params_))
    best_model = grid.best_estimator_
    return best_model


# Decision Tree
def dt_classifier(train_x, train_y, logger):

    '''
    parameters = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [5, 10, 15]
    }
    model_g = DecisionTreeClassifier(random_state=3407)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("dt_para: %s" % (grid.best_params_))
    best_model = grid.best_estimator_
    return best_model
    '''

    model_g = DecisionTreeClassifier(random_state=3407)
    clf = SelfPacedEnsembleClassifier(random_state=3407, estimator=model_g)
    clf.fit(train_x, train_y)

    return clf

# Support Vector Machine
def svm_classifier(train_x, train_y, logger):
    parameters = {
        'C': [0.1, 1, 10, 50],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 1]
    }
    model_g = SVC(random_state=3407, probability=True)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("svm_para: %s" % (grid.best_params_))
    best_model = grid.best_estimator_
    return best_model


# XGBoost
def xgb_classifier(train_x, train_y, logger):
    parameters = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.9],
        'colsample_bytree': [0.9]
    }
    model_g = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=3407)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("xgb_para: %s" % (grid.best_params_))
    best_model = grid.best_estimator_
    return best_model


def methods_container(train_x, train_y, test_x, test_y, logger, class_type):

    # Standardization
    scaler1 = StandardScaler()
    train_x_scaled1 = scaler1.fit_transform(train_x)
    test_x_scaled1 = scaler1.transform(test_x)

    # Normalization
    scaler2 = MinMaxScaler()
    train_x_scaled2 = scaler2.fit_transform(train_x)
    test_x_scaled2 = scaler2.transform(test_x)

    # logger.info("==================Logistic Regression==============")
    # lr_model = lr_classifier(train_x_scaled1, train_y, logger)
    # evaluate_method(lr_model, "lr", logger, test_x_scaled1, test_y, class_type)
    # calculate_scores(logger, "lr", class_type)

    logger.info("==================Random Forest==============")
    rf_model = rf_classifier(train_x, train_y, logger)
    evaluate_method(rf_model, "rf", logger, test_x, test_y, class_type)
    calculate_scores(logger, "rf", class_type)

    # logger.info("==================KNN================")
    # knn_model = knn_classifier(train_x_scaled2, train_y, logger)
    # evaluate_method(knn_model, "knn", logger, test_x_scaled2, test_y, class_type)
    # calculate_scores(logger, "knn", class_type)

    # logger.info("==================Gradient Boosting================")
    # gb_model = gb_classifier(train_x, train_y, logger)
    # evaluate_method(gb_model, "gb", logger, test_x, test_y, class_type)
    # calculate_scores(logger, "gb", class_type)

    # logger.info("==================MLP================")
    # mlp_model = mlp_classifier(train_x, train_y, logger)
    # evaluate_method(mlp_model, "mlp", logger, test_x, test_y, class_type)
    # calculate_scores(logger, "mlp", class_type)
    
    # logger.info("==================Decision Tree================")
    # dt_model = dt_classifier(train_x, train_y, logger)
    # evaluate_method(dt_model, "dt", logger, test_x, test_y, class_type)
    # calculate_scores(logger, "dt", class_type)
    
    # logger.info("==================Support Vector Machine================")
    # svm_model = svm_classifier(train_x, train_y, logger)
    # evaluate_method(svm_model, "svm", logger, test_x, test_y, class_type)
    # calculate_scores(logger, "svm", class_type)
    
    # logger.info("==================XGBoost================")
    # xgb_model = xgb_classifier(train_x, train_y, logger)
    # evaluate_method(xgb_model, "xgb", logger, test_x, test_y, class_type)
    # calculate_scores(logger, "xgb", class_type)