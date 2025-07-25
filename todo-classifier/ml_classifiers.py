import datetime
import warnings
from collections import Counter

import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, roc_curve, auc, \
    roc_auc_score
import glo

SCORING = {'accuracy': 'accuracy', 'precision': make_scorer(precision_score), 'recall': make_scorer(recall_score),
           'f1': make_scorer(f1_score),
           'AUC': make_scorer(roc_auc_score)}
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def calculate_scores(logger, method_name, method_info):
    tmp_key = method_info + "_" + method_name
    temp_predict = glo.get_val(tmp_key)
    res = "\n flod " + str(temp_predict[6]) + "\tAccuaryMean: " + str(temp_predict[5] / temp_predict[6]) \
          + "\tWeightedPrecisionMean: " + str(temp_predict[2] / temp_predict[6]) + "\tWeightedRecallMean: " \
          + str(temp_predict[3] / temp_predict[6]) + "\t WeightedF1Mean： " + str(temp_predict[4] / temp_predict[6]) \
          + "\tPrecisonMean: " + str(temp_predict[0] / temp_predict[6]) + "\t NegativePrecisionMean： " + str(
        temp_predict[1] / temp_predict[6]) \
          + "\tRecallMean: " + str(temp_predict[7] / temp_predict[6]) + "\t NegativeRecallMean： " + str(temp_predict[8] / temp_predict[6]) \
          + "\tF1Mean: " + str(temp_predict[9] / temp_predict[6]) + "\t NegativeF1Mean： " + str(temp_predict[10] / temp_predict[6])
    print(res)
    logger.info(res)


def evaluate_method(model, method_name, logger, test_x, test_y, method_info):
    logger.info("test data distrubution: " + str(sorted(Counter(test_y).items())))
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
    fpr, tpr, thresholds = roc_curve(test_y, y_score, pos_label=0)
    aucArea = auc(fpr, tpr)
    aucScore = roc_auc_score(y_true=test_y, y_score=y_score)
    
    tmp_key = method_info + "_" + method_name
    if glo.get_val(tmp_key) is None:
        glo.set_val(tmp_key, [precision_res[1], precision_res[0], weighted_precision, weighted_recall,
                                      weighted_f1, acc_res, 1, recall_res[1], recall_res[0], f1_res[0],
                                      f1_res[1]])
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
        temp_predict[9] += f1_res[1]
        temp_predict[10] += f1_res[0]
        glo.set_val(tmp_key, temp_predict)


def lr_classifier(train_x, train_y, logger):
    parameters = {'C': np.linspace(0.0001, 20, 20),
                  'solver': ["newton-cg", "lbfgs", "liblinear", "sag"],
                  'multi_class': ['ovr'],
                  'dual': [False],
                  'verbose': [False],
                  'max_iter': [500]
                  }
    model_g = LogisticRegression()
    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=25)
    grid.fit(train_x, train_y)
    logger.info("lr_ para: %s " % (grid.best_params_))
    best_para = grid.best_params_
    model_lr = LogisticRegression(C=best_para['C'], random_state=3407,
                               solver=best_para['solver'], multi_class='ovr', dual=False, verbose=False,
                               max_iter=500)
    model = model_lr.fit(train_x, train_y)
    return model


def rf_classifier(train_x, train_y, logger):
    parameters = {
                  'n_estimators': list(range(10, 110, 10))
                  }
    model_g = RandomForestClassifier()
    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=25)
    grid.fit(train_x, train_y)
    logger.info("rf_ para: %s " % (grid.best_params_))
    best_para = grid.best_params_
    model_rf = RandomForestClassifier(n_estimators=best_para['n_estimators'], random_state=3407)
    model = model_rf.fit(train_x, train_y)
    return model


def knn_classifier(train_x, train_y, logger):
    parameters = {
                'n_neighbors': np.arange(1, 11, 1),
                'algorithm': ['auto']
                  }
    model_g = KNeighborsClassifier()
    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=25)
    grid.fit(train_x, train_y)
    logger.info("knn_ para: %s " % (grid.best_params_))
    best_para = grid.best_params_
    model_knn = KNeighborsClassifier(n_neighbors=best_para['n_neighbors'], algorithm='auto')
    model = model_knn.fit(train_x, train_y)
    return model


def gb_classifier(train_x, train_y, logger):
    parameters = {"loss": ["deviance"],
                 "learning_rate": [0.1],
                 "min_samples_split": np.linspace(0.1, 0.4, 5),
                 "min_samples_leaf": np.linspace(0.1, 0.3, 5),
                 "max_depth": [3, 8, 16],
                 "max_features": ["auto"],
                 "criterion": ["friedman_mse"],
                 "subsample": [0.95],
                 "n_estimators": [100]
                 }
    model_g = GradientBoostingClassifier()
    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=25)
    grid.fit(train_x, train_y)
    logger.info("gb_ para: %s " % (grid.best_params_))
    best_para = grid.best_params_
    model_gb = GradientBoostingClassifier(loss="deviance",
                                       learning_rate=best_para['learning_rate'],
                                       min_samples_split=best_para['min_samples_split'],
                                       min_samples_leaf=best_para['min_samples_leaf'],
                                       max_depth=best_para['max_depth'],
                                       max_features=best_para['max_features'],
                                       criterion=best_para['criterion'],
                                       subsample=best_para['subsample'],
                                       n_estimators=10)
    model = model_gb.fit(train_x, train_y)
    return model


def mlp_classifier(train_x, train_y, logger):
    parameters = {
                "hidden_layer_sizes": [(100,), (100, 50), (100, 100), (100, 75), (100, 25)],
                  "activation": ["logistic"],
                  "solver": ['adam']
                  }
    model_g = MLPClassifier()
    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=25)
    grid.fit(train_x, train_y)
    logger.info("mlp_ para: %s " % (grid.best_params_))
    best_para = grid.best_params_
    model_mlp = MLPClassifier(hidden_layer_sizes=best_para['hidden_layer_sizes'],
                          activation=best_para['activation'],
                          solver=best_para['solver'],
                          verbose=False)
    model = model_mlp.fit(train_x, train_y)
    return model

def dt_classifier(train_x, train_y, logger):
    parameters = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model_g = DecisionTreeClassifier()
    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=25)
    grid.fit(train_x, train_y)
    logger.info("dt_ para: %s " % (grid.best_params_))
    best_para = grid.best_params_
    model_dt = DecisionTreeClassifier(
        max_depth=best_para['max_depth'],
        min_samples_split=best_para['min_samples_split'],
        min_samples_leaf=best_para['min_samples_leaf'],
        random_state=3407
    )
    model = model_dt.fit(train_x, train_y)
    return model

def svm_classifier(train_x, train_y, logger):
    parameters = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 1]
    }
    model_g = SVC(probability=True)
    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=25)
    grid.fit(train_x, train_y)
    logger.info("svm_ para: %s " % (grid.best_params_))
    best_para = grid.best_params_
    model_svm = SVC(
        C=best_para['C'],
        kernel=best_para['kernel'],
        gamma=best_para['gamma'],
        probability=True,
        random_state=3407
    )
    model = model_svm.fit(train_x, train_y)
    return model

def xgb_classifier(train_x, train_y, logger):
    parameters = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    model_g = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=25)
    grid.fit(train_x, train_y)
    logger.info("xgb_ para: %s " % (grid.best_params_))
    best_para = grid.best_params_
    model_xgb = XGBClassifier(
        n_estimators=best_para['n_estimators'],
        max_depth=best_para['max_depth'],
        learning_rate=best_para['learning_rate'],
        subsample=best_para['subsample'],
        colsample_bytree=best_para['colsample_bytree'],
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=3407
    )
    model = model_xgb.fit(train_x, train_y)
    return model

def min_max_normalization(np_array):
    min_max_scaler = preprocessing.MinMaxScaler()
    res = min_max_scaler.fit_transform(np_array)
    return res


def methods_container(train_x, train_y, test_x, test_y, logger, class_type):
    # dump the model?
    # Logistic Regression, Random Forest, Gradient Boosting, KNN, MLP

    logger.info("==================MultinomialNB==============")
    model_g = MultinomialNB()
    mnb_model = model_g.fit(min_max_normalization(train_x), train_y)
    evaluate_method(mnb_model, "mnb", logger, min_max_normalization(test_x), test_y, class_type)
    calculate_scores(logger, "mnb", class_type)
    
    logger.info("==================Logistic Regression==============")
    lr_model = lr_classifier(train_x, train_y, logger)
    evaluate_method(lr_model, "lr", logger, test_x, test_y, class_type)
    calculate_scores(logger, "lr", class_type)

    logger.info("==================Random Forest==============")
    rf_model = rf_classifier(train_x, train_y, logger)
    evaluate_method(rf_model, "rf", logger, test_x, test_y, class_type)
    calculate_scores(logger, "rf", class_type)

    logger.info("==================KNN================")
    knn_model = knn_classifier(train_x, train_y, logger)
    evaluate_method(knn_model, "knn", logger, test_x, test_y, class_type)
    calculate_scores(logger, "knn", class_type)

    logger.info("==================Boosting================")
    gb_model = gb_classifier(train_x, train_y, logger)
    evaluate_method(gb_model, "gb", logger, test_x, test_y, class_type)
    calculate_scores(logger, "gb", class_type)

    logger.info("==================MLP================")
    mlp_model = mlp_classifier(train_x, train_y, logger)
    evaluate_method(mlp_model, "mlp", logger, test_x, test_y, class_type)
    calculate_scores(logger, "mlp", class_type)
    
    logger.info("==================Decision Tree================")
    dt_model = dt_classifier(train_x, train_y, logger)
    evaluate_method(dt_model, "dt", logger, test_x, test_y, class_type)
    calculate_scores(logger, "dt", class_type)
    
    logger.info("==================Support Vector Machine================")
    svm_model = svm_classifier(train_x, train_y, logger)
    evaluate_method(svm_model, "svm", logger, test_x, test_y, class_type)
    calculate_scores(logger, "svm", class_type)
    
    logger.info("==================XGBoost================")
    xgb_model = xgb_classifier(train_x, train_y, logger)
    evaluate_method(xgb_model, "xgb", logger, test_x, test_y, class_type)
    calculate_scores(logger, "xgb", class_type)

    logger.info("================================DummyClassifier==============================")
    model_g = DummyClassifier(strategy="stratified")
    dummy_model = model_g.fit(train_x, train_y)
    evaluate_method(dummy_model, "dummy", logger, test_x, test_y, class_type)
    calculate_scores(logger, "dummy", class_type)