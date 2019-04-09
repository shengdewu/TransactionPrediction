from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
import eli5
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(X, X_test, y, params, folds, model_type='lgb', plot_feature_importance=True, averaging='usual',
                model=None):
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        X_train, X_valid = X.loc[train_index], X.loc[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        if model_type == 'lgb':
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)

            model = lgb.train(params,
                              train_data,
                              num_boost_round=20000,
                              valid_sets=[train_data, valid_data],
                              verbose_eval=1000,
                              early_stopping_rounds=200)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)


        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            y_pred_valid = model.predict_proba(X_valid).reshape(-1, )
            score = roc_auc_score(y_valid, y_pred_valid)
            # print(f'Fold {fold_n}. AUC: {score:.4f}.')
            # print('')

            y_pred = model.predict_proba(X_test)[:, 1]


        oof[valid_index] = y_pred_valid.reshape(-1, )
        scores.append(roc_auc_score(y_valid, y_pred_valid))

        if averaging == 'usual':
            prediction += y_pred
        elif averaging == 'rank':
            prediction += pd.Series(y_pred).rank().values

        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= 4

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    if model_type == 'lgb':
        feature_importance["importance"] /= 4
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            plt.show()
            return oof, prediction, feature_importance
        return oof, prediction, scores

    else:
        return oof, prediction, scores

class mLearn(object):

    def __init__(self):
        return


    def execute(self):
        train = pd.read_csv('data/train.csv')
        test = pd.read_csv('data/test.csv')

        trainData = train.drop(['ID_code', 'target'], axis=1)
        testData = test.drop(['ID_code'], axis=1)
        targetData = train['target']

        kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=None)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }

        #train_model(trainData, testData, targetData, params, kfold)
        preValue = np.zeros(len(testData))
        preScore = []
        featureImportance = pd.DataFrame()
        for nFold,(trainIndex, validIndex) in enumerate(kfold.split(trainData, targetData)):
            xTrain = trainData.loc[trainIndex]; xValid = trainData.loc[validIndex]
            yTrain = targetData[trainIndex]; yValid = targetData[validIndex]

            train = lgb.Dataset(xTrain, yTrain)
            valid = lgb.Dataset(xValid, yValid)

            gbm = lgb.train(params=params, train_set=train, valid_sets=[train,valid], num_boost_round=20000, early_stopping_rounds=5)

            yPreValid = gbm.predict(data=xValid)
            preScore.append(roc_auc_score(yValid, yPreValid))
            importance = pd.DataFrame()
            importance['importance'] = gbm.feature_importance()
            importance['kfold'] = nFold
            importance['feature'] = trainData.columns
            feature_importance = pd.concat([featureImportance, importance], axis=0)
            value = gbm.predict(data=testData, num_iteration=gbm.best_iteration)
            preValue += value
        return

