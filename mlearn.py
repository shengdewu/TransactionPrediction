from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import eli5
import lightgbm as lgb

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
        for nFold,(trainIndex, validIndex) in enumerate(kfold.split(trainData, targetData)):
            xTrain = trainData.loc[trainIndex]; xValid = trainData.loc[validIndex]
            yTrain = targetData[trainIndex]; yValid = targetData[validIndex]

            train = lgb.Dataset(xTrain, yTrain)
            valid = lgb.Dataset(xValid, yValid)

            gbm = lgb.train(params=params, train_set=train, valid_sets=[train,valid], num_boost_round=20000, early_stopping_rounds=5)

            yPreValid = gbm.predict(data=xValid)

            pre = gbm.predict(data=testData, num_iteration=gbm.best_iteration)


            socre = roc_auc_score(yPreValid, yValid)

            print(pre)
        return

