from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
import eli5
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import math

class mLearn(object):

    def __init__(self):
        return

    def predict(self, trainData, targetData,testData, knFold=4):

        kfold = StratifiedKFold(n_splits=knFold, shuffle=True, random_state=None)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'l2', 'l1'},
            'num_leaves': 3,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 2,
            'verbose': 0
        }

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
            featureImportance = pd.concat([featureImportance, importance], axis=0)
            value = gbm.predict(data=testData, num_iteration=gbm.best_iteration)
            preValue += value

        preValue /= knFold

        return preValue, featureImportance, preScore

    def execute(self, nflod):
        train = pd.read_csv('data/train.csv')
        test = pd.read_csv('data/test.csv')

        trainData = train.drop(['ID_code', 'target'], axis=1)
        testData = test.drop(['ID_code'], axis=1)
        targetData = train['target']

        preValue, featureImportance, preScore= self.predict(trainData,targetData, testData, nflod)
        importance = featureImportance.loc[:,['feature', 'importance']].groupby(['feature']).mean()
        importance['feature'] = importance.index
        plt.figure(figsize=(16,12))
        sns.barplot(x='importance',y='feature',data=importance.sort_values(by='importance',ascending=False)[:50])
        plt.waitforbuttonpress()

        sub = pd.read_csv('./data/sample_submission.csv')
        sub['target'] = preValue
        sub.to_csv('lgb.csv',index=False)

        print(str('auc socre mean {} std{}').format(np.mean(preScore), np.std(preScore)))
        return

