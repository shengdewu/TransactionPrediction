import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

class LGBBayes(object):
    def __init__(self):
        self.__param__ = {
            'boosting':'gbdt',
            'objective': 'binary',
            'metric': {'l2', 'l1'},
            'num_leaves': 3,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 2,
            'verbose': 0,
        }
        self.clf = []
        return

    def optRound(self, data):
        index = 0
        featurePre = 'var_'
        label = data['target']
        oround = {}
        for i in range(200):
            dataSet = lgb.Dataset(data=data[['var_'+str(i)]], label=label)
            eval = lgb.cv(params=self.__param__,train_set=dataSet,
                          metrics='binary_logloss',
                          num_boost_round=10000,
                          nfold=3,
                          early_stopping_rounds=200,
                          verbose_eval=100)
            oround['var_'+str(i)] = len(eval['binary_logloss-mean'])
        return oround

    def fit(self, trainData):
        optRound = self.optRound(trainData)

        self.clf.clear()
        for i in range(200):
            self.__param__['n_estimators'] = optRound[i]
            lgbClass = lgb.LGBMClassifier(**self.__param__)
            self.clf.append(lgbClass.fit(X=trainData['var_'+str(i)], y=trainData['target']))
        return

    def preProba(self, testData):
        logSum = np.zeros(shape=(testData.shape[0], 2))
        for i in range(200):
            clf = self.clf[i]
            prob = clf.predict_proba(testData[['var_'+str(i)]])
            logSum += np.log(prob)

        return

    def execute(self, nflod):
        train = pd.read_csv('data/train.csv').drop(['ID_code'], axis=1)

        feature = train.columns[1:]

        total = len(train.index)
        t1 = train['target' == 1]
        oneSum = np.sum(train.loc['target' == 1])
        zeroSum = np.sum(train.loc['target' == 0])


        kfold = StratifiedKFold(n_splits=nflod, shuffle=True)
        for ti, vi in kfold.split(train[feature], train['target']):
            trainData = train.loc[ti]
            validData = train.loc[vi]
            yvalid = validData['target']
            self.fit(trainData)
            ypre = self.preProba(validData)
            roc_auc_score(yvalid, ypre)

        test = pd.read_csv('data/test.csv').drop(['ID_code'], axis=1)

        return


