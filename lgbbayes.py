import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.utils.fixes import logsumexp

class LGBBayes(object):
    def __init__(self):
        self.__param__ = {
            'boosting':'gbdt',
            'objective': 'binary',
            'num_leaves': 3,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 2,
            'verbose': 0,
        }
        self.clf = []
        self.one = 0
        self.zero = 0
        self.total = 0
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

    def fit(self, trainData, optRound):
        self.clf.clear()
        for i in range(200):
            self.__param__['n_estimators'] = optRound['var_'+str(i)]
            lgbClass = lgb.LGBMClassifier(**self.__param__)
            train = trainData[['var_'+str(i)]].to_numpy()
            y = trainData['target']
            lgbFit = lgbClass.fit(X=train, y=y)
            self.clf.append(lgbFit)
        return

    def preProba(self, testData):
        logSum = np.zeros(shape=(testData.shape[0], 2))
        for i in range(200):
            clf = self.clf[i]
            prob = clf.predict_proba(testData[['var_'+str(i)]])
            logSum += np.log(prob)

        logSum += np.array([np.log(self.zero) - np.log(self.total), np.log(self.one) - np.log(self.total)])
        log_prob_x = logsumexp(logSum, axis=1)
        return np.exp(logSum - np.atleast_2d(log_prob_x).T)


    def execute(self, nflod):
        train = pd.read_csv('data/train.csv').drop(['ID_code'], axis=1)

        feature = train.columns[1:]

        total = len(train.index)
        oneSum = train[train['target'] == 1]['target'].count()
        zeroSum = train[train['target'] == 0]['target'].count()
        self.one = oneSum
        self.zero = zeroSum
        self.total = total

        optRound = {'var_0': 324, 'var_1': 414, 'var_2': 330, 'var_3': 322, 'var_4': 126, 'var_5': 362, 'var_6': 322, 'var_7': 1, 'var_8': 170, 'var_9': 592, 'var_10': 98, 'var_11': 228, 'var_12': 440, 'var_13': 412, 'var_14': 122, 'var_15': 212, 'var_16': 136, 'var_17': 2, 'var_18': 264, 'var_19': 223, 'var_20': 284, 'var_21': 452, 'var_22': 344, 'var_23': 208, 'var_24': 310, 'var_25': 194, 'var_26': 346, 'var_27': 1, 'var_28': 214, 'var_29': 6, 'var_30': 1, 'var_31': 394, 'var_32': 366, 'var_33': 318, 'var_34': 342, 'var_35': 380, 'var_36': 386, 'var_37': 140, 'var_38': 1, 'var_39': 94, 'var_40': 368, 'var_41': 50, 'var_42': 82, 'var_43': 222, 'var_44': 290, 'var_45': 154, 'var_46': 34, 'var_47': 84, 'var_48': 341, 'var_49': 248, 'var_50': 256, 'var_51': 430, 'var_52': 272, 'var_53': 368, 'var_54': 255, 'var_55': 204, 'var_56': 344, 'var_57': 154, 'var_58': 198, 'var_59': 138, 'var_60': 110, 'var_61': 154, 'var_62': 228, 'var_63': 246, 'var_64': 274, 'var_65': 94, 'var_66': 186, 'var_67': 300, 'var_68': 164, 'var_69': 126, 'var_70': 243, 'var_71': 274, 'var_72': 146, 'var_73': 56, 'var_74': 262, 'var_75': 400, 'var_76': 348, 'var_77': 180, 'var_78': 346, 'var_79': 90, 'var_80': 698, 'var_81': 382, 'var_82': 224, 'var_83': 210, 'var_84': 132, 'var_85': 224, 'var_86': 248, 'var_87': 346, 'var_88': 230, 'var_89': 276, 'var_90': 254, 'var_91': 412, 'var_92': 316, 'var_93': 298, 'var_94': 814, 'var_95': 264, 'var_96': 20, 'var_97': 186, 'var_98': 46, 'var_99': 310, 'var_100': 2, 'var_101': 282, 'var_102': 280, 'var_103': 2, 'var_104': 372, 'var_105': 189, 'var_106': 270, 'var_107': 379, 'var_108': 1212, 'var_109': 322, 'var_110': 368, 'var_111': 256, 'var_112': 550, 'var_113': 180, 'var_114': 252, 'var_115': 312, 'var_116': 228, 'var_117': 76, 'var_118': 320, 'var_119': 296, 'var_120': 242, 'var_121': 306, 'var_122': 304, 'var_123': 610, 'var_124': 52, 'var_125': 272, 'var_126': 46, 'var_127': 526, 'var_128': 282, 'var_129': 126, 'var_130': 296, 'var_131': 337, 'var_132': 242, 'var_133': 346, 'var_134': 278, 'var_135': 252, 'var_136': 1, 'var_137': 306, 'var_138': 390, 'var_139': 474, 'var_140': 162, 'var_141': 476, 'var_142': 292, 'var_143': 154, 'var_144': 262, 'var_145': 254, 'var_146': 493, 'var_147': 294, 'var_148': 312, 'var_149': 388, 'var_150': 378, 'var_151': 318, 'var_152': 104, 'var_153': 72, 'var_154': 412, 'var_155': 258, 'var_156': 206, 'var_157': 232, 'var_158': 8, 'var_159': 146, 'var_160': 112, 'var_161': 22, 'var_162': 266, 'var_163': 228, 'var_164': 342, 'var_165': 322, 'var_166': 380, 'var_167': 306, 'var_168': 196, 'var_169': 408, 'var_170': 414, 'var_171': 136, 'var_172': 354, 'var_173': 356, 'var_174': 346, 'var_175': 272, 'var_176': 102, 'var_177': 540, 'var_178': 274, 'var_179': 308, 'var_180': 380, 'var_181': 308, 'var_182': 28, 'var_183': 34, 'var_184': 306, 'var_185': 44, 'var_186': 362, 'var_187': 185, 'var_188': 362, 'var_189': 90, 'var_190': 448, 'var_191': 424, 'var_192': 424, 'var_193': 162, 'var_194': 180, 'var_195': 296, 'var_196': 370, 'var_197': 292, 'var_198': 320, 'var_199': 324}#self.optRound(train)
        kfold = StratifiedKFold(n_splits=nflod, shuffle=True)
        score = []
        for ti, vi in kfold.split(train[feature], train['target']):
            trainData = train.loc[ti]
            validData = train.loc[vi]
            yvalid = validData['target']
            self.fit(trainData, optRound)
            ypre = self.preProba(validData)[:,1]
            score.append(roc_auc_score(yvalid, ypre))

        print('mean {} std {}\n'.format(np.mean(score), np.std(score)))
        test = pd.read_csv('data/test.csv').drop(['ID_code'], axis=1)
        self.fit(train, optRound)
        pre = self.preProba(test)[:,1]

        return


