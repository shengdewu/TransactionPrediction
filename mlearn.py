from sklearn.model_selection import StratifiedKFold
import eli5
import  lightgbm as lgb

class mLearn(object):

    def __init__(self):
        return


    def execute(self, data):

        trainData = None
        testData = None
        targetData = None
        kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=None)

        for nFold,(trainIndex, validIndex) in enumerate(kfold.split(trainData, targetData)):
            xTrain = trainData.loc[trainIndex]; xValid = trainData.loc[validIndex]
            yTrain = targetData[trainIndex]; yValid = targetData[validIndex]

            train = lgb.Dataset(xTrain, yTrain)
            valid = lgb.Dataset(xValid, yValid)

        return

