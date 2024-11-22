import numpy as np
from scipy import stats
from sklearn.svm import SVR
from numba import jit
# find the best parameter optimised
from sklearn.model_selection import ShuffleSplit,KFold,GridSearchCV,RandomizedSearchCV,train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import utils.util_doc.utils_csv as ucsv
import utils.util_doc.utils_file as ufile
import utils.util_rdkit.util_rdkit as urd
import time
import pandas as pd
from sklearn.preprocessing import normalize
def runCross_val_score(model,x,y,cv=5,metrix=["r2","neg_mean_absolute_error"]):
    res = []
    dictstr = {}
    for score in metrix:
        scoreCrossVali = cross_val_score(model, x, y, cv=cv, scoring=score)
        if score == "neg_mean_absolute_error":
            score = "mae"
        minV = min(scoreCrossVali)
        maxV = max(scoreCrossVali)
        strrest = "{}: average score of {} is {}\nmin {} max {}".format(str(model), score, scoreCrossVali.mean(), minV, maxV)
        print(strrest)
        dictstr.update({"model":str(model),"metrix_"+score:score,"cv_ave_"+score:scoreCrossVali.mean(),"cv_min_"+score:minV,"cv_max_"+score:maxV})
    return dictstr
def visualizeR2_score(density_test,density_pred,name="trainData_r2",imageDir="../data/pics/"):
    # 可视化
    slope, intercept, r_value, p_value, std_error = stats.linregress(density_test, density_pred)
    yy = slope * np.asarray(density_test) + intercept
    plt.scatter(density_test, density_pred, c='red', s=1)
    plt.plot(density_test, yy,
             label='Predicted TDEC = ' + str(round(slope, 2)) + ' * True TDEC + ' + str(round(intercept, 2)))
    plt.xlabel('True TDEC')
    score = r2_score(density_pred,density_test)
    mae = mean_absolute_error(density_pred,density_test)
    dictPd = {"mae":[mae],"score":[score],"density_pred":density_pred,"density_test":density_test}
    plt.xlabel('r2_score :{},mae: {}'.format(score,mae))
    plt.legend()
    plt.savefig(ufile.checkDirExsist(imageDir+ +".jpg"))
    plt.show()

    newDf = pd.DataFrame(pd.DataFrame.from_dict(dictPd,orient='index').values.T,columns=list(dictPd.keys()))
    newDf.to_csv(ufile.checkDirExsist(imageDir+name+".csv"))
    return score

def getConbineFeature(dataFile, wfn,topology,zero):
    featureType = ["DFT", "Topology","DFT_Topology","FingerPrint","FingerPrint_DFT","FingerPrint_Topology","FingerPrint_DFT_Topology"]
    featuerTypeValues = []
    smiles = ucsv.getValueByColumns(dataFile,"SMILES")
    dftData = ucsv.getDataframe(dataFile,wfn,True)
    topologyData = ucsv.getDataframe(dataFile,topology,True)
    dft_topologyData  = ucsv.getDataframe(dataFile,wfn+topology,True)
    # 如何在列方向上合并
    #dftData+topologyData
    fingerprintData = np.asarray([urd.fingerPrint(smi) for smi in smiles])
    fingerPrint_dftData,labels,FeatureNames = ucsv.getFeatureLabelsFromCsv(dataFile, propsNames=wfn, fingeprint=True)
    fingerPrint_topologyData,labels,FeatureNames = ucsv.getFeatureLabelsFromCsv(dataFile, propsNames=topology, fingeprint=True)
    fingerPrint_dft_topologyData,labels,FeatureNames = ucsv.getFeatureLabelsFromCsv(dataFile, propsNames=topology+wfn,exludeColNames = ["FILENAME","SMILES","TDEC","str_NitroBondShort","str_NitroBondLong"] ,fingeprint=True)
    return featureType, [dftData,topologyData,dft_topologyData,fingerprintData,fingerPrint_dftData,fingerPrint_topologyData,fingerPrint_dft_topologyData],labels
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank:{0}".format(i))
            print("Mean validation score:{0:.3f}(std:{1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters:{0}".format(results['params'][candidate]))
            print("")
# 网格搜索交叉验证（GridSearchCV）：以穷举的方式遍历所有可能的参数组合
# 随机采样交叉验证（RandomizedSearchCV）：依据某种分布对参数空间采样，随机的得到一些候选参数组合方案
def getBestParameter(model,paraDict={'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf']},
                     randomise=True,n_iter_search=80,scoring=["r2","neg_mean_absolute_error"]):

    res = None
    clf = None
    start = time.time()
    if randomise:
        clf = RandomizedSearchCV(model, param_distributions=paraDict, n_iter=n_iter_search,scoring=scoring)
        print("RandomizedSearchCV took %.3f seconds for %d candidates"
              "parameter settings." % ((time() - start), n_iter_search))
    else:
        clf = GridSearchCV(model, paraDict,scoring=scoring)
        print("GridSearchCV took %.3f seconds for %d candidates"
              "parameter settings." % ((time() - start), n_iter_search))

    report(clf.cv_results_)
    return res


if __name__ == '__main__':
    # TODO 调整参数
    # parameters = {"kernel":("linear","rbf"),"C":range(1,1000)}
    sigma = 5
    modelKRR = KernelRidge(kernel='rbf')
    modelRF = RandomForestRegressor()
    modelSVR = SVR(
        kernel='rbf',  # 'linear'，'poly'，'rbf'，'sigmoid'，'precomputed'或者callable之一
        degree=3,
        gamma='scale',
        coef0=0.0,
        tol=0.001,
        C=100000,
        epsilon=0.4,
        shrinking=True,
        cache_size=500,
        verbose=False, max_iter=-1
    )
    folds = 5
    dataSplit = ShuffleSplit(n_splits=folds, test_size=float(1 / folds))
    dataSplit = KFold(n_splits=folds, random_state=1, shuffle=True)

    RunResultOutPath = "../../data/dataFeatureCal/"
    dataProcessResult = "../../data/dataFeatureCal/wfnparserOut/wfnData.csv"
    models = [modelKRR,modelRF,modelSVR]
    featureType = ["DFT", "Topology", "FingerPrint"]
    type01 = [0,1]
    for zero in type01:
        pathOut = "../../data/dataFeatureCal/{}/".format(zero)
        wfn = [i for i in  ufile.readFileLines("../../data/dataFeatureCal/wfnparserOut/FeatureDeduplicated.txt",True) if "---------" not in i ]
        topology = [i for i in ufile.readFileLines(pathOut + "FeatureDeduplicated.txt",True) if "---------" not in i ]
        # dataFile = pathOut+ "combineRes.csv"
        dataFile = pathOut+ "combineRes.csv"
        featureTypes,featuresArrays,labelsss = getConbineFeature(dataFile,wfn,topology,zero)
        row2write = []
        for conbine,data in zip(featureTypes,featuresArrays):
            if len(featuresArrays) != len(featuresArrays):
                exit(-8)
            for modeltemp in models:
                # 0 or 1 | featureCombine | models
                resdict = runCross_val_score(modeltemp,data,labelsss,cv=dataSplit)
                resdict.update({"zero":zero,"featureType":conbine})
                # draw pic
                r2scores = []
                r2Train = []
                maes = []
                r2data = []
                for train, test in dataSplit.split(range(len(labelsss))):
                    # print(train,test)
                    traindata = [data[i] for i in train]
                    trainlabel = [labelsss[i] for i in train]

                    testdata = [data[i] for i in test]
                    testlabel = [labelsss[i] for i in test]

                    # 对每一次的划分 拟合最好的模型
                    modeltemp.fit(traindata,trainlabel)
                    train_pred = modeltemp.predict(traindata)
                    test_pred = modeltemp.predict(testdata)
                    data_pred = modeltemp.predict(data)
                    r2data.append(r2_score(data_pred,labelsss))
                    r2scoreT = r2_score(test_pred, testlabel)
                    r2scores.append(r2scoreT)
                    maeT = mean_absolute_error(test_pred, testlabel)
                    maes.append(maeT)
                    r2Train.append(r2_score(train_pred,trainlabel))
                    if r2scoreT > 0.6:
                        visualizeR2_score(trainlabel,train_pred,"{}_{}_{}_{}".format(zero,conbine,str(modeltemp),"in TrainData"),pathOut+"picsRegression/")
                        visualizeR2_score(testlabel,test_pred,"{}_{}_{}_{}".format(zero,conbine,str(modeltemp),"in ValiData"),pathOut+"picsRegression/")
                avg_r2score = np.mean(r2scores)
                avg_maes = np.mean(maes)
                resdict.update({"r2":avg_r2score})
                resdict.update({"train_r2":np.mean(r2Train)})
                resdict.update({"avg_maes":avg_maes})
                resdict.update({"r2_data":np.mean(r2data)})
                ufile.showObj(resdict)
                row2write.append(resdict)

        ucsv.writeDicts2Csv(row2write,pathOut+"Kfold_5_Result_{}_{}.csv".format(zero,ufile.getFileNameByPath(dataFile)))