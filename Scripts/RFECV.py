import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score

from sklearn.datasets import make_classification
import utils.util_alg.util_math as umath
import utils.util_doc.utils_file as ufile
import utils.util_doc.utils_csv as ucsv

# StratifiedKFold 只支持分类的问题
modelSvr = SVR(kernel='linear')
modelRF = RandomForestRegressor()
# 实验不需要标准化，标准化只是为了加快神经网络的收敛速度。
NORMALIZE_flagV = 0
MIN_FEATURES_TO_SELECT = 20
interation = 20
model = modelRF
# stepDelete = 10
stepDelete = 2
foldNum = 5
rfecv = RFECV(estimator=model, step=stepDelete, cv=KFold(foldNum), n_jobs=-1,
              scoring='r2', min_features_to_select=MIN_FEATURES_TO_SELECT)


def drawRFCV(rfecv, outPakage):
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected by {} ".format(str(model)))
    plt.ylabel("Cross validation score (nb of correct regression)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.savefig(outPakage + ufile.getFileRenamedByTime("effectionOfSelection.png"))
    plt.show()


def showRFECV(rfecv):
    print("Optimal number of features : {} ".format(rfecv.n_features_))
    print("Ranking of features ( {} ) : {} ".format(len(rfecv.ranking_), rfecv.ranking_))
    print("Support ( {} ) is {}".format(len(rfecv.support_), rfecv.support_))
    print("Grid Scores( {} ) {}".format(len(rfecv.grid_scores_), rfecv.grid_scores_))


def runCross_val_score(model, x, y, metrix="r2"):
    # metrix = "neg_mean_absolute_error"
    print("model {}".format(str(model)))
    for i in range(10):
        scoreCrossVali = cross_val_score(model, x, y, cv=KFold(foldNum), scoring=metrix)
        print("average value of {}, {}".format(i, scoreCrossVali.mean()))
        # for i in scoreCrossVali:
        #     print(i)
        minV = min(scoreCrossVali)
        maxV = max(scoreCrossVali)
        print("round {} : min {} max {}".format(i, minV, maxV))


def rfecvFit(rfecv, features, labels, outPakage):
    rfecv.fit(features, labels)
    dictFeatureFilterd = []
    names = []
    print("feNames {} len(rfecv.ranking_) {}".format(len(feNames), len(rfecv.ranking_)))
    if len(feNames) != len(rfecv.ranking_):
        exit(-1)
    for rank, name in zip(rfecv.ranking_, feNames):
        if rank < 2:
            dictFeatureFilterd.append({"rank": rank, "FeatureName": name})
            names.append(name)
    ucsv.writeDicts2Csv(dictFeatureFilterd, outPakage + ufile.getFileRenamedByTime("ResultSelection.csv"))
    # drawRFCV(rfecv,outPakage)
    # showRFECV(rfecv)
    return names


if __name__ == '__main__':
    type01 = [0, 1]
    # dataProcessResult = "../../data/dataProcess/out/dataProcessResult.csv"
    dataProcessResult = "../../data/dataFeatureCal/wfnparserOut/wfnData.csv"
    dataFeatureCal = "dataFeatureCal2"
    for i in type01:
        pathOut = "../../data/{}/rfecvSelection/{}/".format(dataFeatureCal, i)
        featureCsv = "../../data/{}/{}/".format(dataFeatureCal, i) + "combineRes.csv"

        FeatureToplofile = featureCsv.replace("combineRes.csv", "FeatureTopology.txt")
        FeatureWfnfile = dataProcessResult.replace("wfnData.csv", "FeatureWfn.txt")
        allFeNameWfn = ucsv.getRowHeader(dataProcessResult)
        allFeNameRdkit = ucsv.getRowHeader(featureCsv.replace("combineRes.csv", "topologyData.csv"))
        FeatureToplo = ufile.readFileLines(FeatureToplofile, True)
        FeatureWfn = ufile.readFileLines(FeatureWfnfile, True)

        features, labels, feNames = ucsv.getFeatureLabelsFromCsv(featureCsv, NORMALIZE_flag=NORMALIZE_flagV)
        featuresFiltered, labelsFiltered, _ = ucsv.getFeatureLabelsFromCsv(featureCsv,
                                                                           NORMALIZE_flag=NORMALIZE_flagV)
        print(type(features), "RFECV特征总数", len(feNames))
        # runCross_val_score(model,featuresFiltered,labelsFiltered)
        # 跑五十次,求个交集生成到临时文件目录
        for i in range(interation):
            rfecvFit(rfecv, features, labels, pathOut + "temp/")

        rankfile_from_dir = ufile.getFilesFromDir(pathOut + "temp/", "csv", abs=True)
        fitNameS = [set(ucsv.getValueByColumns(i, "FeatureName")) for i in rankfile_from_dir
                    if len(ucsv.getValueByColumns(i, "FeatureName")) < 150]
        print(fitNameS)
        rest = fitNameS[0]
        for i in fitNameS[1:]:
            rest.intersection_update(i)
        print("随机森林支持{}个特征".format(len(rest)))
        ufile.writeObjectsToTxt(rest, pathOut + "RandomForestSupport.txt")

        add2topoogy = [i for i in rest if i not in FeatureToplo and i in allFeNameRdkit]
        add2wfn = [i for i in rest if i not in FeatureWfn and i in allFeNameWfn]

        add2topoogy.insert(0, "\n---------------\n")
        add2wfn.insert(0, "\n--------------\n")
        ufile.writeObjectsToTxt(add2topoogy, FeatureToplofile, mode="a")
        ufile.writeObjectsToTxt(add2wfn, FeatureWfnfile, mode="a")
# Optimal number of features : 3
# Ranking of features : [ 5  1 12 19 15  6 17  1  2 21 23 11 16 10 13 22  8 14  1 20  7  9  3  4 18]
