import rdkit,collections

from scipy.stats import pearsonr
# deepchem 只有一百多个
# import deepchem as dc
# rdkit_featurizer = dc.feat.RDKitDescriptors()
import utils.util_doc.utils_csv as ucsv
import utils.util_alg.util_math as umath
import matplotlib.pyplot as plt
import os,sys,csv,re
# import seaborn
# import deepchem as dc
import utils.util_doc.utils_file as ufile
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as rdDes

import pandas as pd
import numpy as np
# from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import Lipinski,AllChem,Descriptors
from rdkit.Chem import rdChemicalFeatures
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.ML.Descriptors import Descriptors
from rdkit.ML.Descriptors.Descriptors import DescriptorCalculator
from rdkit.Chem import AllChem
import utils.util_rdkit.util_rdkit as urd
from shutil import copy
from sys import exit
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#
# f_regression,
# mutual_info_regression
# * atompaircounts
# * morgan3counts
# * morganchiral3counts
# * morganfeature3counts
# * rdkit2d
# * rdkit2dnormalized
# * rdkitfpbits
def getRdkitFeatureHeaderAndGennerator(normalize):
    if normalize:
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
    else:
        # generator = MakeGenerator(("RDKit2D",))
        rdDescriptors.RDKit2D()
    RdkitFeatureHeader = []
    for name, numpy_type in generator.GetColumns():
        RdkitFeatureHeader.append(name)
        # print("name: {} data type: {}".format(name, numpy_type))
    return generator,RdkitFeatureHeader[1:]
def getRDKit2D(generator,smiles):
    data = generator.process(smiles)
    if not data[0]:
        print(smiles,"failed while gennerating rdkitFeature")
        exit(-1)
    return data[1:]
    # 输出到csv
def getFeaturesDictTopology(smile,generator,rdheader,FeatureNames = [],exluds = [],featureZeroOne=True):
    dictRdkitFeature = {}
    # headerRdkit.append("SMILES")
    d = getRDKit2D(generator, smile)
    for index,fName in enumerate(rdheader):
        if fName not in exluds:
            if FeatureNames:
                if fName in FeatureNames :
                    dictRdkitFeature.update({fName:d[index]})
            else:
                dictRdkitFeature.update({fName: d[index]})
    # dictRdkitFeature.update({"SMILES":smile})
    mol = Chem.MolFromSmiles(smile)
    inforMore = urd.getNitroType_0or1(mol,featureZeroOne)
    for keyInfo in inforMore.keys():
        if keyInfo not in exluds:
            if FeatureNames:
                if keyInfo in FeatureNames :
                    dictRdkitFeature.update({keyInfo:inforMore[keyInfo]})
            else:
                dictRdkitFeature.update({keyInfo:inforMore[keyInfo]})
    # 排序特征向量
    dictRdkitFeature = collections.OrderedDict(sorted(dictRdkitFeature.items(),key=lambda x:x[0]))
    return dictRdkitFeature

def getFeatures(csv="../data/csv/mpnn2022-03-16.csv",
                csvOutStructureInfo = "",featureZeroOne=True,normalize=False):
    gennerator, headerRdkit = getRdkitFeatureHeaderAndGennerator(normalize)
    SMILES = ucsv.getValueByColumns(csv,"SMILES")
    TDECs = ucsv.getValueByColumns(csv,"TDEC")
    FileNames = ucsv.getValueByColumns(csv,"FILENAME")
    rows = []
    for smile,tdec,filename in zip(SMILES,TDECs,FileNames):
        baseInfo = {"SMILES":smile,"TDEC":tdec,"FILENAME":filename}
        fea =   getFeaturesDictTopology(smile,gennerator,headerRdkit,featureZeroOne=featureZeroOne)
        fea.update(baseInfo)
        rows.append(fea)
    ucsv.writeDicts2Csv(rows,csvOutStructureInfo)
# inMol: a molecule
# confId: (optional) the conformation ID to use
# useAtomicMasses: (optional) toggles use of atomic masses in the calculation. Defaults to True
def getInertia(mol,useMass = False):
    # resIner = None
    # resIner = rdDes.CalcInertialShapeFactor(mol,useAtomicMasses=useMass)
    resIner3D = Chem.Descriptors3D.InertialShapeFactor(mol,useAtomicMasses=useMass)

    PMI1 = Chem.Descriptors3D.PMI1(mol,useAtomicMasses=useMass)
    PMI2 = Chem.Descriptors3D.PMI2(mol,useAtomicMasses=useMass)
    PMI3 = Chem.Descriptors3D.PMI3(mol,useAtomicMasses=useMass)

    # return resIner,resIner3D,PMI1,PMI2,PMI3
    return PMI1,PMI2,PMI3
# 计算并且筛选特征
import copy as copyObj
if __name__ == '__main__':
    # getRDKit2D()
    # for row in ucsv.getRowsDict(r"../dataProcess\traditional\famous_smiles.csv"):
    #
    #     mol = Chem.MolFromInchi(row["INCHI"], removeHs=False)
    #     molHere = copyObj.deepcopy(mol)
    #     try:
    #         molHere = urd.OPTMolAddHs(molHere)
    #     except:
    #         pass
    #     if molHere:
    #         # conformer = molHere.GetConformer()
    #         print((row["NAME"],Chem.MolToSmiles(Chem.MolFromInchi(row["INCHI"])) ),getInertia(molHere))

    print(rdkit.__version__)
    type01 = [0,1]
    dataProcessResult = "dataProcessResult.csv"
    outpath = "dataFeatureCal2"
    for i in type01:
        # pathOut = "../../data/{}/{}/".format(outpath,i)
        pathOut = ''
        dataPath = pathOut + "topologyData.csv"
        getFeatures(dataProcessResult,dataPath,featureZeroOne=i,normalize=True)
        # exit(-1)
        Featurerows,allRelation = umath.getCorelationDicts(dataPath,pearsonLimit=0.4,mutualLimit=0.1,entropyLimit=0.1)
        # 关注线性相关性质
        Featurerows = ufile.getSortedDictsByKey(Featurerows,"pearson")
        ucsv.writeDicts2Csv(Featurerows,pathOut+"topologyRelation.csv")
        ucsv.writeDicts2Csv(allRelation,pathOut+"topologyRelationAll.csv")

        FeatuerNames= ucsv.getValueByColumns(pathOut+"topologyRelation.csv", "FeatureName")
        ufile.writeObjectsToTxt(FeatuerNames,pathOut+"FeatureTopology.txt")

        # 合并wfn
        csv2 = dataPath
        csv1 = r"../../data\dataFeatureCal\wfnparserOut\wfnData.csv"
        ucsv.combine2OneCsv(csv1, csv2, csvOut=pathOut+"combineRes.csv", indexColName="SMILES", checkColName="TDEC")