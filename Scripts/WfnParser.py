import utils.g09.AlphaBand as bandGap
# Balance of charges (nu):   0.12558124
# Product of sigma^2_tot and nu:   0.00006509 a.u.^2 (   25.63019 (kcal/mol)^2)
# Internal charge separation (Pi):   0.02653619 a.u. (     16.65172 kcal/mol)
# 有机分子融化热、表面张力、液体/晶体密度：Chem. Phys., 204, 289 (1996)
# Politzer等人提出的General interaction properties function(GIPF)就建立了这么一套关系，见
# J.Mol.Struct. (THEOCHEM),307,55
# 。GIPF的本质上含义是，依靠基于分子表面上静电势分布定义的分子描述符，通过拟合QSPR方程，
# 可以很好地预测上述性质。
# 这些分子描述符包括分子整体/静电势为正/静电势为负区域的表面积，表面上静电势最大值、最小值，
# 表面上静电势的整体/正值/负值部分的平均值/平均偏差/方差，以及它们进一步相互运算得到的几个描述符
# （具体公式见Multiwfn 2.5手册3.15.1节）。计算并利用这些描述符也是定量分子表面分析的主要组成部分。
# Sum of above integrals:             2.00005879
# Integral下面的数值就是相应原子的自旋布居
import math,os,pandas as pd,re,time
import utils.g09.NitroCharge as Ncharge
import utils.g09.AlphaBand as bandGap
import utils.util_doc.utils_file as ufile
import utils.util_doc.utils_csv as ucsv
import utils.g09.NitroBondLen as bondL
import utils.util_rdkit.util_rdkit as urd
import utils.util_alg.util_math as umath
from rdkit import Chem

# ----------------------计算文件路径------------------------------
# 表面积
outPath_surfaceAnalyse = "D:\softChem\calOuts\wfnout1\\"
# MPP 什么堆积啥的 http://sobereva.com/618
outPathWfn_MPP = "D:\softChem\calOuts\outmpp\\"
# 自选密度
outPathIntegral = "D:\softChem\calOuts\integrelRes\\"
#
logPathHof = "D:/softChem/logs/"
logPath2Second = "D:\softChem\calOuts\logsSeconde\logSecond\\"
# -------------------------------------------------------------

def getStrBetween(fileos,start,end):
    reg =  start + "(.*?)" + end
    findall = re.findall(reg, fileos,re.S)
    return findall
def getStrBetweenHunger(fileos,start,end):
    reg =  start + ".*" + end
    findall = re.findall(reg, fileos,re.S)
    return findall

def getSphericity(log_content):
    # Sphericity:
    lineSphericity = [i.strip() for i in log_content if i.startswith(" Sphericity:")][0]
    Sphericity = lineSphericity.split(":")[1].strip()
    return Sphericity
def getDensityAndMpiAndSphericityAndPi(log_content):
    lines = log_content.split("\n")
    lineDen = [i.strip() for i in log_content if i.startswith(" Estimated density according to mass and volume (M/V)")][0]
    lineDen = lineDen.split(":")[1].strip().split(" ")[0]

    linempi = [i.strip() for i in lines if i.startswith(" Molecular polarity index (MPI)")][0]
    MPI = linempi.split(":")[1].strip().split("(")[1].strip().split(" ")[0]
    # Sphericity:
    lineSphericity = [i.strip() for i in log_content if i.startswith(" Sphericity:")][0]
    Sphericity = lineSphericity.split(":")[1].strip()

    linePi = [i.strip() for i in log_content if i.startswith(" Internal charge separation (Pi)")][0]
    Pi = linePi.split(":")[1].strip().split(" ")[0]
    return lineDen,MPI,Sphericity,Pi
aS = 2.130
bS = 0.93
cS = -17.844
aG = 0.000267
bG = 1.650087
cG = 2.966078
#表观电荷
def getThermalVapAndSteam(lines):
    lineSurface = [i.strip() for i in lines if i.startswith(" Overall surface area:")][0]
    A = lineSurface.split(":")[1].strip().split(" ")[0]

    linesigma = [i.strip() for i in lines if i.startswith(" Overall variance (sigma^2_tot):")][0]
    sigma = linesigma.split(":")[1].strip().split("(")[1].strip()
    sigma = float(sigma)
    A =float(A)
    vap =  2.130 * math.sqrt(A) + 0.930 * math.sqrt(sigma) - 17.844
    steam =0.000267 * (A ** 2) + 1.650087 * math.sqrt(sigma) + 2.966078
    return vap,steam
# header = ["TDEC","Density", "MPI" ,"Sphericity","Pi","ThermalGas","ThermalSteam","posSur","negSuf","posV","negV"]
def getMPPSDP(lines):
    linesigma = [i.strip() for i in lines if i.startswith(" Molecular planarity parameter (MPP) is")][0]
    mpp = linesigma.split(" ")[-2].strip()
    linesigma = [i.strip() for i in lines if i.startswith(" Span of deviation from plane (SDP) is")][0]
    sdp = linesigma.split(" ")[-2].strip()
    return mpp,sdp
# ^-?([1-9]\d*\.\d*|0\.\d*[1-9]\d*|0?\.0+|0)$　 //匹配浮点数
def matchFloatNumer(str):
    matcher = re.compile("""(?<![a-zA-Z:])[-+]?\d*\.?\d+""")
    findall = matcher.findall(str)
    return findall[0] if findall else ''
def matchFloatNumers(str):
    matcher = re.compile("""(?<![a-zA-Z:])[-+]?\d*\.?\d+""")
    findall = matcher.findall(str)
    return findall if findall else ''
# 所有的表面分析性质
def getAllPros(log_content):
    start = "================= Summary of surface analysis =================\n \n"
    end = "\n \n Surface analysis finished!"
    between = getStrBetween(log_content,start, end)
    between = between[0]
    lines = between.split("\n")
    prosLen = len(lines)
    header = []
    row = []
    dictRow = {}
    for i in lines:
        lineName = i.split(":")[0].strip()
        header.append(lineName)
        restName = i.split(":")[1]
        inBlank = getStrBetween(restName, "\( ", "\)")
        inBlank = '' if not inBlank else inBlank[0]
        numer = matchFloatNumer(inBlank) or matchFloatNumer(restName)
        row.append(numer)
        dictRow[lineName] = numer
    return dictRow,header,row
# logcontent and logcontentLines
def readFile(fileName):
    try:
        with open(fileName, 'r') as g:
            log_content = g.read()
            lines = log_content.split("\n")
            if log_content and lines:
                return [log_content,lines]
            else:
                raise Exception("文件不能为空",fileName)
                return None
    except Exception as e:
        raise Exception(e,"读文件异常")

def getIntegrals(lines):
    linesigma = [i.strip() for i in lines if i.startswith(" Sum of above integrals:")][0]
    integral = linesigma.split(":")[1].strip()
    return integral
# 统一获取剩下没有计算完的文件
def genRestMols(rowOrigin,file):
    needToCalSmiles = ufile.readFileLines(file,True)
    for s in needToCalSmiles:
        if s.endswith(".mol"):
            dict_search = ucsv.getDictRowByDictSearch(rowOrigin, {"FILENAME": s})
            s = dict_search["SMILES"]
        else:
            dict_search = ucsv.getDictRowByDictSearch(rowOrigin, {"SMILES": s})
        print(s)
        print(dict_search)
        FILENAME = dict_search["FILENAME"].strip()
        mol = Chem.MolFromSmiles(s)
        urd.writeMolFile(mol, "../../data/dataFeatureCal2/nofile/"+file+"/" + str(len(needToCalSmiles)) + "/", FILENAME)
# 获取构象原子间距离
def getConnectionBondOrder(logContent):
    betweenNAO = getStrBetween(logContent, "Atom-atom overlap-weighted NAO bond order:",
                               "Atom-atom overlap-weighted NAO bond order, Totals by atom:")
    betweenNAO = betweenNAO[-1]
    linesNAO = [i for i in betweenNAO.split("\n") if i.strip() and "----" not in i and "Atom" not in i]
    indexs = []
    symbols = []
    matrix = []
    caled = set([])
    ditcBondOrder = {}
    for line in linesNAO:
        group = re.search(r"[0-9]+\.  [A-Z]", line).group().split(".")
        index = int(group[0].strip())
        symbol = group[1].strip()
        lineFloat = line[9:]
        numers = matchFloatNumers(lineFloat)
        if index not in indexs:
            indexs.append(index)
            symbols.append(symbol)
            matrix.append(numers)
        else:
            matrix[index-1].extend(numers)
    for i in range(0,len(indexs)):
        for j in range(0,len(matrix[i])):
            # caled aready
            caledCombination = "".join(sorted([str(i),str(j)]))
            if caledCombination in caled:
                continue
            conName =  "".join(sorted([symbols[i],symbols[j]]))
            if "H" not in conName:
                caled.add(caledCombination)
                valueBorder = matrix[i][j]
                if float(valueBorder) > 0.05:
                    oldV =  ditcBondOrder.setdefault(conName,[])
                    oldV.append(valueBorder)
                    ditcBondOrder.update({conName:oldV})
    dictRes = {"CCMin":0,"CCMax":0,
               "CNMin":0,"CNMax":0,
               "COMin":0,"COMax":0,
               "NNMin":0,"NNMax":0,
               "NOMin":0,"NOMax":0,
               }
    for k,v in ditcBondOrder.items():
        kmax = k+"Max"
        vmax = max(v)
        kmin = k+"Min"
        vmin = min(v)
        dictRes.update(
            {kmax:vmax,kmin:vmin}
        )
    return dictRes


def filterDescriptors(wfnDataPath = "../data/dataFeatureCal2/wfnparserOut/wfnData.csv"):
    wfnparserOut = "../../data/dataFeatureCal2/wfnparserOut/"
    pathRelationFiltered = wfnparserOut+ "wfnRelation.csv"
    pathRelationAll = wfnparserOut+"wfnRelationAll.csv"
    pathFeatureNames = wfnparserOut +"FeatureWfn.txt"
    
    dictsCor,allRetaion = umath.getCorelationDicts(wfnDataPath,pearsonLimit=0.4,mutualLimit=0.1,entropyLimit=0.1)
    ucsv.writeDicts2Csv(dictsCor, pathRelationFiltered,True)
    # 不能有空值
    ucsv.writeDicts2Csv(allRetaion, pathRelationAll,True)
    FeatuerNameFilterd = ucsv.getValueByColumns(pathRelationFiltered, "FeatureName")
    ufile.writeObjectsToTxt(FeatuerNameFilterd, pathFeatureNames)
#错误文件输出路径
molOutPath =  "../../data/dataFeatureCal2/mols/"
dataProcessResult =  "../../data/dataProcess/out/dataProcessResult.csv"
wfnDataPath = "../../data/dataFeatureCal2/wfnparserOut/wfnData.csv"

def parse():
    # 遍历check文件
    files = ufile.getFilesFromDir(outPath_surfaceAnalyse, ".fchk.wfnout")
    filesSurface = [i for i in files if "ochem" not in i]
    filesCBS = ufile.getFilesFromDir(logPath2Second, ".log")
    rowOrigin = ucsv.getRowsDict(dataProcessResult)

    rows = []
    count = 0
    SMILEBDE = []
    needToCalSmiles = []
    needToCalFileNames = []
    needToCalSmilesCBS = []
    needToCalFileNamesCBS = []
    # for i in rowOrigin[0:20]:
    for i in rowOrigin[:]:
        TDEC = i["TDEC"]
        FILENAME = i["FILENAME"]
        SMILES = i["SMILES"]
        filename = i["FILENAME"].split(".")[0]

        if "entry" not in filename:
            filename = filename.replace("New_","")
            filename = filename.replace("_","-")
        # search = ucsv.getDictRowByDictSearch(rowOrigin, {"FILENAME": filename + ".mol"})
        # wfn文件 是否存在 log fchk wfn
        fileExist = [i for i in files if filename in i]
        fileExistCBS = [i for i in filesCBS if filename in i]
        if not fileExist:
            needToCalSmiles.append(SMILES)
            needToCalFileNames.append(FILENAME)
            mol = Chem.MolFromSmiles(SMILES)
            urd.writeMolFile(mol, molOutPath + "B3/", FILENAME)
            continue
        elif not fileExistCBS:
            needToCalSmilesCBS.append(SMILES)
            needToCalFileNamesCBS.append(FILENAME)
            mol = Chem.MolFromSmiles(SMILES)
            urd.writeMolFile(mol, molOutPath + "CBS/", FILENAME)
            continue
        else:
            count = count + 1
        print(count, filename)
        # continue
        fileNameSufface = outPath_surfaceAnalyse + fileExist[0]
        fileNameMpp = outPathWfn_MPP + fileExist[0]
        # surface 3 + N
        try:
            fileNameSufface1 = readFile(fileNameSufface)
            if fileNameSufface1:
                dictRow = getAllPros(fileNameSufface1[0])[0]
                Sphericity = getSphericity(fileNameSufface1[1])
                ThermalGas, ThermalSteam = getThermalVapAndSteam(fileNameSufface1[1])
                dictTemp = {"Sphericity": Sphericity, "ThermalGas": ThermalGas, "ThermalSteam": ThermalSteam}
                dictRow.update(dictTemp)
        except Exception as e:
            print(e)
            dictTemp = {"Sphericity": None, "ThermalGas": None, "ThermalSteam": None}
            print("表面分析报错：", fileNameSufface)
            mol = Chem.MolFromSmiles(SMILES)
            urd.writeMolFile(mol, molOutPath + "B3/", FILENAME)
            continue
        dictRow.update(dictTemp)
        # zixuanmidu 4 + N
        fileNameIntegral1 = readFile(outPathIntegral + fileExist[0])
        try:
            if fileNameIntegral1:
                integral = getIntegrals(fileNameIntegral1[1])
                dictRow.update({"Integral": integral})
        except Exception as e:
            print(e)
            dictRow.update({"Integral": None})
            print("自旋密度报错：", outPathIntegral + fileExist[0])
            mol = Chem.MolFromSmiles(SMILES)
            urd.writeMolFile(mol, molOutPath + "B3/", FILENAME)
            continue
        # 计算生成含 hof,bandEnergy,nitroCharge,bde 4 + N + 6

        logfile = logPathHof + "B3LYP6-31xG2dfp." + filename + ".log"
        exeHOF = "perl " + logPathHof + "HOF.pl " + logfile

        logNameCBS_4M = "CBS-4M." + filename + ".log"
        exeBDE = "perl " + logPath2Second + "CalcBDE.pl " + logPath2Second + logNameCBS_4M
        try:
            BDE = os.popen(exeBDE).read()
            float(BDE)
            # TODO
            dictRow.update({"BDE": BDE})
        except Exception as e:
            SMILEBDE.append(SMILES)
            print("BDE 出错了: " + logPath2Second + logNameCBS_4M, e)
            mol = Chem.MolFromSmiles(SMILES)
            urd.writeMolFile(mol, molOutPath + "CBS/", FILENAME)
        try:
            logfileContent = readFile(logfile)
            HOF = os.popen(exeHOF).read()
            NAOOrderDict = getConnectionBondOrder(logfileContent[0])
            NitroCharge = Ncharge.getWeakestNitroCharge(logfileContent[0])
            # 浮点数转化错误             getAlphaBand
            alphaGap = bandGap.getAlphaBand(logfileContent[0], i)
            nitroLongest, nitroShortest, bondMax, bondMin = bondL.getWeakestBondLength(logfileContent[1])

            dictTemp = {"HOF": HOF, "NitroCharge": NitroCharge,
                        "alphaGap": alphaGap, "NitroBondMax": nitroLongest, "NitroBondMin": nitroShortest,
                        "BondMax": bondMax, "BondMin": bondMin
                        }
            dictRow.update(dictTemp)

            dictRow.update(NAOOrderDict)
        except Exception as e:
            print("B3LYP6-31xG2dfp 某属性出错了: " + logfile)
            print(e)
            mol = Chem.MolFromSmiles(SMILES)
            urd.writeMolFile(mol, molOutPath + "B3/", FILENAME)
            continue
        # 4 + N + 6
        # MPP PI堆积
        try:
            fileNameMpp1 = readFile(fileNameMpp)
            if fileNameMpp1:
                MPP, SDP = getMPPSDP(fileNameMpp1[1])
                dictTemp = {"MPP": MPP, "SDP": SDP, "MPP+SDP": float(MPP) + float(SDP)}
                dictRow.update(dictTemp)
        except Exception as e:
            print("MPP,SDP 出错了: " + fileNameMpp)
            print(e)
            mol = Chem.MolFromSmiles(SMILES)
            urd.writeMolFile(mol, molOutPath + "B3/", FILENAME)
            continue
        dictRow.update({"SMILES": SMILES, "TDEC": TDEC, "FILENAME": FILENAME})
        rows.append(dictRow)
    print(len(rows))
    ucsv.writeDicts2Csv(rows, wfnDataPath, True)

# 计算并且筛选特征
if __name__ == '__main__':

    # 1 / get datafile.csv
    parse()
    # TODO 获取每两个特征之间的相关系数矩阵
    filterDescriptors(wfnDataPath)

    # 获取没有计算的，没有文件，会被记录在一个temp.txt 里面
    # ufile.writeObjectsToTxt(needToCalSmiles, "nofileB3lyp.file")
    # ufile.writeObjectsToTxt(needToCalSmiles, "nofileCBS.file")
    # genRestMols(rowOrigin,"nofileB3lyp")
    # genRestMols(rowOrigin,"nofileCBS")

    # 删除pdframe 里面的SMILES FILENAME
    # dataC = pd.read_csv('wfnData.csv')
    # frameCor = dataC.corr()
    # frameCor.to_csv('wfnPirsonNoNoneValue.csv')
    # dataC = pd.read_csv('wfnData.csv')
    # frameCor = dataC.corr()
    # frameCor.to_csv('wfnPirson.csv')


