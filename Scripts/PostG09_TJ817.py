#Version:2022/08/15 11:20 am
import sys
import os,csv,re,math,random,argparse
from math import sqrt, sin
import copy as copyObj
from tqdm import tqdm

import collections
def GetMolList(WorkDir):
    MolList=[]
    FileList = os.listdir(WorkDir)
    for File in FileList:
        File = File.lstrip()
        File = File.rstrip()
        Level = File.split('.')[0]
        MolName = re.sub(Level+'.','',File)
        DotInName = re.findall(r'\.', MolName)
        if len(DotInName) >= 2 or re.findall(r'no2', MolName, re.I):
            continue
        if re.search(r'\.log$', MolName) != None:
            MolName = re.sub(r'\.log','',MolName)
            if MolName in MolList:
                continue
            MolList.append(MolName)
        if re.search(r'\.wfn$', MolName) != None:
            MolName = re.sub(r'\.wfn', '', MolName)
            if MolName in MolList:
                continue
    return MolList
def getStrBetween(fileos,start,end):
    reg =  start + "(.*?)" + end
    findall = re.findall(reg, fileos,re.S)
    return findall
def getConnectionBondOrder(logContent):
    def matchFloatNumers(str):
        matcher = re.compile("""(?<![a-zA-Z:])[-+]?\d*\.?\d+""")
        findall = matcher.findall(str)
        return findall if findall else ''
    CHON = ['C', 'H', "O", 'N', ' ']
    def CHONonly(formula):
        formula.strip()
        s = ''.join([i for i in formula if i not in CHON])
        s = ''.join([i for i in s if not i.isdigit()])
        s.strip()
        if len(s) == 0:
            return True
        return False
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
        group = re.search(r"[0-9]+\.\s+\w+", line).group().split(".")
        index = int(group[0].strip())
        symbol = group[1].strip()
        lineFloat = line[9:]
        numers = matchFloatNumers(lineFloat)
        if index not in indexs:
            indexs.append(index)
            symbols.append(symbol)
            matrix.append(numers)
        else:
            matrix[index - 1].extend(numers)
    for i in range(0, len(indexs)):
        for j in range(0, len(matrix[i])):
            # caled aready
            caledCombination = "".join(sorted([str(i), str(j)]))
            if caledCombination in caled:
                continue
            conName = "".join(sorted([symbols[i], symbols[j]]))
            if not CHONonly(conName):
                continue
            if "H" not in conName:
                caled.add(caledCombination)
                valueBorder = matrix[i][j]
                if float(valueBorder) > 0.05:
                    oldV = ditcBondOrder.setdefault(conName, [])
                    oldV.append(valueBorder)
                    ditcBondOrder.update({conName: oldV})
    dictRes = {"CCMin": 0, "CCMax": 0,
               "CNMin": 0, "CNMax": 0,
               "COMin": 0, "COMax": 0,
               "NNMin": 0, "NNMax": 0,
               "NOMin": 0, "NOMax": 0,
               }
    for k, v in ditcBondOrder.items():
        kmax = k + "Max"
        vmax = max(v)
        kmin = k + "Min"
        vmin = min(v)
        dictRes.update(
            {kmax: vmax, kmin: vmin}
        )

    return dictRes


def ReadCBS(log):
    Dic = {}
    AtomCountDic = {}
    AtomEnthalpyDic = {'C':-37.786156,
                       'H':-0.500991,
                       'O':-74.991202,
                       'N':-54.522462,
                       'F':-99.649394,
                       'Cl':-459.674576}
    AtomHOFDic = {'C':0.27296900056,
                 'H':0.0830203899,
                 'O':0.0949038425,
                 'N':0.18004192466,
                 'F':0.0300857026,
                 'Cl':0.0463454652}
    try:
        with open (log, 'r') as LOG:
            LogBlock = LOG.read()
            AtomBlock = (re.findall('Symbolic Z-matrix:\n(.*?)\n NAtoms=', LogBlock, re.S))[0]
            AtomLine = AtomBlock.split("\n")
            AtomLine = AtomLine[1:-1]
            for line in AtomLine:
                AtomType = (line.split())[0]
                if AtomType in AtomCountDic.keys():
                    AtomCountDic[AtomType] += 1
                else:
                    AtomCountDic[AtomType] = 1
            Enthalpy = float((re.findall('CBS-4 Enthalpy=\s+(.*?)\s+CBS-4 Free Energy=', LogBlock, re.S))[-1])
            HOF_G = Enthalpy
            for key in AtomCountDic.keys():
                HOF_G += -(AtomEnthalpyDic[key] - AtomHOFDic[key])*AtomCountDic[key]
            if 'Mass' not in Dic.keys():
                Dic['Mass'] = (re.findall('Molecular mass:\s+(.*?)\s+amu', LogBlock))[0]
            Dic['Formula'] = (re.findall('Stoichiometry\s+(.*?)\s+Framework group', LogBlock))[0]
            Dic['HOF_G']=HOF_G*2625.51
            Dic['AtomCountDic'] = AtomCountDic
    except Exception as e:
        print(Dic['Mass'])
        pass

    return Dic

def ReadWFN(wfn):
    #TODO
    def getIntegrals(lines):
        try:
            linesigma = [i.strip() for i in lines if i.startswith(" Sum of above integrals:")][0]
            integrals = linesigma.split(":")[1].strip()
        except:
            raise Exception("Integrals extract failed in fchk2wfn")
        return float(integrals)
    def getMPPSDP(lines):
        linesigma = [i.strip() for i in lines if i.startswith(" Molecular planarity parameter (MPP) is")][0]
        mpp = linesigma.split(" ")[-2].strip()
        linesigma = [i.strip() for i in lines if i.startswith(" Span of deviation from plane (SDP) is")][0]
        sdp = linesigma.split(" ")[-2].strip()
        return {"MPP":mpp, "SDP":sdp}
    def getMolecularSize(wfnContent):
        resDic = {}
        try:
            farestDis = (re.findall('Farthest distance:   .+:\s+(.*?)\s+', wfnContent))[-1]
            Diameter = (re.findall('Diameter of the system:\s+(.*?)\s+', wfnContent))[-1]
            Radius = (re.findall('Radius of the system:\s+(.*?)\s+', wfnContent))[-1]
            threeSides = (re.findall('Length of the three sides:\s+(.*?)\sAngstrom', wfnContent))[-1].split("     ")
            resDic = {"FarthestDis": farestDis, "Diameter": Diameter, "Radius": Radius, "ThreeSides": threeSides}
        except:
            pass
        return resDic
    Dic = {}
    try:
        with open(wfn,'r') as WFN:
            WFNBlock = WFN.read()
            linesWFN = WFNBlock.split("\n")

            Dic['Volume'] = (re.findall('Volume:.*\(\s+(.*?)\s+Angstrom', WFNBlock))[0]
            Dic['Ro(g/cm^3)'] = (re.findall('Estimated density.*:\s+(.*?)\s+g\/cm', WFNBlock))[0]
            Dic['Area'] = (re.findall('Overall surface area:.*\(\s+(.*?)\s+Angstrom', WFNBlock))[0]
            Dic['PosArea'] = (re.findall('Positive surface area:.*\(\s+(.*?)\s+Angstrom', WFNBlock))[0]
            Dic['NegArea'] = (re.findall('Negative surface area:.*\(\s+(.*?)\s+Angstrom', WFNBlock))[0]
            Dic['Ave'] = (re.findall('Overall average value:.*\(\s+(.*?)\s+kcal\/mol', WFNBlock))[0]
            Dic['PosAve'] = (re.findall('Positive average value:.*\(\s+(.*?)\s+kcal\/mol', WFNBlock))[0]
            Dic['NegAve'] = (re.findall('Negative average value:.*\(\s+(.*?)\s+kcal\/mol', WFNBlock))[0]
            Dic['Sigma2'] = (re.findall('Overall variance.*\(\s+(.*?)\s+\(kcal', WFNBlock))[0]
            Dic['PosSigma2'] = (re.findall('Positive variance:.*\(\s+(.*?)\s+\(kcal', WFNBlock))[0]
            Dic['NegSigma2'] = (re.findall('Negative variance:.*\(\s+(.*?)\s+\(kcal', WFNBlock))[0]
            Dic['Nu'] = (re.findall('Balance of charges.*:\s+(.*?)\s+', WFNBlock))[0]
            Dic['NuSigma2'] = (re.findall('Product of sigma.*\(\s+(.*?)\s+\(kcal', WFNBlock))[0]
            Dic['Pi'] = (re.findall('Internal charge separation \(Pi\):.*\(\s+(.*?)\s+kcal\/mol', WFNBlock))[0]
            Dic['MPI'] = (re.findall('Molecular polarity index \(MPI\):.*\(\s+(.*?)\s+kcal\/mol', WFNBlock))[0]
            Dic['NonPolarArea'] = (re.findall('Nonpolar surface area.*:\s+(.*?)\s+Angstrom', WFNBlock))[0]
            Dic['PolarArea'] = (re.findall('Polar surface area.*:\s+(.*?)\s+Angstrom', WFNBlock))[0]
            Dic['Sphericity'] = (re.findall('Sphericity:\s+(.*?)\s+', WFNBlock))[0]
            Dic.update(getMolecularSize(WFNBlock))
            Dic.update(getMPPSDP(linesWFN))
            Dic.update({"Integrals":getIntegrals(linesWFN)})
    except FileNotFoundError as e:
        Dic.update({"NOFILE":str(e).split(":")[-1]})
        raise e
    except Exception as e:
        print("WFN Parser in File{}: Err {}".format(wfn,e))
    return Dic

def ReadOpt(log):
    def GetDipole(LOG):
        matcherStr = (re.findall('Dipole        =(.*)\n', LOG))[-1]
        if not matcherStr:
            return {}
        matcherStr = matcherStr.replace("D", "e")
        du = int(len(matcherStr) / 3)
        v1 = math.pow(float(matcherStr[:du]), 2)
        v2 = math.pow(float(matcherStr[du:du * 2]), 2)
        v3 = math.pow(float(matcherStr[du * 2:]), 2)
        diploe = math.sqrt(v1 + v2 + v3)
        return {"Dipole": diploe}
    def getOrientation(LogBlock):
        try:
            OrientBlock = (re.findall('Input orientation:\s+(.*?)\s+Distance matrix', LogBlock, re.S))[-1]
        except:
            OrientBlock = (re.findall('Standard orientation:\s+(.*?)\s+Rotational constants', LogBlock, re.S))[-1]
        OrientLine = (OrientBlock.split("\n"))[4:-1]
        return OrientLine
    Dic = {}
    AtomCountDic = {}
    AtomTypeList = []
    XList = []
    YList = []
    ZList = []
    ChargeList = []
    try:
        with open(log,'r') as LOG:
            LogBlock = LOG.read()
            # try:
            #     Dic.update(getConnectionBondOrder(LogBlock))
            # except Exception as e:
            #     raise Exception("While GetConnectionBondOrder: {}".format(e))
            # finally:
            #     pass
            AtomBlock = (re.findall('Symbolic Z-matrix:\n(.*?)\n NAtoms=', LogBlock, re.S))[0]
            AtomLine = AtomBlock.split("\n")
            blankIndex = AtomLine.index(" ")
            AtomLine = AtomLine[1:blankIndex]
            for line in AtomLine:
                AtomType = (line.split())[0]
                AtomTypeList.append(AtomType)
                if AtomType in AtomCountDic.keys():
                    AtomCountDic[AtomType] += 1
                else:
                    AtomCountDic[AtomType] = 1
            Dic['AtomTypeList'] = AtomTypeList
            Dic['AtomCountDic'] = AtomCountDic
            Dic.update(GetDipole(LogBlock))
            OrientBlock = getOrientation(LogBlock)
            for line in OrientBlock:
                X, Y, Z = (line.split())[-3:]
                XList.append(X)
                YList.append(Y)
                ZList.append(Z)
            Dic['XList'] = XList
            Dic['YList'] = YList
            Dic['ZList'] = ZList
            ChargeBlock = (re.findall('APT charges:\s+(.*?)\s+Sum of APT charges.*', LogBlock, re.S))[-1]
            ChargeLine = (ChargeBlock.split("\n"))[1:]
            for line in ChargeLine:
                ChargeList.append((line.split())[2])
            Dic['ChargeList'] = ChargeList
            if 'Mass' not in Dic.keys():
                Dic['Mass'] = (re.findall('Molecular mass:\s+(.*?)\s+amu', LogBlock))[0]
            Dic['Formula'] = (re.findall('Stoichiometry\s+(.*?)\s+Framework group', LogBlock))[0]
    except FileNotFoundError as e:
        Dic.update({"NOFILE":str(e).split(":")[-1]})
        raise e
    except Exception as e:
        raise Exception("Read Opt Err:{} {}".format(Mol,e))
    return Dic

def GetOB(Mass, AtomCount):
    Dic = {}
    try:
        Dic['OB1'] = 1600*(AtomCount['O']-AtomCount['C']-0.5*AtomCount['H'])/float(Mass)
        Dic['OB2'] = 1600*(AtomCount['O']-2*AtomCount['C']-0.5*AtomCount['H'])/float(Mass)
    except Exception as e:
        # print(e)
        pass
    Dic['OB1'] = '%.1f' % Dic['OB1']
    Dic['OB2'] = '%.1f' % Dic['OB2']
    return Dic


def GetLinkList(Type, X, Y, Z):
    CovalentRadiusDic = {'C':0.8,
                       'H':0.45,
                       'O':0.7,
                       'N':0.76,
                       'F':0.75,
                       'Cl':1.1}

    LinkList = []
    Dic = {}
    if Type and X and Y and Z:
        for i in range(0,len(X)):
            LocalLink = []
            for j in range(0,len(X)):
                if i == j:
                    continue
                xi = float(X[i])
                xj = float(X[j])
                yi = float(Y[i])
                yj = float(Y[j])
                zi = float(Z[i])
                zj = float(Z[j])
                Rij = math.sqrt((xi -xj)**2 + (yi -yj)**2 + (zi -zj)**2)
                MaxBondLength = 1.15*(CovalentRadiusDic[Type[i]] + CovalentRadiusDic[Type[j]])
                if Rij < MaxBondLength:
                    LocalLink.append(j)
            LinkList.append(LocalLink)
        Dic['LinkList'] = LinkList

    return Dic


def GetHSub(Area, NuSigma2):
    Dic = {}
    A = 0.000267
    B = 1.650087
    C = 2.966078
    if Area and NuSigma2:
        Dic['HSub'] = 4.182*(A*float(Area)**2 + B*math.sqrt(float(NuSigma2))+C)

    return Dic


def GetHOF_S(HOF_G, HSub):
    Dic = {}
    if HOF_G and HSub:
        Dic['HOF_S'] = float(HOF_G)-float(HSub)
        Dic['HOF_S'] = '%.1f' % Dic['HOF_S']

    return Dic


def GetDensity(Ro, NuSigma2, Mol):
    Dic = {}
    ExpDensityDic = {'TATB':1.937,
                     'RDX':1.806,
                     'CL20':2.04,
                     'HMX':1.905,
                     'FOX-7':1.878,
                     'TNT':1.654}
    Beta1 = 1.0462
    Beta2 = 0.0021
    Beta3 = -0.1586
    if Ro and NuSigma2:
        Dic['Density(g/cm^3)'] = Beta1*float(Ro) + Beta2*float(NuSigma2) + Beta3
    if Mol in ExpDensityDic.keys():
        Dic['Density(g/cm^3)'] = ExpDensityDic[Mol]
    Dic['Density(g/cm^3)'] = '%.3f' % Dic['Density(g/cm^3)']

    return Dic


def KJComput(Density, HOF_S, Mass, AtomCountDic):
    Dic = {}
    for Atom in ['C', 'H', 'O', 'N']:
        if Atom not in AtomCountDic.keys():
            AtomCountDic[Atom] = 0
    if Density and HOF_S and Mass and AtomCountDic:
        Mass = float(Mass)
        HOF_S = float(HOF_S)
        Density = float(Density)
        if (AtomCountDic['O'] > (2 * AtomCountDic['C']+0.5 * AtomCountDic['H'])):
            Ngas = (AtomCountDic['H'] + 2 * AtomCountDic['N'] + 2 * AtomCountDic['O']) / (4 * Mass)
            Mgas = 1 / Ngas
            Q = (28.96 * AtomCountDic['H'] + 94.05 * AtomCountDic['C'] + HOF_S / 4.186) * 1000 / Mass
        elif (0.5 * AtomCountDic['H'] > AtomCountDic['O']):
            Ngas = (AtomCountDic['H'] + AtomCountDic['N']) / (2 * Mass)
            Mgas = (2 * AtomCountDic['H'] + 28 * AtomCountDic['N'] + 32 * AtomCountDic['O']) / (AtomCountDic['H'] + AtomCountDic['N'])
            Q = (57.8 * AtomCountDic['O'] + HOF_S / 4.186) * 1000 / Mass
        else:
            Ngas = (AtomCountDic['H'] + 2 * AtomCountDic['N'] + 2 * AtomCountDic['O']) / (4 * Mass)
            Mgas = (56 * AtomCountDic['N'] + 88 * AtomCountDic['O'] - 8 * AtomCountDic['H']) / (AtomCountDic['H'] + 2 * AtomCountDic['N'] + 2 * AtomCountDic['O'])
            Q = (28.96 * AtomCountDic['H'] + 94.05 * (AtomCountDic['O'] / 2 - AtomCountDic['H'] / 4) + HOF_S / 4.186) * 1000 / Mass # Q unit cal/g
        Dic['DetoV(L/kg)'] = 22.4*Ngas*1000 # L/kg
        Dic['DetoT(K)'] = Dic['DetoV(L/kg)']*Q*4.186/1000
        Dic['DetoD(m/s)'] = 1.01*math.sqrt(Ngas*math.sqrt(Mgas*Q))*(1 + 1.3*Density) * 1000
        Dic['DetoP(GPa)'] = 1.588*Density**2*Ngas*math.sqrt(Mgas*Q)
        Dic['DetoQ(J/g)'] = Q*4.186 #J/g
        Dic['DetoV(L/kg)'] = '%d' % Dic['DetoV(L/kg)']
        Dic['DetoT(K)'] = '%d' % Dic['DetoT(K)']
        Dic['DetoD(m/s)'] = '%d' % Dic['DetoD(m/s)']
        Dic['DetoP(GPa)'] = '%.1f' % Dic['DetoP(GPa)']
        Dic['DetoQ(J/g)'] = '%d' % Dic['DetoQ(J/g)']

    return Dic


def GetNitroList(LinkList, AtomTypeList):
    Dic = {}
    NitroList = []
    for Self in range(0, len(AtomTypeList)):
        if AtomTypeList[Self] != 'N' or len(LinkList[Self]) != 3:
            continue
        TailO = []
        for Neighbor in LinkList[Self]:
            if AtomTypeList[Neighbor] != 'O':
                continue
            if len(LinkList[Neighbor]) < 2:
                TailO.append(int(Neighbor))
        if len(TailO) ==2:
            NitroList.append([Self]+TailO)
    Dic['NitroList'] = NitroList

    return Dic


def GetWeakestNitroCharge(NitroList, ChargeList):
    Dic = {}
    NitroChargeList = []
    if NitroList and ChargeList:
        for Nitro in NitroList:
            NitroCharge = 0
            for Atom in Nitro:
                NitroCharge += float(ChargeList[Atom])
            NitroChargeList.append(NitroCharge)
        WeakestNitroCharge = max(NitroChargeList)
        Dic['WeakestNitroCharge'] = WeakestNitroCharge

    return Dic


def GetBDE(OptLevel, Mol):
    Dic = {}
    try:
        MolEnthalpy = GetEnthalpy(OptLevel + '.' + Mol + '.log')
        NO2Enthalpy = GetEnthalpy(OptLevel+'.NO2.R.log')
        FileList = os.listdir('featureCal_Select/Totianjie0811/')
        BDEList = []
        for file in FileList:
            if re.search(OptLevel+'\.'+Mol+'\..*\.log', file, re.I):
                FragmentEnthalpy = GetEnthalpy(file)
                BDEList.append(2625.51*(FragmentEnthalpy+NO2Enthalpy-MolEnthalpy))
        Dic['BDE'] = min(BDEList)
        Dic['BDE'] = '%.1f' % Dic['BDE']
    except:
        pass

    return Dic


def GetEnthalpy(log):
    with open(log, 'r') as LOG:
        LOGBlock = LOG.read()
        Enthalpy = (re.findall('Sum of electronic and thermal Enthalpies=\s+(.*?)\s+', LOGBlock))[-1]

    return float(Enthalpy)


def getXYZ(filePath):
    x = []
    y = []
    z = []
    file = open(filePath, "r")
    lines = file.readlines()
    file.close()
    if filePath.endswith(".mol"):
        atomNum = int(lines[3][:3])
        for i in range(4, atomNum + 4):
            line = lines[i]
            x.append(float(line[:10]))
            y.append(float(line[10:20]))
            z.append((float(line[20:30])))
    else:
        if filePath.endswith(".log"):
            startLine = 0
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].count("Standard orientation"):
                    startLine = i + 5
                    break
            for i in range(startLine, len(lines)):
                if lines[i].endswith("--\n"):
                    break
                contentList = [s for s in lines[i].strip().split(" ") if s]
                x.append(float(contentList[3]))
                y.append(float(contentList[4]))
                z.append(float(contentList[5]))
        else:
            raise Exception("unsupported file type")

    return x, y, z


def max_rij(x, y, z):
    atomNum = len(x)
    max_rij = 0
    for i in range(atomNum - 1):
        for j in range(i + 1, atomNum - 1):
            r_ij = sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2)
            max_rij = r_ij if r_ij > max_rij else max_rij

    return max_rij


def min_r(Seed, NumSwarm, cutoff, x, y, z):
    loop = len(Seed)
    score = [9999999]*NumSwarm
    best = 9999999
    iter = 0
    sw = [1]*NumSwarm*loop
    pi = [1]*NumSwarm*loop
    pg = [1]*4
    velocity = [0]*NumSwarm*loop
    keepbest = 0
    for i in range(NumSwarm):
        for j in range(loop):
            sw[i * loop + j] = Seed[j] * random.uniform(-1, 1)
    while ((iter < 1000) and (keepbest < cutoff)):
        refresh = 0
        for i in range(NumSwarm):
            sum = 0
            a = sw[loop * i]
            b = sw[loop * i + 1]
            c = sw[loop * i + 2]
            d = sw[loop * i + 3]
            for j in range(loop):
                sum = sum + (a * x[j] + b * y[j] + c * z[j] + d) ** 2 / (a ** 2 + b ** 2 + c ** 2)
            pscore = sqrt(sum)
            if pscore < (score[i]-0.01):
                score[i] = pscore
                pi[loop * i] = a
                pi[loop * i + 1] = b
                pi[loop * i + 2] = c
                pi[loop * i + 3] = d
            if pscore < (best - 0.01):
                best = pscore
                pg[0] = a
                pg[1] = b
                pg[2] = c
                pg[3] = d
                refresh += 1
            for k in range(loop):
                velocity[i * loop + k] = 0.0 * velocity[i * loop + k] + 2 * random.uniform(0, 1) * (
                        pg[k] - sw[i * loop + k]) + 2 * random.uniform(0, 1) * (pi[i * loop + k] - sw[i * loop + k])
                sw[i * loop + k] = sw[i * loop + k] + velocity[i * loop + k]
        if refresh > 0:
            keepbest = 0
        else:
            keepbest += 1
        iter += 1

    return best



def writeDicts2Csv(dicts, loc, NoNoneValues=False,card = "a"):

    descList = [ "Mass", "Volume", "Ro(g/cm^3)", "Area", "PosArea", "NegArea", "Ave", "PosAve"
        , "NegAve", "Sigma2", "PosSigma2", "NegSigma2", "Nu", "NuSigma2", "Pi", "MPI", "NonPolarArea",
                "PolarArea", "Sphericity", "HOF_G", "HSub", "WeakestNitroCharge", "Dipole", "Plat",
    "Integrals","MPP","SDP","FarthestDis","Diameter","Radius","ThreeSides",
                 ]
    propList= ["Density(g/cm^3)", "HOF_S(kJ/mol)", "DetoD(m/s)", "DetoP(GPa)", "DetoQ(J/g)", "DetoV(L/kg)", "DetoT(K)", "OB2", "BDE"]
    keysW = ["ID", "Formula",]
    if card == "a":
        keysW.extend(propList + descList)
    elif card == "d":
        keysW.extend( descList)
    else:
        keysW.extend(propList )
    keysWset = set(keysW)
    def checkDirExsist(path):
        def getDirNameByPath(f):
            (fileDir, tempfilename) = os.path.split(f)
            (filename, extension) = os.path.splitext(tempfilename)
            return fileDir
        Destdir = getDirNameByPath(path)
        if Destdir and not os.path.isdir(Destdir):
            os.makedirs(Destdir)
        return path
    def intOrFlt(value):
        if type(value)==type("") and not value:return "***"
        if type(value) == type([]):return [intOrFlt(i) for i in value]
        try:
            return float(value)
        except Exception:
            return value
        try:
            return int(value)
        except Exception:
            return value
    columnNames = set([])
    res = []
    # construct header and filter None values out
    if type(dicts) == type({}):
        dicts = [dicts]
    for row in dicts:
        keys = set(row.keys())
        columnNames = columnNames | keys
    columnNames = [i for i in keysWset if i in columnNames]
    for row in dicts:
        continueFlag = 0
        values = row.values()
        if NoNoneValues:
            for i in values:
                if i is None:
                    continueFlag = 1
        if continueFlag:
            continue
        keysDicT = list(row.keys())
        for keyV in keysDicT:
            if keyV not in columnNames:
                row.pop(keyV)
        resDic = {key: intOrFlt(item) for key, item in row.items()}
        blankKV = {key:"***" for key in keysWset if key not in resDic.keys()}
        resDic.update(blankKV)
        res.append(resDic)
    checkDirExsist(loc)
    # with open(loc, 'w', newline='', encoding='utf_8_sig') as f:
    with open(loc, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keysW)
        writer.writeheader()
        writer.writerows(res)

def GetPlat(infile):
    NumSwarm = 1000
    Seed = [1000, 1000, 1000, 1000]
    cutoff = 5
    x, y, z = getXYZ(infile)
    if len(x) < 4:
        plat = 1
    else:
        signal = min_r(Seed, NumSwarm, cutoff, x, y, z) / max_rij(x, y, z)
        if signal < 0.01:
            plat = 1
        else:
            plat = 0
    return {"Plat":plat}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--card', type=str, help='card = a(ll)/p(roperties)/d(escriptors)', default="a")
    parser.add_argument('-O', '--OptLevel', type=str, help='short level for Opt calculation', default="B3LYP6-31Gdp")
    # parser.add_argument('-O', '--OptLevel', type=str, help='card = full/properties/descriptors', default="B3LYP6-31Gdp")
    parser.add_argument('-H', '--HOFLevel', type=str, help='short level for HOF calculation', default="CBS-4M")
    parser.add_argument('-out', '--out', type=str, help='out = *.csv', default="out.csv")
    parser.add_argument('-noNone', '--noNone', type=bool, help='True or False,if True ,None Value Row be killed', default=False)
    parser.add_argument('-test', '--test1', type=str, help='test from err csv', default="")
    args = parser.parse_args()
    #HOFLevel = 'CBS-4M'
    #OptLevel = 'B3LYP6-31Gdp'
    MolList = GetMolList('./')
    if args.test1:
        import utils.util_doc.utils_csv  as ucsv
        # MolList = [i  for i in MolList if i in ucsv.getValueByColumns(args.test1,"ID")]
        MolList = ["DAAF_DXGGVCCZDGLQHW_UHFFFAOYSA_N"]
    Dict2Write = []
    for Mol in tqdm(MolList[:]):
        Dic = {"ID":Mol}
        # print(Mol)
        # print(args.OptLevel + '.' + Mol + '.log')
        try:
            Dic.update(ReadCBS(args.HOFLevel+'.'+Mol+'.log'))
        except Exception as e:
            #print(e)
            pass
        try:
            Dic.update(ReadWFN(args.OptLevel+'.'+Mol+'.wfn'))
        except Exception as e:
            print(e)
            pass
        try:
            Dic.update(ReadOpt(args.OptLevel+'.'+Mol+'.log'))
        except Exception as e:
            print(e)
            pass
        try:
            Dic.update(GetOB(Dic['Mass'], Dic['AtomCountDic']))
        except Exception as e:
            #print(e)
            pass
        try:
            Dic.update(GetLinkList(Dic['AtomTypeList'], Dic['XList'], Dic['YList'], Dic['ZList']))
        except Exception as e:
            #print(e)
            pass
        try:
            Dic.update(GetHSub(Dic['Area'], Dic['NuSigma2']))
        except Exception as e:
            # print(e)
            pass

        try:
            Dic.update(GetPlat(args.OptLevel + '.' + Mol + '.log'))
        except Exception as e:
            #print(e)
            pass
        try:
            Dic.update(GetHOF_S(Dic['HOF_G'], Dic['HSub']))
        except Exception as e:
            #print(e)
            pass
        try:
            Dic.update(GetDensity(Dic['Ro(g/cm^3)'], Dic['NuSigma2'], Mol))
        except Exception as e:
            #print(e)
            pass
        try:
            Dic.update(KJComput(Dic['Density(g/cm^3)'], Dic['HOF_S'], Dic['Mass'], Dic['AtomCountDic']))
        except Exception as e:
#            print(e)
            pass
        try:
            Dic.update(GetNitroList(Dic['LinkList'], Dic['AtomTypeList']))
        except Exception as e:
            #print(e)
            pass
        try:
            Dic.update(GetWeakestNitroCharge(Dic['NitroList'], Dic['ChargeList']))
        except Exception as e:
            print("GetWeakestNitroCharge",e)
            pass
        try:
            Dic.update(GetBDE(args.OptLevel, Mol))
        except Exception as e:
            #print(e)
            pass
        Dict2Write.append(Dic)
    writeDicts2Csv(Dict2Write,args.out,args.noNone,args.card)
