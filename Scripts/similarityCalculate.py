from rdkit import Chem, DataStructs
import pandas as pd
import numpy as np
import utils.util_doc.utils_csv as ucsv
import utils.util_rdkit.util_rdkit as urd
from tqdm import tqdm


def parseDataSheet(input_file, mol_path):
    if input_file.split('.')[-1] == 'csv':
        data_sheet = pd.read_csv(input_file)
    else:
        data_sheet = pd.read_excel(input_file)
    mols_list = data_sheet['molname'].tolist()
    rdMol = [Chem.MolFromMolFile(mol_path + f'{mol}.mol', removeHs=False) for mol in mols_list]
    fps = [Chem.RDKFingerprint(mol) for mol in rdMol]
    return fps
def getMolAddH(smi):
    test_mol = Chem.MolFromSmiles(smi)
    test_mol = Chem.AddHs(test_mol)
    return test_mol
# 相似度 取最大相似度
def tanimoto_similarity(csvTest,csvTrain,outCsv,type="avg"):
    train_smis = pd.read_csv(csvTrain)['SMILES'].tolist()
    test_smis = pd.read_csv(csvTest)['SMILES'].tolist()
    tanimoto = []
    tanimotoavg = []
    for test_smi in test_smis:
        test_mol = getMolAddH(test_smi)
        test_fp = Chem.RDKFingerprint(test_mol)
        # test_fp = urd.fingerPrint(mol=test_mol)
        similarity_list = []
        for train_smi in train_smis:
            train_mol = getMolAddH(train_smi)
            train_fp = Chem.RDKFingerprint(train_mol)
            # train_fp = urd.fingerPrint(mol=train_mol)
            similarity = DataStructs.FingerprintSimilarity(test_fp, train_fp)
            if similarity == 1:
                print(test_smi,train_smi)
                exit(-8)
            similarity_list.append(similarity)
        similarity_max = np.max(similarity_list)
        similarity_avg = np.mean(similarity_list)
        tanimoto.append(similarity_max)
        tanimotoavg.append(similarity_avg)
        print(test_smi, similarity_max,similarity_avg)
    df = pd.read_csv(csvTest)
    df['tanimoto_max'] = tanimoto
    df['tanimoto_avg'] = tanimotoavg
    print(df)
    ucsv.writeDataFrame2Csv(df,outCsv+'GUBEN_Ratio2.csv')
    # df.to_csv(outCsv+'GUBEN_Ratio2.csv', index=False)
    return np.mean(tanimoto),np.std(tanimoto)


def main():
    test_fps_list = parseDataSheet(input_file='../datasets/测试集.xlsx', mol_path='../MSMol/')
    train_fps_list = parseDataSheet(input_file='../datasets/训练集.xlsx', mol_path='../MSMol/')
    count = 0
    max_similarity_list = []
    for test_fps in test_fps_list:
        similarity_list = []
        for train_fps in train_fps_list:
            similarity = DataStructs.FingerprintSimilarity(test_fps, train_fps)
            similarity_list.append(similarity)
        similarity_max = np.max(similarity_list)
        if similarity_max == 1:
            print(test_fps_list.index(test_fps), train_fps_list.index(train_fps))
        max_similarity_list.append(similarity_max)
        count += 1
    #     print(count, similarity_max, len(similarity_list))
    # print(max_similarity_list)
    # print(np.mean(max_similarity_list))
    # print(np.std(max_similarity_list))
    # df = pd.read_csv('../测试集.csv')
    # df['similarity_max'] = max_similarity_list
    # # print(df)
    # df.to_csv('../相似度数据.csv', index=False)

def checkDuplicate(csv1,csv2):

    smiles1 = ucsv.getValueByColumns(csv1,"SMILES")
    smiles2 = ucsv.getValueByColumns(csv2,"SMILES")
    set1 = set(smiles1)
    set2 = set(smiles2)
    if len(set1) != len(smiles1):
        print("csv1 duplicate")
    if len(set2) != len(smiles2):
        print("csv2 duplicate")
    if set1 & set2 :
        print("csv1 & csv2 duplicate")
    return
if __name__ == "__main__":
    # main()
    # 分解温度实现在runDelete
    csv1 = r"dataTDEC\bestTest_delete3\test.csv"
    csv2 = r"dataTDEC\bestTest_delete3\train.csv"

    # similarityAvg
    # 0.7676603948517132, similarityStd
    # 0.13093133425634818
    csv1 = r"C:\Users\jaysnow\PycharmProjects\vitualScreen\data\csv\FinalDataSet\BestTestSetDensity.csv"
    csv2 = r"C:\Users\jaysnow\PycharmProjects\vitualScreen\data\csv\FinalDataSet\BestTrainSetDensity.csv"
    checkDuplicate(csv1,csv2)

    similarityAvg, similarityStd= tanimoto_similarity(csv1, csv2, r"data\csv\FinalDataSet/draw/", "avg")
    print("similarityAvg {}, similarityStd {}".format(similarityAvg, similarityStd))
