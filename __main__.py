import configparser
import numpy as np
import pandas as pd

from cluster import Clustering
from genetic import Genetic
from generation import Generation

NORMALIZATION = False


def readVars(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    budget = int(config.get("vars", "budget"))
    kmax = int(config.get("vars", "kmax"))  # Maximum number of Clusters
    numOfInd = int(config.get("vars", "numOfInd"))  # number of individual
    Ps = float(config.get("vars", "Ps"))
    Pm = float(config.get("vars", "Pm"))
    Pc = float(config.get("vars", "Pc"))

    return budget, kmax, Ps, Pm, Pc, numOfInd


# minmax normalization
def minmax(data):
    normData = data
    data = data.astype(float)
    normData = normData.astype(float)
    for i in range(1, data.shape[1]):
        print(str(i) + ' of ' + str(data.shape[1]))
        tmp = data.iloc[:, i]
        # max of each column
        maxElement = np.amax(tmp)
        # min of each column
        minElement = np.amin(tmp)

        # norm_dat.shape[0] : size of row
        for index, row in data.iterrows():
            normData.loc[index, normData.iloc[:, i].name] = float(
                data.loc[index, data.iloc[:, i].name] - minElement
            ) / (maxElement - minElement)

    normData.to_csv('result/norm_data.csv', index=None, header=None)
    return normData

if __name__ == '__main__':
    config_file = "config.txt"
    if(NORMALIZATION):
        data = pd.read_csv('data/program_only_keywords.csv')
        data = minmax(data)  # normalize
    else:
        data = pd.read_csv('result/norm_data.csv', header=None)
    print('Data normalized!')
    # size of column
    dim = data.shape[1]

    # kmeans parameters & GA parameters
    generationCount = 0
    budget, kmax, Ps, Pm, Pc, numOfInd = readVars(config_file)

    print("-------------GA Info-------------------")
    print("budget", budget)
    print("kmax", kmax)
    print("numOfInd", numOfInd)
    print("Ps", Ps)
    print("Pm", Pm)
    print("Pc", Pc)
    print("---------------------------------------")

    # dim or pattern id
    # 11k 11k 11k
    chromosome_length = kmax * dim

    #-------------------------------------------------------#
    # 							main 						#
    #-------------------------------------------------------#
    print('Setting Generation Class')
    initial = Generation(numOfInd, 0)
    print('Generating random initial chromosomes')
    initial.randomGenerateChromosomes(
        chromosome_length)  # initial generate chromosome

    print('Setting Clustering Class')
    clustering = Clustering(initial, data, kmax)  # eval fit of chromosomes

    # ------------------calc fitness------------------#
    print('Calculating initial fitness')
    generation = clustering.calcChromosomesFit()

    # ------------------------GA----------------------#
    print('Looping through each generation')
    while generationCount <= budget:
        print('Generation ' + str(generationCount) + ':')
        print('\tSetting up Genetic class')
        GA = Genetic(numOfInd, Ps, Pm, Pc, budget, data, generationCount, kmax)
        print('\tExecuting genetic process')
        generation, generationCount = GA.geneticProcess(
            generation)
        iBest = generation.chromosomes[0]
        clustering.printIBest(iBest)

    # ------------------output result-------------------#
    clustering.output_result(iBest, data)
