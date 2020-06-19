# data access file for robust survival model
import configparser
from math import floor, ceil
from numpy.random import choice
import math
import pandas
import multiprocessing
from multiprocessing import Pool
import numpy as np

# READ CONFIG
Config = configparser.ConfigParser()
Config.read("params.conf")


def ConfigSectionMap(Config, section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                print(("skip: %s" % option))
        except:
            print(("exception on %s!" % option))
            dict1[option] = None
    return dict1


# definitions
exponLikelihood = lambda scale, time: np.log(1 / float(scale)) * np.exp(-time / float(scale)) if float(
    scale) != 0 else 0


def preProcess(crime, args):
    """
    Implement method to setup train and test data
    Returns train and test Data (list of list), raw incident data (list of list) and data in dataframe format.
    Raw data is only the list of incidents. Train and test data are appended with features and must have the
    appropriate random variable as one of the columns
    """
    dict_return = {'incidentRawDataTrain': None, 'trainingDataTrain': None, 'dfTrain': None,
                   'incidentRawDataTest': None, 'trainingDataTest': None, 'dfTest': None}

    return dict_return


# return event features used in the survival model for a given event
def getEventDynamicFeatures(time, grid, dataFrame):
    '''
    For a given time and location in the grid, return incident features. Use dataframe to create features based
    on past incidents
    '''
    # TODO: create appropriate features for the incident, e.g, weather, past incidents, etc.
    return None


# wrapper around regression formula
def getRegressionFormula(case="default"):
    rFormula = "Surv(time,death) ~ temp + rain + season + weekend + timezone + pawn + liq + liqret + homeless + " \
               "pastGrid2 + pastGridWeek + pastGridMonth + pastNeighbor2 + pastNeighborWeek + pastNeighborMonth"

    return rFormula


# for a set of features - return the scale
def getScaleGivenFeatures(features, coef):
    scalePower = 0
    for counter in range(len(coef)):
        scalePower += coef[counter] * features[counter]
    return np.exp(scalePower)


def getTotalLikelihoodLogistic(data, coef):
    logL = 0
    for row in data:
        y = row[2]
        features = row[3:]
        thetaW = 0
        for counterFeature in range(len(features)):
            thetaW += features[counterFeature] * coef[counterFeature]
        # sigmaThetaW = 1/float(1+np.exp(-1*thetaW))
        # tempL = y*np.log(sigmaThetaW) + (1-y)*np.log(1-sigmaThetaW)
        # print tempL
        tempL = 0
        tempL += y * thetaW
        tempL -= np.log(1 + np.exp(thetaW))
        logL += tempL

    return logL


def getTotalLikelihoodPoisson(data, coef):
    logL = 0
    for row in data:
        tempRow = row[3:]
        tempCount = row[2]
        temp = 0
        thetaW = 0
        for i in range(len(tempRow)):
            thetaW += tempRow[i] * coef[i]
        temp += tempCount * thetaW
        temp -= np.exp(thetaW)
        temp -= np.log(np.math.factorial(tempCount))
        logL += temp

    return logL


# updated likelihood calculation - not according to R documentation, according to IJCAI and ICCPS
def getTotalLikelihoodUpdated(data, coef):
    logL = 0
    for row in data:
        tempRow = row[3:]
        tempTime = row[2]
        temp = 0
        temp += np.log(tempTime)
        for featureCounter in range(len(tempRow)):
            temp -= coef[featureCounter] * tempRow[featureCounter]

        expTerm = np.exp(temp)
        temp -= expTerm
        # temp += alpha
        logL += temp

    return logL


# get total likelihood for data
def getTotalLikelihood(data, coef):
    # if cluster is set to None, then likelihood of the whole model is evaluated.
    # get batch test data likelihood
    # method could be one of survival, logistic or Poisson

    logL = 0
    for row in data:
        tempRow = row[2:]  # keep the death value - use it to multiply the intercept
        tempScale = getScaleGivenFeatures(tempRow[1:], coef)
        timeTemp = tempRow[0]
        if np.isnan(tempScale):
            continue
        tempLikelihood = exponLikelihood(tempScale, timeTemp)
        if tempLikelihood == -float("Inf"):
            continue
        # print templ
        logL += tempLikelihood

    return logL


def second_smallest(numbers):
    m1, m2 = float('inf'), float('inf')
    for x in numbers:
        if x <= m1:
            m1, m2 = x, m1
        elif x < m2:
            m2 = x
    return m2


def convertToDistribution(dist):
    sumDist = sum(dist)
    for counter in range(len(dist)):
        dist[counter] /= float(sumDist)

    # check if dist sums to 1
    if sum(dist) != 1 and len(dist) > 0:
        diff = 1 - sum(dist)
        dist[0] += diff
    return dist


def getNeighbors(cell):
    # TODO: code a dictionary with cell_id --> neighbors
    pass


def generateRandomGraph(validGrids, fraction=1):
    """
    Method used to create a random graph of neighbors. If fraction is 1, all neighbors are returned
    """
    neighbors = {}
    for tempGrid in validGrids:
        tempNeighbors = getNeighbors(tempGrid)
        dist = [tempNeighbors]
        dist = convertToDistribution(dist)
        draw = list(choice(tempNeighbors, int(ceil(len(tempNeighbors) * fraction)), replace=False, p=dist))
        neighbors[tempGrid] = draw

    return neighbors


def getShiftsSplitPoisson(input):
    # TODO: parse input
    # input must contain the following:
    # tempGrid = input[0] # current cell
    # coef = input[1] # current defender parameters
    # df = input[2] # dataframe of incidents
    # neighborGraph = input[3] # neighborhood structure
    # tempBin = input[4] # current time bin
    tempBin, tempGrid, coef, df, neighborGraph = None, None, None, None, None

    tempNeighbors = neighborGraph[tempGrid]
    countNeighbors = {i: 0 for i in tempNeighbors}
    # TODO: create feature map for each of the neighbors for this bin
    features = {}
    for i in tempNeighbors:
        countNeighbors[i] = len(df.loc[(df['grid'] == i) & (df['bin'] == tempBin)])

    try:
        # expected format - currGrid,coef,options,countsPerGrid,df,currTime,scale=False
        choices = getRunningLikelihoodForGridsPoisson(options=tempNeighbors, df=df, featureMap=features, coef=coef,
                                                      bin=tempBin)
        pick = min(choices, key=lambda key: choices[key])

    except ValueError:
        pick = tempGrid

    return pick

def getShiftsSplitLogistic(input):
    # TODO: Create same structure as shiftsplit Poisson and pass params
    choices = getRunningLikelihoodForGridsLogistic()
    pick = min(choices, key=lambda key: choices[key])
    return pick


def getShiftsSplit(input, problem):
    coreCount = multiprocessing.cpu_count()
    pool = Pool(coreCount - 2)  # leave two cores
    if problem == "Logistic":
        results = pool.map(getShiftsSplitLogistic(), input)
    elif problem == "Poisson":
        results = pool.map(getShiftsSplitPoisson(), input)
    pool.close()
    pool.join()
    return results


def getRunningLikelihoodForGridsLogistic(options, df, featureMap, coef):
    choices = {}
    for tempGrid in options:
        y = 1  # we want to measure the chances of events happening at all grids

        # get likelihood
        thetaW = 0
        for counterFeature in range(len(featureMap[tempGrid])):
            thetaW += featureMap[tempGrid][counterFeature] * coef[counterFeature]

        tempL = 0
        tempL += y * thetaW
        tempL -= np.log(1 + np.exp(thetaW))
        choices[tempGrid] = tempL

    return choices


def getRunningLikelihoodForGridsPoisson(options, df, featureMap, coef, bin):
    choices = {}
    for tempGrid in options:
        # updated count if the point moves to the grid
        # tempCount = <select count of incidents from df at current time bin> + 1
        tempCount = 0

        # get likelihood
        tempL = 0
        thetaW = 0
        for counterFeature in range(len(featureMap[tempGrid])):
            thetaW += featureMap[tempGrid][counterFeature] * coef[counterFeature]
        tempL += tempCount * thetaW
        tempL -= np.exp(thetaW)
        tempL -= np.log(np.math.factorial(tempCount))

        choices[tempGrid] = tempL

    return choices


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def getResultFileName(args):
    if args.algorithm == "gradient":
        resultName = os.getcwd() + "/logs/" + args.model + "_" + args.crime + "_area=" + args.area + "_gradient" + "_step=" + args.step + "_slice=" + args.slice + "_neighbor=" + args.neighbors + "_fraction=" + args.fractionNeighborTest
    elif args.algorithm == "dual":
        resultName = os.getcwd() + "/logs/" + args.model + "_" + args.crime + "_area=" + args.area + "_dual" + "_slice=" + args.slice + "_neighbor=" + args.neighbors + "_fraction=" + args.fractionNeighborTest
    else:
        resultName = None

    return resultName


def getModelSaveName(args):
    if args.algorithm == "gradient":
        modelSaveName = os.getcwd() + "/results/" + args.model + "_" + args.crime + "_area=" + args.area + "_gradient" + "_step=" + args.step + "_slice=" + args.slice + "_neighbor=" + args.neighbors + "_fraction=" + args.fractionNeighborTrain
    elif args.algorithm == "dual":
        modelSaveName = os.getcwd() + "/results/" + args.model + "_" + args.crime + "_area=" + args.area + "_dual" + "_slice=" + args.slice + "_neighbor=" + args.neighbors + "_fraction=" + args.fractionNeighborTrain
    else:
        modelSaveName = None

    return modelSaveName
