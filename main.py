'''
Main File for Robust Spatial-Temporal Incident Prediction
'''
import logging
import configparser
from utils import *
from copy import deepcopy
from gradientBased import stepGuidedGradient
import sys
import os
import argparse
import pickle
import os
from dualBased import dualSolve


# READ CONFIG
Config = configparser.ConfigParser()
Config.read("params.conf")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', help='primary algorithm to use - one of dual or gradient')
parser.add_argument('--neighbors', help='attacker budget')
parser.add_argument('--model', help='is one of survival, poisson or logistic depending upon the type of robust model')

# GLOBALS
updatedSurvivalData = []
validGrids = []


# METHODS
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


def main(args, resultName, saveName):
    global updatedSurvivalData

    # set neighbor levels
    args.neighbors = float(args.neighbors)
    data = setupProblem(args)
    # get random neighbor graph
    neighborGraphTrain = generateRandomGraph(validGrids)

    data['neighborGraph'] = neighborGraphTrain
    if args.algorithm == 'gradient':

        # Read adGrad specific parameters
        maxIter = int(ConfigSectionMap(Config, "adGrad")["maxiter"])

        # do step guided-gradient descent
        coefRobust, coefNormal = stepGuidedGradient(data, problem='Survival')
        with open(saveName + ".pickle", 'wb') as f:
            pickle.dump([coefRobust, coefNormal], f)

    elif args.algorithm == 'dual':
        maxIter = int(ConfigSectionMap(Config, "dual")["maxiter"])
        # try:
        if args.model == "survival":
            coefRobust, coefNormal = dualSolve(data, problem='Survival')
        elif args.model == "poisson":
            # updatedCountData, incidentData, df, resultName, args, startDate, endDate,
            # neighborGraph = None, maxIter = 20, purpose = None, epsilon = 10
            coefRobust, coefNormal = dualSolve(data, problem='Poisson')

        elif args.model == "logistic":
            coefRobust, coefNormal = dualSolve(data, problem='Logistic')
            with open(saveName + ".pickle", 'wb') as f:
                pickle.dump([coefRobust, coefNormal], f)


def setupProblem(debugMode, crime):
    # setup problem data
    preProcessedData = preProcess(
        crime, args)
    # TODO: Format according to what is expected from setup problem.
    return preProcessedData


def setDefaults(args):
    # set defaults
    if args.crime is None:
        args.crime = "poaching"

    if args.neighbors is None:
        args.neighbors = '1'


if __name__ == "__main__":
    args = parser.parse_args()

    if args.algorithm is None:
        print('Algorithm name not provided')
        sys.exit()
    else:
        resultFileName = getResultFileName(args)
        saveName = getModelSaveName(args)

    main(args, resultFileName, saveName)
