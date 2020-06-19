import numpy as np
from utils import *
import pickle
from copy import deepcopy
import configparser
from multiprocessing import Pool
from dualBased import dualSolve
import multiprocessing

# READ CONFIG
Config = configparser.ConfigParser()
Config.read("params.conf")

stepFactor = 1000  # gradient steps would be 1/stepFactor


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


def stepGuidedGradient(data, problem=None, purpose=None):
    currCoef = dualSolve(data, problem=problem, purpose="init")
    originalCoef = deepcopy(currCoef)

    iteration = 0
    epsilon = 0  # TODO: declare convergence gap
    updatedData = deepcopy(data['incidentData'])

    resultName = ''  # TODO: declare the file name for storing results
    maxIter = 20  # TODO: set maximum number of iterations for the gradient algorithm

    # while likelihood does not suffer
    with open(resultName, 'w+') as f:
        while iteration < maxIter:
            f.write("Iteration : {}\n".format(iteration))
            inputs = []
            for tempData in updatedData:
                # set scale to true
                inputs.append([tempData, currCoef, data['df'], data['neighborGraph']])

            coreCount = multiprocessing.cpu_count()
            pool = Pool(coreCount - 2)  # leave two cores
            results = pool.map(getShiftsSplit, inputs)
            pool.close()
            pool.join()

            # aggregate results
            updatedDF = deepcopy(data['df'])
            # TODO: mark each shift in the updated df set using 'results'
            # TODO: create training data based on shifts
            updatedCountData = None

            '''Take Gradient Step'''
            # create vector of times (x), features (w). Can also be numpy arrays
            times = []
            features = []
            oldCoef = deepcopy(currCoef)

            if problem == "poisson":
                currCoef = doGradientStepPoissonVector(times, features, currCoef)
                updatedCountData = None  # TODO: create training data based on shifts
                newLikelihood = getTotalLikelihoodPoisson(updatedCountData, currCoef)
                likelihoodPriorDefender = getTotalLikelihoodPoisson(updatedCountData, oldCoef)

            elif problem == "logistic":
                currCoef = doGradientStepLogisticVectorized(times, features, currCoef)
                updatedCountData = None  # TODO: create training data based on shifts
                newLikelihood = getTotalLikelihoodLogistic(updatedCountData, currCoef)
                likelihoodPriorDefender = getTotalLikelihoodLogistic(updatedCountData, oldCoef)

            f.write(
                "Likelihood at iteration {} before gradient step is {}\n".format(iteration, likelihoodPriorDefender))
            f.write(
                "Likelihood at iteration {} after graident step is {}\n".format(iteration, newLikelihood))

            gap = likelihoodPriorDefender - newLikelihood

            if newLikelihood > likelihoodPriorDefender and abs(gap) < epsilon:
                f.flush()
                break

            iteration += 1

    return currCoef, originalCoef


def doGradientStepPoissonVector(x, y, theta):
    x = np.asarray(x)
    y = np.asarray(y)
    y = y.reshape(-1, 1)
    theta = np.asarray(theta)
    theta = theta.reshape(-1, 1)

    grad = x.T @ (y - np.exp(x @ theta))
    theta = theta + (1 / stepFactor) * grad
    return list(theta.flat)


def doGradientStepLogisticVectorized(x, y, theta):
    x = np.asarray(x)
    y = np.asarray(y)
    y = y.reshape(-1, 1)
    theta = np.asarray(theta)
    theta = theta.reshape(-1, 1)

    grad = x.T @ (y - sigmoid(x @ theta))
    theta += (1 / stepFactor) * grad
    return list(theta.flat)
