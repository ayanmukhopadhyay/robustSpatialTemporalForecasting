
from copy import deepcopy
import cvxpy as cp
from utils import *


class poissonProblemVector:
    def __init__(self, numFeatures):
        self.numFeatures = numFeatures
        self.delta = cp.Variable()
        self.theta = cp.Variable(numFeatures)
        self.constraints = []

        # define objective
        self.obj = cp.Maximize(self.delta)
        self.formulation = cp.Problem(self.obj, self.constraints)

    def addConstraint(self, updatedData):
        """
        Adds a constraint based on the current best response of the attacker.
        :param updatedData: Set of manipulations chosen by the attacker at current time step
        :return:
        """
        x = None  # TODO: retrive x from updated data.
        y = None  # TODO: retrive y from updated data.
        x = np.asarray(x)
        y = np.asarray(y)
        logFact = sum([np.log(np.math.factorial(i)) for i in y])
        cons = self.delta
        tempL = cp.sum(cp.multiply(y, x @ self.theta) - cp.exp(x @ self.theta)) - logFact
        cons -= tempL
        self.constraints.append(cons <= 0)

    def redefineProblem(self):
        """
        Updates the problem with the latest set of constraints
        :return: _
        """
        self.formulation = cp.Problem(self.obj, self.constraints)

    def solve(self):
        """
        Solve the current optimization problem. If SCS returns an error, try without an explicit solver.
        :return: _
        """
        print("Attempting to solve problem instance with {} constraints".format(len(self.constraints)))
        self.formulation.solve(solver='SCS')
        print(self.formulation.status)


class logisticProblemVector:
    def __init__(self, numFeatures):
        self.numFeatures = numFeatures

        self.delta = cp.Variable()
        self.theta = cp.Variable(numFeatures)
        self.constraints = []

        # objective
        self.obj = cp.Maximize(self.delta)
        self.formulation = cp.Problem(self.obj, self.constraints)

    def addConstraint(self, updatedData):
        """
        Adds a constraint based on the current best response of the attacker.
        :param updatedData: Set of manipulations chosen by the attacker at current time step
        :return: _
        """
        x = None  # TODO: retrive x from updated data.
        y = None  # TODO: retrive y from updated data.

        x = np.asarray(x)
        y = np.asarray(y)
        cons = self.delta
        l = cp.sum(cp.multiply(y, x @ self.theta) - cp.logistic(x @ self.theta))
        cons -= l
        self.constraints.append(cons <= 0)

    def redefineProblem(self):
        """
        Updates the problem with the latest set of constraints
        :return: _
        """
        self.formulation = cp.Problem(self.obj, self.constraints)

    def solve(self):
        """
        Solve the current optimization problem. If SCS returns an error, try without an explicit solver.
        :return: _
        """
        print("Attempting to solve problem instance with {} constraints".format(len(self.constraints)))
        self.formulation.solve(solver='SCS')
        print(self.formulation.status)


class survivalProblem:

    def __init__(self, numFeatures):
        self.numFeatures = numFeatures
        self.constraintDataHolder = []
        self.timesHolder = []

        self.theta = cp.Variable()
        self.w = cp.Variable(numFeatures)
        self.constraints = []

        # define objective
        self.obj = cp.Maximize(self.theta)
        self.formulation = cp.Problem(self.obj, self.constraints)

    def addConstraint(self, updatedData):
        data = [x[3:] for x in updatedData]
        times = [x[2] for x in updatedData]
        cons = self.theta
        for counterData in range(len(data)):
            cons -= np.log(times[counterData])

            # calculate theta times w
            thetaT = 0
            for i in range(self.numFeatures):
                thetaT += self.w[i] * data[counterData][i]

            cons += thetaT

            cons += cp.exp(cp.log(times[counterData]) - thetaT)

        self.constraints.append(cons <= 0)

    def redefineProblem(self):
        self.formulation = cp.Problem(self.obj, self.constraints)

    def solve(self):
        """
        Solve the current optimization problem. If SCS returns an error, try without an explicit solver.
        :return: _
        """
        print("Attempting to solve problem instance with {} constraints".format(len(self.constraints)))
        self.formulation.solve(solver='SCS')
        print(self.formulation.status)


def dualSolve(data, problem=None, purpose=None):
    """
    Solves the dual problem implementation for logistic regression
    :param problem: one of survival, poisson or logistic
    :param data: dictionary of updatedCountData, incident data
    :param purpose: if init, return coefficients without adversarial manipulations. This is used to initialize the
                    start value for gradient based methods
    :return: original coefficients and robust coefficients
    """
    maxIter = 20

    numFeatures = None  # get number of features from data['updatedCountData']

    if problem == "Poisson":
        optProb = poissonProblemVector(numFeatures)
    elif problem == "Logistic":
        optProb = logisticProblemVector(numFeatures)
    elif problem == "Survival":
        optProb = problem(numFeatures)

    # add constraint and redefine problem
    optProb.addConstraint(data['updatedCountData'])
    optProb.redefineProblem()
    optProb.solve()
    oldCoef = list(optProb.theta.value)
    # print(oldCoef)
    originalCoef = deepcopy(oldCoef)
    if purpose == "init":
        return oldCoef
    currCoef = oldCoef  # to enable getting shifts inside the loop, which uses currCoef

    updatedData = deepcopy(data['incidentData'])
    resultName = ''  # TODO: declare the file name for storing results

    # iteratively solve model
    with open(resultName, 'w+') as f:
        tempIter = 0
        while tempIter < int(maxIter):
            '''Create a set of inputs and pass to multiproc to get shifts in parallel'''
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

            # update the optimization problem using the attacker's best response
            optProb.addConstraint(updatedCountData)
            optProb.redefineProblem()
            optProb.solve()
            currCoef = list(optProb.theta.value)

            oldLikelihood = getTotalLikelihoodLogistic(updatedCountData, oldCoef)
            newLikelihood = getTotalLikelihoodLogistic(updatedCountData, currCoef)
            gap = oldLikelihood - newLikelihood
            f.write("Likelihood at iteration {} is {}\n".format(tempIter, oldLikelihood))
            f.write("Likelihood at iteration {} is {}\n".format(tempIter, newLikelihood))
            f.write("status:{}\n".format(optProb.formulation.status))
            f.flush()

            tempIter += 1

        return currCoef, originalCoef



