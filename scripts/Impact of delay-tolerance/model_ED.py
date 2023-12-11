# -*- coding: utf-8 -*-
"""
Created on 2018/3/18 下午9:39
author: Tong Jia
email: cecilio.jia@gmail.com
software: PyCharm
Description:
    An implement of ED model using gurobi callback solver. The model is based on formula (61) in paper "APS-18-3-18.pdf".
"""
import sys
import gurobipy
import numpy as np
from six.moves import xrange

def model_ED(I, J, K, L, N, p, s):
    """
    Parameters:
    ----------
    :param I: int
        The total number of available appointment positions.
    :param J: int
        The total number of the user types or categories.
    :param K: int
        The total number of scenarios of user service time matrix for training.
    :param L: float or int
        Time threshold for the last user (total service time in a period).
    :param N: ndarray
        A vector with the number of category users;
        N[j] is the number of users in category j;
        the length of N is J.
        Note:
            the sum of N[j] for j in J is I.
    :param p: ndarray
        Probabilities vector with length of K. In this proposition, we assume that user service times s^~ are identified
        as scenarios s^k with probabilities p[k];
        the length of p is K.
    :param s: ndarray
        Service time tensor;
        s[k][i][j] refers to the service time in position i for a user in category j inside the k-th scenario.

    Returns:
    -------
    :return: solutionsDict: dict
    :return: objValue: float
    """
    # Ckeck point for parameter N
    if not isinstance(N, np.ndarray):
        raise ValueError("the parameter N is not saved as ndarray")
    if (N.ndim != 1):
        raise ValueError("the parameter N is not a vector (1 dim)")
    if (N.shape[0] != J):
        raise ValueError("the length of parameter N (%d) is not equal to J (%d)" % (N.shape[0], J))
    sumN = np.sum(a=N, axis=0, dtype=int)
    if (sumN != I):
        raise ValueError("sum(N[j]) for all j is not equal to I")
    # Ckeck point for parameter p
    if not isinstance(p, np.ndarray):
        raise ValueError("the parameter p is not saved as ndarray")
    if (p.ndim != 1):
        raise ValueError("the parameter p is not a vector (1 dim)")
    if (p.shape[0] != K):
        raise ValueError("the length of parameter p (%d) is not equal to K (%d)" % (p.shape[0], K))
    sumProb = np.sum(a=p, axis=0, dtype=float)
    if (round(float(sumProb), 6) != 1):
        raise ValueError("The sum of p is not equal to 1")

        # Ckeck point for parameter s
    if not isinstance(s, np.ndarray):
        raise ValueError("the parameter s is not saved as ndarray")
    if (s.ndim != 3):
        raise ValueError("the parameter s is not a tensor (3 dim)")
    if (s.shape[0] != K):
        raise ValueError("the first dimension length of parameter s (%d) is not equal to K (%d)" % (s.shape[0], K))
    if (s.shape[1] != I):
        raise ValueError("the second dimension length of parameter s (%d) is not equal to I (%d)" % (s.shape[1], I))
    if (s.shape[2] != J):
        raise ValueError("the third dimension length of parameter s (%d) is not equal to J (%d)" % (s.shape[2], J))

    # Create a model
    model = gurobipy.Model("EM model")

    # Number of constraints
    numberConstraints1 = 0
    for i in xrange(2, I + 1):
        for t in xrange(1, i):
            for k in xrange(1, K + 1):
                numberConstraints1 += 1
    numberConstraints2 = I  # first constraint in formula (6)
    numberConstraints3 = J  # second constraint in formula (6)
    numberConstraints4 = 1  # first constraint in formula (7)
    numberConstraints5 = I - 1  # second constraint in formula (7)
    numberConstraints6 = 1  # third constraint in formula (7)

    numberConstraints = numberConstraints1 + \
                        numberConstraints2 + numberConstraints3 + \
                        numberConstraints4 + numberConstraints5 + numberConstraints6

    # Bound keys of constraints
    bkConstraints1 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints1
    bkConstraints2 = [gurobipy.GRB.EQUAL] * numberConstraints2
    bkConstraints3 = [gurobipy.GRB.EQUAL] * numberConstraints3
    bkConstraints4 = [gurobipy.GRB.EQUAL] * numberConstraints4
    bkConstraints5 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints5
    bkConstraints6 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints6

    bkConstraints = bkConstraints1 + \
                    bkConstraints2 + bkConstraints3 + \
                    bkConstraints4 + bkConstraints5 + bkConstraints6

    # Bound values of constraints
    bvConstraints1 = [0.0] * numberConstraints1
    bvConstraints2 = [1.0] * numberConstraints2
    bvConstraints3 = N.tolist()
    bvConstraints4 = [0.0] * numberConstraints4
    bvConstraints5 = [0.0] * numberConstraints5
    bvConstraints6 = [L] * numberConstraints6

    bvConstraints = bvConstraints1 + \
                    bvConstraints2 + bvConstraints3 + \
                    bvConstraints4 + bvConstraints5 + bvConstraints6

    # Number of variables
    numberVariables_gamma = (I - 1) * K  # i belongs to [2, I]
    numberVariables_x = I * J
    numberVariables_y = I
    numberVariables = numberVariables_gamma + numberVariables_x + numberVariables_y

    # Name of variables
    nameVariables_gamma = []
    for i in xrange(2, I + 1):
        for k in xrange(1, K + 1):
            name_gamma_temp = "gamma_%d_%d" % (i, k)
            nameVariables_gamma.append(name_gamma_temp)
    nameVariables_x = []
    nameVariables_y = []
    for i in xrange(1, I + 1):
        for j in xrange(1, J + 1):
            name_x_temp = "x_%d_%d" % (i, j)
            nameVariables_x.append(name_x_temp)
        name_y_temp = "y_%d" % i
        nameVariables_y.append(name_y_temp)

    nameVariables = nameVariables_gamma + nameVariables_x + nameVariables_y

    # Type of variables
    typeVariables_gamma = [gurobipy.GRB.CONTINUOUS] * numberVariables_gamma
    typeVariables_x = [gurobipy.GRB.BINARY] * numberVariables_x
    typeVariables_y = [gurobipy.GRB.CONTINUOUS] * numberVariables_y

    typeVariables = typeVariables_gamma + typeVariables_x + typeVariables_y

    # Bound values of variables
    lbVariables_gamma = [0.0] * numberVariables_gamma
    ubVariables_gamma = [+gurobipy.GRB.INFINITY] * numberVariables_gamma

    lbVariables_x = [0.0] * numberVariables_x
    ubVariables_x = [1.0] * numberVariables_x

    lbVariables_y = [0.0] * numberVariables_y
    ubVariables_y = [+gurobipy.GRB.INFINITY] * numberVariables_y

    lbVariables = lbVariables_gamma + lbVariables_x + lbVariables_y
    ubVariables = ubVariables_gamma + ubVariables_x + ubVariables_y

    # Objective coefficients of variables
    objcoefVariables_gamma = []
    for i in xrange(2, I + 1):
        for j in xrange(1, K + 1):
            objcoefVariables_gamma.append(p[k - 1])
    objcoefVariables_x = [0.0] * numberVariables_x
    objcoefVariables_y = [0.0] * numberVariables_y

    objcoefVariables = objcoefVariables_gamma + objcoefVariables_x + objcoefVariables_y

    # Populate constraints matrix
    constCoefMatrix = []

    def variable_index(variable_name):
        """Return variable index inside the variable name vector nameVariables based on the specified variable name.
        e.g. input: "gamma_2_1",
            output: 0

        Parameter:
        ---------
        :param variable_name: str
        :return: int
        """
        variableindex = -1
        for j in xrange(numberVariables):
            if (variable_name == nameVariables[j]):
                variableindex = j
                break
        if (variableindex == -1):
            raise ValueError("variable name not in the vector nameVariables")
        else:
            return variableindex

    # Add constraints-1:
    for i in xrange(2, I + 1):
        for t in xrange(1, i):
            for k in xrange(1, K + 1):
                valtemp = [0.0] * numberVariables  # Initialize the specified row of linear coefficient matrix

                # append x
                for l in xrange(t, i):
                    for j in xrange(1, J + 1):
                        name_variable_temp = "x_%d_%d" % (l, j)
                        valtemp[
                            variable_index(
                                variable_name=name_variable_temp
                            )
                        ] = s[k - 1][l - 1][j - 1]

                # append yt
                name_variable_temp = "y_%d" % t
                valtemp[
                    variable_index(
                        variable_name=name_variable_temp
                    )
                ] = 1.0

                # append yi
                name_variable_temp = "y_%d" % i
                valtemp[
                    variable_index(
                        variable_name=name_variable_temp
                    )
                ] = -1.0

                # append gamma
                name_variable_temp = "gamma_%d_%d" % (i, k)
                valtemp[
                    variable_index(
                        variable_name=name_variable_temp
                    )
                ] = -1.0

                constCoefMatrix.append(valtemp)

    # Add constraints-2, first part of formula (6):
    for i in xrange(1, I + 1):
        valtemp = [0.0] * numberVariables
        for j in xrange(1, J + 1):
            name_x_temp = "x_%d_%d" % (i, j)
            valtemp[
                variable_index(
                    variable_name=name_x_temp
                )
            ] = 1.0

        constCoefMatrix.append(valtemp)

    # Add constraints-3, second part of formula (6):
    for j in xrange(1, J + 1):
        valtemp = [0.0] * numberVariables
        for i in xrange(1, I + 1):
            name_x_temp = "x_%d_%d" % (i, j)
            valtemp[
                variable_index(
                    variable_name=name_x_temp
                )
            ] = 1.0

        constCoefMatrix.append(valtemp)

    # Add constraints-4, first part of formula (7):
    valtemp = [0.0] * numberVariables
    name_y_temp = "y_1"
    valtemp[
        variable_index(
            variable_name=name_y_temp
        )
    ] = 1.0

    constCoefMatrix.append(valtemp)

    # Add constraints-5, second part of formula (7):
    for i in xrange(2, I + 1):
        valtemp = [0.0] * numberVariables

        name_y_temp = "y_%d" % (i - 1)
        valtemp[
            variable_index(
                variable_name=name_y_temp
            )
        ] = 1.0
        name_y_temp = "y_%d" % i
        valtemp[
            variable_index(
                variable_name=name_y_temp
            )
        ] = -1.0

        constCoefMatrix.append(valtemp)

    # Add constraints-6, third part of formula (7):
    valtemp = [0.0] * numberVariables
    name_y_temp = "y_%d" % I
    valtemp[
        variable_index(
            variable_name=name_y_temp
        )
    ] = 1.0

    constCoefMatrix.append(valtemp)

    # Setting for variables in algorithm
    variablesVec = []
    for j in xrange(numberVariables):
        variablesVec.append(
            model.addVar(
                lb=lbVariables[j],
                ub=ubVariables[j],
                obj=objcoefVariables[j],
                vtype=typeVariables[j],
                name=nameVariables[j]
            )
        )

    for i in xrange(numberConstraints):
        expr = gurobipy.LinExpr()
        for j in xrange(numberVariables):
            if (constCoefMatrix[i][j] != 0):
                expr += constCoefMatrix[i][j] * variablesVec[j]
        model.addConstr(
            lhs=expr,
            sense=bkConstraints[i],
            rhs=bvConstraints[i]
        )

    # Populate objective
    objective = gurobipy.LinExpr()
    for j in xrange(numberVariables):
        if (objcoefVariables[j] != 0.0):
            objective += objcoefVariables[j] * variablesVec[j]

    # parameter sense in function model.setObjective takes values from {gurobipy.GRB.MAXIMIZE, gurobipy.GRB.MINIMIZE}
    model.setObjective(objective, gurobipy.GRB.MINIMIZE)

    # Solve the model
    model.optimize()

    # Print solutions
    if (model.status == gurobipy.GRB.Status.OPTIMAL):
        solutionsDict = {}
        for variable_index in xrange(len(model.getVars())):
            variable = model.getVars()[variable_index]
            solutionsDict[nameVariables[variable_index]] = variable.x
        objValue = model.objVal
        return solutionsDict, objValue
    elif (model.status == gurobipy.GRB.Status.INF_OR_UNBD):
        print("Model is infeasible or unbounded")
        sys.exit(0)
    elif (model.status == gurobipy.GRB.Status.INFEASIBLE):
        print("Model is infeasible")
        sys.exit(0)
    elif (model.status == gurobipy.GRB.Status.UNBOUNDED):
        print("Model is unbounded")
        sys.exit(0)
    else:
        print("Optimization ended with status %d" % model.status)
        sys.exit(0)