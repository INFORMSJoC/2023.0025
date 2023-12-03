# -*- coding: utf-8 -*-
"""
Created on 2018/3/18 上午11:09
author: Tong Jia
email: cecilio.jia@gmail.com
software: PyCharm
Description:
    An implement of DUM model using gurobi callback solver. The model is based on formula (63) in paper "APS-18-3-18.pdf".
"""
import sys
import gurobipy
import numpy as np
from six.moves import xrange

def model_DUM(I, J, K, L, Lambda, N, p, r, s):
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
    :param Lambda: float
        A float in range of (0, 1].
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
    :param r: ndarray
        Tolerance of delay vector;
        the length of r is J;
        r[j] is the delay tolerance threshold of users in category j.
    :param s: ndarray
        Service time tensor;
        s[k][i][j] refers to the service time in position i for a user in category j inside the k-th scenario.

    Returns:
    -------
    :return: solutionsDict: dict
    :return: objValue: float
    """
    # Ckeck point for parameter Lambda
    if (Lambda <= 0 or Lambda > 1):
        raise ValueError("the parameter Lambda must be in range of (0, 1]")
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
    # Ckeck point for parameter r
    if not isinstance(r, np.ndarray):
        raise ValueError("the parameter r is not saved as ndarray")
    if (r.ndim != 1):
        raise ValueError("the parameter r is not a vector (1 dim)")
    if (r.shape[0] != J):
        raise ValueError("the length of parameter r (%d) is not equal to J (%d)" % (r.shape[0], J))
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
    model = gurobipy.Model("DUM model")

    # Number of constraints
    numberConstraints1 = 0
    for i in xrange(2, I + 1):
        for t in xrange(1, i):
            for k in xrange(1, K + 1):
                numberConstraints1 += 1
    numberConstraints2 = 0
    for i in xrange(2, I + 1):
        for k in xrange(1, K + 1):
            numberConstraints2 += 1
    numberConstraints3 = I  # first constraint in formula (6)
    numberConstraints4 = J  # second constraint in formula (6)
    numberConstraints5 = 1  # first constraint in formula (7)
    numberConstraints6 = I - 1  # second constraint in formula (7)
    numberConstraints7 = 1  # third constraint in formula (7)
    numberConstraints8 = I - 1 # the first costraint (obj in APS-2018-3-17)

    numberConstraints = numberConstraints1 + numberConstraints2 + numberConstraints3 + \
                        numberConstraints4 + numberConstraints5 + \
                        numberConstraints6 + numberConstraints7 + \
                        numberConstraints8

    # Bound keys of constraints
    bkConstraints1 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints1
    bkConstraints2 = [gurobipy.GRB.GREATER_EQUAL] * numberConstraints2
    bkConstraints3 = [gurobipy.GRB.EQUAL] * numberConstraints3
    bkConstraints4 = [gurobipy.GRB.EQUAL] * numberConstraints4
    bkConstraints5 = [gurobipy.GRB.EQUAL] * numberConstraints5
    bkConstraints6 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints6
    bkConstraints7 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints7
    bkConstraints8 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints8

    bkConstraints = bkConstraints1 + bkConstraints2 + bkConstraints3 + \
                    bkConstraints4 + bkConstraints5 + \
                    bkConstraints6 + bkConstraints7 + \
                    bkConstraints8

    # Bound values of constraints
    bvConstraints1 = [0.0] * numberConstraints1
    bvConstraints2 = [0.0] * numberConstraints2
    bvConstraints3 = [1.0] * numberConstraints3
    bvConstraints4 = N.tolist()
    bvConstraints5 = [0.0] * numberConstraints5
    bvConstraints6 = [0.0] * numberConstraints6
    bvConstraints7 = [L] * numberConstraints7
    bvConstraints8 = [0.0] * numberConstraints8

    bvConstraints = bvConstraints1 + bvConstraints2 + bvConstraints3 + \
                    bvConstraints4 + bvConstraints5 + \
                    bvConstraints6 + bvConstraints7 + \
                    bvConstraints8


    # Number of variables
    numberVariables_gamma = (I - 1) * K  # i belongs to [2, I]
    numberVariables_v = I - 1
    numberVariables_x = I * J
    numberVariables_y = I
    numberVariables_h = 1
    numberVariables = numberVariables_gamma + \
                      numberVariables_v + \
                      numberVariables_x + numberVariables_y + \
                      numberVariables_h

    # Name of variables
    nameVariables_gamma = []
    nameVariables_v = []
    for i in xrange(2, I + 1):
        for k in xrange(1, K + 1):
            name_gamma_temp = "gamma_%d_%d" % (i, k)
            nameVariables_gamma.append(name_gamma_temp)

        name_v_temp = "v_%d" % i
        nameVariables_v.append(name_v_temp)
    nameVariables_x = []
    nameVariables_y = []
    for i in xrange(1, I + 1):
        for j in xrange(1, J + 1):
            name_x_temp = "x_%d_%d" % (i, j)
            nameVariables_x.append(name_x_temp)
        name_y_temp = "y_%d" % i
        nameVariables_y.append(name_y_temp)
    nameVariables_h = ["h"] * numberVariables_h
    nameVariables = nameVariables_gamma + \
                    nameVariables_v + \
                    nameVariables_x + nameVariables_y + \
                    nameVariables_h

    # Type of variables
    typeVariables_gamma = [gurobipy.GRB.CONTINUOUS] * numberVariables_gamma
    typeVariables_v = [gurobipy.GRB.CONTINUOUS] * numberVariables_v
    typeVariables_x = [gurobipy.GRB.BINARY] * numberVariables_x
    typeVariables_y = [gurobipy.GRB.CONTINUOUS] * numberVariables_y
    typeVariables_h = [gurobipy.GRB.CONTINUOUS] * numberVariables_h
    typeVariables = typeVariables_gamma + \
                    typeVariables_v + \
                    typeVariables_x + typeVariables_y + \
                    typeVariables_h

    # Bound values of variables
    lbVariables_gamma = [0.0] * numberVariables_gamma
    ubVariables_gamma = [+gurobipy.GRB.INFINITY] * numberVariables_gamma

    lbVariables_v = [-gurobipy.GRB.INFINITY] * numberVariables_v
    ubVariables_v = [+gurobipy.GRB.INFINITY] * numberVariables_v

    lbVariables_x = [0.0] * numberVariables_x
    ubVariables_x = [1.0] * numberVariables_x

    lbVariables_y = [0.0] * numberVariables_y
    ubVariables_y = [+gurobipy.GRB.INFINITY] * numberVariables_y

    lbVariables_h = [-gurobipy.GRB.INFINITY] * numberVariables_h
    ubVariables_h = [+gurobipy.GRB.INFINITY] * numberVariables_h

    lbVariables = lbVariables_gamma + lbVariables_v + lbVariables_x + lbVariables_y + lbVariables_h
    ubVariables = ubVariables_gamma + ubVariables_v + ubVariables_x + ubVariables_y + ubVariables_h

    # Objective coefficients of variables
    objcoefVariables_gamma = [0.0] * numberVariables_gamma
    objcoefVariables_v = [0.0] * numberVariables_v
    objcoefVariables_x = [0.0] * numberVariables_x
    objcoefVariables_y = [0.0] * numberVariables_y
    objcoefVariables_h = [1.0] * numberVariables_h

    objcoefVariables = objcoefVariables_gamma + objcoefVariables_v + objcoefVariables_x + objcoefVariables_y + objcoefVariables_h


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

                # append v
                name_variable_temp = "v_%d" % i
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

    # Add constraints-2:
    for i in xrange(2, I + 1):
        for k in xrange(1, K + 1):
            valtemp = [0.0] * numberVariables

            # append gamma
            name_variable_temp = "gamma_%d_%d" % (i, k)
            valtemp[
                variable_index(
                    variable_name=name_variable_temp
                )
            ] = 1.0

            # append v
            name_variable_temp = "v_%d" % i
            valtemp[
                variable_index(
                    variable_name=name_variable_temp
                )
            ] = 1.0

            constCoefMatrix.append(valtemp)

    # Add constraints-3, first part of formula (6):
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

    # Add constraints-4, second part of formula (6):
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

    # Add constraints-5, first part of formula (7):
    valtemp = [0.0] * numberVariables
    name_y_temp = "y_1"
    valtemp[
        variable_index(
            variable_name=name_y_temp
        )
    ] = 1.0

    constCoefMatrix.append(valtemp)

    # Add constraints-6, second part of formula (7):
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

    # Add constraints-7, third part of formula (7):
    valtemp = [0.0] * numberVariables
    name_y_temp = "y_%d" % I
    valtemp[
        variable_index(
            variable_name=name_y_temp
        )
    ] = 1.0

    constCoefMatrix.append(valtemp)

    # Add constraints-8:
    for i in xrange(2, I + 1):
        valtemp = [0.0] * numberVariables

        # append v
        name_variable_temp = "v_%d" % i
        valtemp[
            variable_index(
                variable_name=name_variable_temp
            )
        ] = 1.0

        # append gamma
        for k in xrange(1, K + 1):
            name_variable_temp = "gamma_%d_%d" % (i, k)
            valtemp[
                variable_index(
                    variable_name=name_variable_temp
                )
            ] = (1 / Lambda) * p[k - 1]

        # append x
        for j in xrange(1, J + 1):
            name_variable_temp = "x_%d_%d" % (i, j)
            valtemp[
                variable_index(
                    variable_name=name_variable_temp
                )
            ] = -r[j - 1]

        # append h
        name_variable_temp = "h"
        valtemp[
            variable_index(
                variable_name=name_variable_temp
            )
        ] = -1.0

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