# -*- coding: utf-8 -*-
"""
Created on 2018/3/29 下午6:34
author: Tong Jia
email: cecilio.jia@gmail.com
software: PyCharm
"""

import math
import sys
import time
import gurobipy
import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment
from six.moves import xrange
import xlrd

def read_simulation_samples_from_excel(filename, sheetname, n_positions, n_types, n_matrixs):
    """
    Parameters:
    ----------
    :param filename: the filename of excel file
    :type filename: str

    :param sheetname: the sheetname of excel file
    :type sheetname: str

    :param n_positions: The number of positions (Rows number in per matrix)
    :type n_positions: int

    :param n_types: The number of patient types (Columns number in per matrix)
    :type n_types: int

    :param n_matrixs: The number of sample matrix (service time matrix)
    :type n_matrix: int

    :return:
    """
    wb = xlrd.open_workbook(filename=filename)
    ws = wb.sheet_by_name(sheet_name=sheetname)

    samples = []
    for sample_index in xrange(n_matrixs):
        matrix = []
        for position_index in xrange(n_positions):
            vec = []
            for type_index in xrange(n_types):
                vec.append(
                    ws.cell_value(
                        rowx=(n_positions + 1) * sample_index + 1 + position_index,
                        colx=type_index
                    )
                )
            matrix.append(vec)
        samples.append(matrix)
    samples = np.asarray(a=samples, dtype=float)
    return samples


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
        runtime = model.runtime
        return solutionsDict, objValue, runtime
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
        
        
        
        
        
        
def tad_with_service_time_dist(I, J, K, L, N, p, r, s):
    """An algorithm of TAD (Tolerance Aware Delay) appointment model with service time distribution.
    It has been formulated into a mixed integer linear programming as follow.

    Parameters:
    ----------
    :param I: int
        The total number of available appointment positions.
    :param J: int
        The total number of the user types or categories.
    :param K: int
        The total number of scenarios of user service time matrixs for training.
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
    model = gurobipy.Model("TAD appointment with service time distribution")

    # Number of constraints
    numberConstraints1 = I - 1  # i belongs to [2, I]
    numberConstraints2 = (I - 1) * K # i belongs to [2, I], k belongs to [1, K]
    numberConstraints3 = 0
    for i in xrange(2, I + 1):
        for t in xrange(1, i):
            for k in xrange(1, K + 1):
                numberConstraints3 += 1
    numberConstraints4 = I  # first constraint in formula (6)
    numberConstraints5 = J  # second constraint in formula (6)
    numberConstraints6 = 1  # first constraint in formula (7)
    numberConstraints7 = I - 1  # second constraint in formula (7)
    numberConstraints8 = 1  # third constraint in formula (7)
    numberConstraints = numberConstraints1 + numberConstraints2 + numberConstraints3 + \
                        numberConstraints4 + numberConstraints5 + \
                        numberConstraints6 + numberConstraints7 + numberConstraints8

    # Bound keys of constraints
    bkConstraints1 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints1
    bkConstraints2 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints2
    bkConstraints3 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints3
    bkConstraints4 = [gurobipy.GRB.EQUAL] * numberConstraints4
    bkConstraints5 = [gurobipy.GRB.EQUAL] * numberConstraints5
    bkConstraints6 = [gurobipy.GRB.EQUAL] * numberConstraints6
    bkConstraints7 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints7
    bkConstraints8 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints8
    bkConstraints = bkConstraints1 + bkConstraints2 + bkConstraints3 + \
                    bkConstraints4 + bkConstraints5 + \
                    bkConstraints6 + bkConstraints7 + bkConstraints8

    # Bound values of constraints
    bvConstraints1 = [0.0] * numberConstraints1
    bvConstraints2 = [0.0] * numberConstraints2
    bvConstraints3 = [0.0] * numberConstraints3
    bvConstraints4 = [1.0] * numberConstraints4
    bvConstraints5 = N.tolist()
    bvConstraints6 = [0.0] * numberConstraints6
    bvConstraints7 = [0.0] * numberConstraints7
    bvConstraints8 = [L] * numberConstraints8
    bvConstraints = bvConstraints1 + bvConstraints2 + bvConstraints3 + \
                    bvConstraints4 + bvConstraints5 + \
                    bvConstraints6 + bvConstraints7 + bvConstraints8


    # Number of variables
    numberVariables_gamma = (I - 1) * K  # i belongs to [2, I]
    numberVariables_alpha = I - 1  # i belongs to [2, I]
    numberVariables_x = I * J
    numberVariables_y = I
    numberVariables = numberVariables_gamma + numberVariables_alpha + numberVariables_x + numberVariables_y

    # Name of variables
    nameVariables_gamma = []
    nameVariables_alpha = []
    for i in xrange(2, I + 1):
        for k in xrange(1, K + 1):
            name_gamma_temp = "gamma_%d_%d" % (i, k)
            nameVariables_gamma.append(name_gamma_temp)

        name_alpha_temp = "alpha_%d" % i
        nameVariables_alpha.append(name_alpha_temp)
    nameVariables_x = []
    nameVariables_y = []
    for i in xrange(1, I + 1):
        for j in xrange(1, J + 1):
            name_x_temp = "x_%d_%d" % (i, j)
            nameVariables_x.append(name_x_temp)
        name_y_temp = "y_%d" % i
        nameVariables_y.append(name_y_temp)
    nameVariables = nameVariables_gamma + nameVariables_alpha + nameVariables_x + nameVariables_y

    # Type of variables
    typeVariables_gamma = [gurobipy.GRB.CONTINUOUS] * numberVariables_gamma
    typeVariables_alpha = [gurobipy.GRB.CONTINUOUS] * numberVariables_alpha
    typeVariables_x = [gurobipy.GRB.BINARY] * numberVariables_x
    typeVariables_y = [gurobipy.GRB.CONTINUOUS] * numberVariables_y
    typeVariables = typeVariables_gamma + typeVariables_alpha + typeVariables_x + typeVariables_y

    # Bound values of variables
    # gamma
    lbVariables_gamma = [0.0] * numberVariables_gamma
    ubVariables_gamma = [+gurobipy.GRB.INFINITY] * numberVariables_gamma

    # alpha
    lbVariables_alpha = [0.0] * numberVariables_alpha
    ubVariables_alpha = [+gurobipy.GRB.INFINITY] * numberVariables_alpha

    # x
    lbVariables_x = [0.0] * numberVariables_x
    ubVariables_x = [1.0] * numberVariables_x

    # y
    lbVariables_y = [0.0] * numberVariables_y
    ubVariables_y = [+gurobipy.GRB.INFINITY] * numberVariables_y

    lbVariables = lbVariables_gamma + lbVariables_alpha + lbVariables_x + lbVariables_y
    ubVariables = ubVariables_gamma + ubVariables_alpha + ubVariables_x + ubVariables_y

    # Objective coefficients of variables
    objcoefVariables_gamma = [0.0] * numberVariables_gamma
    objcoefVariables_alpha = [1.0] * numberVariables_alpha
    objcoefVariables_x = [0.0] * numberVariables_x
    objcoefVariables_y = [0.0] * numberVariables_y

    objcoefVariables = objcoefVariables_gamma + objcoefVariables_alpha + objcoefVariables_x + objcoefVariables_y

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
        valtemp = [0.0] * numberVariables  # Initialize the specified row of linear coefficient matrix
        # append gamma
        for k in xrange(1, K + 1):
            name_gamma_temp = "gamma_%d_%d" % (i, k)
            valtemp[
                variable_index(
                    variable_name=name_gamma_temp
                )
            ] = p[k - 1]
        # append x
        for j in xrange(1, J + 1):
            name_x_temp = "x_%d_%d" % (i, j)
            valtemp[
                variable_index(
                    variable_name=name_x_temp
                )
            ] = -r[j - 1]
        constCoefMatrix.append(valtemp)

    # Add constraints-2:
    for i in xrange(2, I + 1):
        for k in xrange(1, K + 1):
            valtemp = [0.0] * numberVariables
            # append x
            for j in xrange(1, J + 1):
                name_x_temp = "x_%d_%d" % (i, j)
                valtemp[
                    variable_index(
                        variable_name=name_x_temp
                    )
                ] = r[j - 1]
            # append gamma
            name_gamma_temp = "gamma_%d_%d" % (i, k)
            valtemp[
                variable_index(
                    variable_name=name_gamma_temp
                )
            ] = -1.0
            # append alpha
            name_alpha_temp = "alpha_%d" % i
            valtemp[
                variable_index(
                    variable_name=name_alpha_temp
                )
            ] = -1.0

            constCoefMatrix.append(valtemp)

    # Add constraints-3:
    for i in xrange(2, I + 1):
        for t in xrange(1, i):
            for k in xrange(1, K + 1):
                valtemp = [0.0] * numberVariables

                # append x
                for l in xrange(t, i):
                    for j in xrange(1, J + 1):
                        name_x_temp = "x_%d_%d" % (l, j)
                        valtemp[
                            variable_index(
                                variable_name=name_x_temp
                            )
                        ] = s[k - 1][l - 1][j - 1]
                # append y
                name_y_temp = "y_%d" % t
                valtemp[
                    variable_index(
                        variable_name=name_y_temp
                    )
                ] = 1.0
                valtemp[
                    variable_index(
                        variable_name="y_%d" % i
                    )
                ] = -1.0

                # append gamma
                name_gamma_temp = "gamma_%d_%d" % (i, k)
                valtemp[
                    variable_index(
                        variable_name=name_gamma_temp
                    )
                ] = -1.0

                constCoefMatrix.append(valtemp)

    # Add constraints-4, first part of formula (6):
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

    # Add constraints-5, second part of formula (6):
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

    # Add constraints-6, first part of formula (7):
    valtemp = [0.0] * numberVariables
    name_y_temp = "y_1"
    valtemp[
        variable_index(
            variable_name=name_y_temp
        )
    ] = 1.0

    constCoefMatrix.append(valtemp)

    # Add constraints-7, second part of formula (7):
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

    # Add constraints-8, third part of formula (7):
    valtemp = [0.0] * numberVariables
    name_y_temp = "y_%d" % I
    valtemp[
        variable_index(
            variable_name=name_y_temp
        )
    ] = 1.0

    constCoefMatrix.append(valtemp)

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
        runtime = model.runtime
        return solutionsDict, objValue, runtime
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
        
        
        
        
        
        
        

def model_DUM_revised(I, J, K, L, Lambda, N, p, r, s, index_set):
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
    # if (Lambda <= 0 or Lambda > 1):
    #     raise ValueError("the parameter Lambda must be in range of (0, 1]")
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
    model = gurobipy.Model("DUM model revised")

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
            ] = (1 / Lambda[i - 2]) * p[k - 1]

        # append x
        for j in xrange(1, J + 1):
            name_variable_temp = "x_%d_%d" % (i, j)
            valtemp[
                variable_index(
                    variable_name=name_variable_temp
                )
            ] = -r[j - 1]
        
        # append h
        if i not in index_set:
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
        runtime = model.runtime
        return solutionsDict, objValue, runtime
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
        
        
        


def model_DUM_single(I_temp, J, K, Lambda, p, r, w):
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

    # Create a model
    model = gurobipy.Model("DUM model single")

    # Number of constraints
    numberConstraints1 = 0
    for t in xrange(1, I_temp):
        for k in xrange(1, K + 1):
            numberConstraints1 += 1
    numberConstraints2 = K
    numberConstraints8 = 1

    numberConstraints = numberConstraints1 + numberConstraints2 + numberConstraints8

    # Bound keys of constraints
    bkConstraints1 = [gurobipy.GRB.GREATER_EQUAL] * numberConstraints1
    bkConstraints2 = [gurobipy.GRB.GREATER_EQUAL] * numberConstraints2
    bkConstraints8 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints8

    bkConstraints = bkConstraints1 + bkConstraints2 + bkConstraints8

    # Bound values of constraints
    bvConstraints1 = []
    for k in xrange(1, K + 1):
        for t in xrange(1, I_temp):
            bvConstraints1.append(w[k][t-1])
    bvConstraints2 = [0.0] * numberConstraints2
    bvConstraints8 = [r] * numberConstraints8

    bvConstraints = bvConstraints1 + bvConstraints2 + bvConstraints8


    # Number of variables
    numberVariables_gamma = K  # i belongs to [2, I]
    numberVariables_v = 1
    numberVariables_h = 1
    numberVariables = numberVariables_gamma + \
                      numberVariables_v + \
                      numberVariables_h

    # Name of variables
    nameVariables_gamma = []
    for k in xrange(1, K + 1):
        name_gamma_temp = "gamma_%d" % k
        nameVariables_gamma.append(name_gamma_temp)

    nameVariables_v = ["v"] * numberVariables_v
    nameVariables_h = ["h"] * numberVariables_h
    nameVariables = nameVariables_gamma + \
                    nameVariables_v + \
                    nameVariables_h

    # Type of variables
    typeVariables_gamma = [gurobipy.GRB.CONTINUOUS] * numberVariables_gamma
    typeVariables_v = [gurobipy.GRB.CONTINUOUS] * numberVariables_v
    typeVariables_h = [gurobipy.GRB.CONTINUOUS] * numberVariables_h
    typeVariables = typeVariables_gamma + \
                    typeVariables_v + \
                    typeVariables_h

    # Bound values of variables
    lbVariables_gamma = [0.0] * numberVariables_gamma
    ubVariables_gamma = [+gurobipy.GRB.INFINITY] * numberVariables_gamma

    lbVariables_v = [-gurobipy.GRB.INFINITY] * numberVariables_v
    ubVariables_v = [+gurobipy.GRB.INFINITY] * numberVariables_v

    lbVariables_h = [-gurobipy.GRB.INFINITY] * numberVariables_h
    ubVariables_h = [+gurobipy.GRB.INFINITY] * numberVariables_h

    lbVariables = lbVariables_gamma + lbVariables_v + lbVariables_h
    ubVariables = ubVariables_gamma + ubVariables_v + ubVariables_h

    # Objective coefficients of variables
    objcoefVariables_gamma = [0.0] * numberVariables_gamma
    objcoefVariables_v = [0.0] * numberVariables_v
    objcoefVariables_h = [1.0] * numberVariables_h

    objcoefVariables = objcoefVariables_gamma + objcoefVariables_v + objcoefVariables_h


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
    for k in xrange(1, K + 1):
        for t in xrange(1, I_temp):
            valtemp = [0.0] * numberVariables  # Initialize the specified row of linear coefficient matrix

            # append v
            name_variable_temp = "v"
            valtemp[
                variable_index(
                    variable_name=name_variable_temp
                )
            ] = 1.0

            # append gamma
            name_variable_temp = "gamma_%d" % k
            valtemp[
                variable_index(
                    variable_name=name_variable_temp
                )
            ] = 1.0

            constCoefMatrix.append(valtemp)

    # Add constraints-2:
    for k in xrange(1, K + 1):
        valtemp = [0.0] * numberVariables

        # append gamma
        name_variable_temp = "gamma_%d" % k
        valtemp[
            variable_index(
                variable_name=name_variable_temp
            )
        ] = 1.0

        # append v
        name_variable_temp = "v"
        valtemp[
            variable_index(
                variable_name=name_variable_temp
            )
        ] = 1.0

        constCoefMatrix.append(valtemp)


    # Add constraints-8:
    valtemp = [0.0] * numberVariables

    # append v
    name_variable_temp = "v"
    valtemp[
        variable_index(
            variable_name=name_variable_temp
        )
    ] = 1.0

    # append gamma
    for k in xrange(1, K + 1):
        name_variable_temp = "gamma_%d" % k
        valtemp[
            variable_index(
                variable_name=name_variable_temp
            )
        ] = (1 / Lambda) * p[k - 1]

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
        runtime = model.runtime
        return solutionsDict, objValue, runtime
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





def evaluate_waiting_time(solutionsDict, s, i, J, K):
    w_final = {}
    for k in xrange(1, K + 1):
        w_final[k] = []
        for t in xrange(1, i):
            w_temp = 0
            for l in xrange(t, i):
                for j in xrange(1, J + 1):
                    name_variable_temp = "x_%d_%d" % (l, j)
                    w_temp += s[k - 1][l - 1][j - 1]*solutionsDict[name_variable_temp]
            y_temp_1 = "y_%d" % i
            y_temp_2 = "y_%d" % t
            w_final[k].append(w_temp - solutionsDict[y_temp_1] + solutionsDict[y_temp_2])
    
    return w_final
            


def evaluate_tolerance(solutionsDict, i, J, r):
    tau_temp = 0
    for j in xrange(1, J + 1):
        name_variable_temp = "x_%d_%d" % (i, j)
        tau_temp += r[j - 1]*solutionsDict[name_variable_temp]
    return tau_temp





if __name__ == '__main__':
    I = 10      # Number of appointment positions
    K = 200     # Training sample size
    L = 20      # Total session length of the planning horizon
    p = [1 / K] * K
    p = np.asarray(a=p, dtype=float)
    n_matrixs = 1000
    J = 2       # Number of user types
    
    num_types = {
        "2": [5, 5],
        "3": [4, 3, 3],
        "4": [3, 2, 3, 2],
        "5": [2, 2, 2, 2, 2]
    }           # Number of category users
    tolerance = {
        "2": [0.5, 1.5],
        "3": [0.5, 1, 1.5],
        "4": [0.5, 0.8, 1.2, 1.5],
        "5": [0.5, 0.8, 1.0, 1.2, 1.5]
    }
    
    
    for type_index in list(num_types.keys()):
        N = num_types[type_index]  # Number of category users
        N = np.asarray(a=N, dtype=int)
        
        tolerance[type_index] = np.asarray(a=tolerance[type_index], dtype=float)
        toleranceStr = type_index
        
    
        s_all = read_simulation_samples_from_excel(
            filename="..//data//Simulation samples-1000-10-" + type_index + ".xlsx",
            sheetname="Sheet1",
            n_matrixs=n_matrixs,
            n_positions=I,
            n_types=J
        )
        s_train = s_all[: K, :, :]
        s_test = s_all

    

        # ED model
        try:
            start_time_ED = time.time()
            solutionsDict_ED, objValue_ED, runtime_ED = model_ED(
                I=I,
                J=J,
                K=K,
                L=L,
                N=N,
                p=p,
                s=s_train
            )
            end_time_ED = time.time()
            time_cost_ED = end_time_ED - start_time_ED

        except gurobipy.GurobiError as e:
            print("ERROR: %s" % str(e.errno))
            if e.msg is not None:
                print("\t%s" % e.msg)
                sys.exit(1)
        except AttributeError:
            print("Encountered an attribute error")
        except:
            import traceback
    
            traceback.print_exc()
            sys.exit(1)



        # TAD model
        try:
            start_time_TAD = time.time()
            solutionsDict_TAD, objValue_TAD, runtime_TAD = tad_with_service_time_dist(
                I=I,
                J=J,
                K=K,
                L=L,
                N=N,
                p=p,
                r=tolerance[toleranceStr],
                s=s_train
            )
            end_time_TAD = time.time()
            time_cost_TAD = end_time_TAD - start_time_TAD

        except gurobipy.GurobiError as e:
            print("ERROR: %s" % str(e.errno))
            if e.msg is not None:
                print("\t%s" % e.msg)
                sys.exit(1)
        except AttributeError:
            print("Encountered an attribute error")
        except:
            import traceback
    
            traceback.print_exc()
            sys.exit(1)



        # DUM model
        # Search Lambda
        error_bound = 0.1
        
        index_updated = []
        alpha_updated = []
        index_count = 0
        runtime_DUM = []
        runtime_DUM_single = []
    
        try:
            start_time_DUM = time.time()
            while(index_count < I-1):
                Lambda_L = 0.005
                Lambda_U = 1.0
                Lambda_L_revised = []
                Lambda_U_revised = []
                for i in range(I - 1):
                    if i+2 in index_updated:
                        Lambda_L_revised.append(alpha_updated[index_updated.index(i+2)])
                        Lambda_U_revised.append(alpha_updated[index_updated.index(i+2)])
                    else:
                        Lambda_L_revised.append(Lambda_L)
                        Lambda_U_revised.append(Lambda_U)
                    
                solutionsDict_DUM_L, objValue_DUM_L, runtime_DUM_temp = model_DUM_revised(
                    I=I,
                    J=J,
                    K=K,
                    L=L,
                    Lambda=Lambda_L_revised,
                    N=N,
                    p=p,
                    r=tolerance[toleranceStr],
                    s=s_train, 
                    index_set=index_updated
                )
                runtime_DUM.append(runtime_DUM_temp)
                solutionsDict_DUM_U, objValue_DUM_U, runtime_DUM_temp = model_DUM_revised(
                    I=I,
                    J=J,
                    K=K,
                    L=L,
                    Lambda=Lambda_U_revised,
                    N=N,
                    p=p,
                    r=tolerance[toleranceStr],
                    s=s_train, 
                    index_set=index_updated
                )
                runtime_DUM.append(runtime_DUM_temp)
                # while ((objValue_DUM_L - objValue_DUM_U) > error_bound or (Lambda_U - Lambda_L) > error_bound+0.1):
                while ((objValue_DUM_L - objValue_DUM_U) > error_bound):
                # while ((Lambda_U - Lambda_L) > error_bound):
                    Lambda_C = (Lambda_L + Lambda_U) / 2
                    Lambda_C_revised = []
                    for i in range(I - 1):
                        if i+2 in index_updated:
                            Lambda_C_revised.append(alpha_updated[index_updated.index(i+2)])
                        else:
                            Lambda_C_revised.append(Lambda_C)
                            
                    solutionsDict_DUM_C, objValue_DUM_C, runtime_DUM_temp = model_DUM_revised(
                        I=I,
                        J=J,
                        K=K,
                        L=L,
                        Lambda=Lambda_C_revised,
                        N=N,
                        p=p,
                        r=tolerance[toleranceStr],
                        s=s_train, 
                        index_set=index_updated
                    )
                    runtime_DUM.append(runtime_DUM_temp)
                    if (objValue_DUM_C <= 0):
                        Lambda_U = Lambda_C
                        Lambda_U_revised = Lambda_C_revised
                    else:
                        Lambda_L = Lambda_C
                        Lambda_L_revised = Lambda_C_revised
                    solutionsDict_DUM_L, objValue_DUM_L, runtime_DUM_temp = model_DUM_revised(
                        I=I,
                        J=J,
                        K=K,
                        L=L,
                        Lambda=Lambda_L_revised,
                        N=N,
                        p=p,
                        r=tolerance[toleranceStr],
                        s=s_train, 
                        index_set=index_updated
                    )
                    runtime_DUM.append(runtime_DUM_temp)
                    solutionsDict_DUM_U, objValue_DUM_U, runtime_DUM_temp = model_DUM_revised(
                        I=I,
                        J=J,
                        K=K,
                        L=L,
                        Lambda=Lambda_U_revised,
                        N=N,
                        p=p,
                        r=tolerance[toleranceStr],
                        s=s_train, 
                        index_set=index_updated
                    )
                    runtime_DUM.append(runtime_DUM_temp)
                
                
                if (index_count < I-2):
                    rho_temp = []
                    for i in xrange(2, I + 1):
                        
                        if i in index_updated:
                            rho_temp.append(0)
                        else:
                            waiting_time_temp = evaluate_waiting_time(solutionsDict_DUM_U, s_train, i, J, K)
                            tolerance_temp = evaluate_tolerance(solutionsDict_DUM_U, i, J, tolerance[toleranceStr])
                            
                            Lambda_L_single = 0.005
                            Lambda_U_single = 1.0
                                
                            solutionsDict_DUM_L_single, objValue_DUM_L_single, runtime_DUM_single_temp = model_DUM_single(
                                I_temp=i,
                                J=J,
                                K=K,
                                Lambda=Lambda_L_single,
                                p=p,
                                r=tolerance_temp,
                                w=waiting_time_temp
                            )
                            runtime_DUM_single.append(runtime_DUM_single_temp)
                            solutionsDict_DUM_U_single, objValue_DUM_U_single, runtime_DUM_single_temp = model_DUM_single(
                                I_temp=i,
                                J=J,
                                K=K,
                                Lambda=Lambda_U_single,
                                p=p,
                                r=tolerance_temp,
                                w=waiting_time_temp
                            )
                            runtime_DUM_single.append(runtime_DUM_single_temp)
                            # while ((objValue_DUM_L_single - objValue_DUM_U_single) > error_bound or (Lambda_U_single - Lambda_L_single) > error_bound):
                            while ((objValue_DUM_L_single - objValue_DUM_U_single) > error_bound):
                            # while ((Lambda_U_single - Lambda_L_single) > error_bound):
                                Lambda_C_single = (Lambda_L_single + Lambda_U_single) / 2
                                solutionsDict_DUM_C_single, objValue_DUM_C_single, runtime_DUM_single_temp = model_DUM_single(
                                    I_temp=i,
                                    J=J,
                                    K=K,
                                    Lambda=Lambda_C_single,
                                    p=p,
                                    r=tolerance_temp,
                                    w=waiting_time_temp
                                )
                                runtime_DUM_single.append(runtime_DUM_single_temp)
                                if (objValue_DUM_C_single <= 0):
                                    Lambda_U_single = Lambda_C_single
                                else:
                                    Lambda_L_single = Lambda_C_single
                                solutionsDict_DUM_L_single, objValue_DUM_L_single, runtime_DUM_single_temp = model_DUM_single(
                                    I_temp=i,
                                    J=J,
                                    K=K,
                                    Lambda=Lambda_L_single,
                                    p=p,
                                    r=tolerance_temp,
                                    w=waiting_time_temp
                                )
                                runtime_DUM_single.append(runtime_DUM_single_temp)
                                solutionsDict_DUM_U_single, objValue_DUM_U_single, runtime_DUM_single_temp = model_DUM_single(
                                    I_temp=i,
                                    J=J,
                                    K=K,
                                    Lambda=Lambda_U_single,
                                    p=p,
                                    r=tolerance_temp,
                                    w=waiting_time_temp
                                )
                                runtime_DUM_single.append(runtime_DUM_single_temp)
                                
                            rho_temp.append(Lambda_U_single)
                    
                    index_updated.append(rho_temp.index(max(rho_temp)) + 2)
                    alpha_updated.append(Lambda_U_single)
                
                index_count += 1 
    
            
            end_time_DUM = time.time()
            time_cost_DUM = end_time_DUM - start_time_DUM
            

        except gurobipy.GurobiError as e:
            print("ERROR: %s" % str(e.errno))
            if e.msg is not None:
                print("\t%s" % e.msg)
                sys.exit(1)
        except AttributeError:
            print("Encountered an attribute error")
        except:
            import traceback
    
            traceback.print_exc()
            sys.exit(1)
    

        J += 1
        