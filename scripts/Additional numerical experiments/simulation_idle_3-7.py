# -*- coding: utf-8 -*-

import sys
import gurobipy
import numpy as np
from six.moves import xrange
import xlrd


def read_simulation_samples_from_excel(filename, sheetname="Sheet1", I=10, J=2, numSamples=3000):
    wb = xlrd.open_workbook(filename=filename)
    ws = wb.sheet_by_name(sheet_name=sheetname)

    samples = []
    for sample_index in xrange(numSamples):
        matrix = []
        for position_index in xrange(I):
            vec = []
            for type_index in xrange(J):
                vec.append(
                    ws.cell_value(
                        rowx=11 * sample_index + 1 + position_index,
                        colx=type_index
                    )
                )
            matrix.append(vec)
        samples.append(matrix)
    samples = np.asarray(a=samples, dtype=float)
    return samples



def tad_with_service_time_dist_idle(I, J, K, L, N, p, r, s, tau):
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
    for i in xrange(2, I + 1):       # for i in xrange(a, b + 1): a, a+1, ..., b
        for t in xrange(1, i):
            for k in xrange(1, K + 1):
                numberConstraints3 += 1
    numberConstraints4 = I  # first constraint in feasible set X
    numberConstraints5 = J  # second constraint in feasible set X
    numberConstraints6 = 1  # first constraint in feasible set Y
    numberConstraints7 = I - 1  # second constraint in feasible set Y
    numberConstraints8 = 1  # third constraint in feasible set Y
    numberConstraintsext0 = 1
    numberConstraintsext1 = K
    numberConstraintsext2 = K
    numberConstraintsext3 = I * K
    numberConstraints = numberConstraints1 + numberConstraints2 + numberConstraints3 + \
                        numberConstraints4 + numberConstraints5 + \
                        numberConstraints6 + numberConstraints7 + numberConstraints8 + \
                        numberConstraintsext0 + numberConstraintsext1 + numberConstraintsext2 + numberConstraintsext3

    # Bound keys of constraints
    bkConstraints1 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints1
    bkConstraints2 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints2
    bkConstraints3 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints3
    bkConstraints4 = [gurobipy.GRB.EQUAL] * numberConstraints4
    bkConstraints5 = [gurobipy.GRB.EQUAL] * numberConstraints5
    bkConstraints6 = [gurobipy.GRB.EQUAL] * numberConstraints6
    bkConstraints7 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints7
    bkConstraints8 = [gurobipy.GRB.LESS_EQUAL] * numberConstraints8
    bkConstraintsext0 = [gurobipy.GRB.LESS_EQUAL] * numberConstraintsext0
    bkConstraintsext1 = [gurobipy.GRB.LESS_EQUAL] * numberConstraintsext1
    bkConstraintsext2 = [gurobipy.GRB.LESS_EQUAL] * numberConstraintsext2
    bkConstraintsext3 = [gurobipy.GRB.LESS_EQUAL] * numberConstraintsext3
    bkConstraints = bkConstraints1 + bkConstraints2 + bkConstraints3 + \
                    bkConstraints4 + bkConstraints5 + \
                    bkConstraints6 + bkConstraints7 + bkConstraints8 + \
                    bkConstraintsext0 + bkConstraintsext1 + bkConstraintsext2 + bkConstraintsext3

    # Bound values of constraints/right-hand-side
    bvConstraints1 = [0.0] * numberConstraints1
    bvConstraints2 = [0.0] * numberConstraints2
    bvConstraints3 = [0.0] * numberConstraints3
    bvConstraints4 = [1.0] * numberConstraints4
    bvConstraints5 = N.tolist()
    bvConstraints6 = [0.0] * numberConstraints6
    bvConstraints7 = [0.0] * numberConstraints7
    bvConstraints8 = [L] * numberConstraints8
    bvConstraintsext0 = [tau] * numberConstraintsext0
    bvConstraintsext1 = [-tau] * numberConstraintsext1
    bvConstraintsext2 = [-L] * numberConstraintsext2
    bvConstraintsext3 = [0.0] * numberConstraintsext3
    bvConstraints = bvConstraints1 + bvConstraints2 + bvConstraints3 + \
                    bvConstraints4 + bvConstraints5 + \
                    bvConstraints6 + bvConstraints7 + bvConstraints8 + \
                    bvConstraintsext0 + bvConstraintsext1 + bvConstraintsext2 + bvConstraintsext3


    # Number of variables
    numberVariables_gamma = (I - 1) * K  # i belongs to [2, I]
    numberVariables_alpha = I - 1  # i belongs to [2, I]
    numberVariables_x = I * J
    numberVariables_y = I
    numberVariables_gamma_idle = K
    numberVariables_alpha_idle = 1
    numberVariables = numberVariables_gamma + numberVariables_alpha + numberVariables_x + numberVariables_y + numberVariables_gamma_idle + numberVariables_alpha_idle

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
        
    nameVariables_gamma_idle = []
    nameVariables_alpha_idle = []
    for k in xrange(1, K + 1):
        name_gamma_idle_temp = "gamma_idle_%d" % k
        nameVariables_gamma_idle.append(name_gamma_idle_temp)

    name_alpha_idle_temp = "alpha_idle_%d" % 1
    nameVariables_alpha_idle.append(name_alpha_idle_temp)
    
    nameVariables = nameVariables_gamma + nameVariables_alpha + nameVariables_x + nameVariables_y + nameVariables_gamma_idle + nameVariables_alpha_idle

    # Type of variables
    typeVariables_gamma = [gurobipy.GRB.CONTINUOUS] * numberVariables_gamma
    typeVariables_alpha = [gurobipy.GRB.CONTINUOUS] * numberVariables_alpha
    typeVariables_x = [gurobipy.GRB.BINARY] * numberVariables_x
    typeVariables_y = [gurobipy.GRB.CONTINUOUS] * numberVariables_y
    typeVariables_gamma_idle = [gurobipy.GRB.CONTINUOUS] * numberVariables_gamma_idle
    typeVariables_alpha_idle = [gurobipy.GRB.CONTINUOUS] * numberVariables_alpha_idle
    typeVariables = typeVariables_gamma + typeVariables_alpha + typeVariables_x + typeVariables_y + typeVariables_gamma_idle + typeVariables_alpha_idle

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
    
    # gamma_idle
    lbVariables_gamma_idle = [-gurobipy.GRB.INFINITY] * numberVariables_gamma_idle
    ubVariables_gamma_idle = [+gurobipy.GRB.INFINITY] * numberVariables_gamma_idle

    # alpha_idle
    lbVariables_alpha_idle = [0.0] * numberVariables_alpha_idle
    ubVariables_alpha_idle = [+gurobipy.GRB.INFINITY] * numberVariables_alpha_idle

    lbVariables = lbVariables_gamma + lbVariables_alpha + lbVariables_x + lbVariables_y + lbVariables_gamma_idle + lbVariables_alpha_idle
    ubVariables = ubVariables_gamma + ubVariables_alpha + ubVariables_x + ubVariables_y + ubVariables_gamma_idle + ubVariables_alpha_idle

    # Objective coefficients of variables
    objcoefVariables_gamma = [0.0] * numberVariables_gamma
    objcoefVariables_alpha = [1.0] * numberVariables_alpha
    objcoefVariables_x = [0.0] * numberVariables_x
    objcoefVariables_y = [0.0] * numberVariables_y
    objcoefVariables_gamma_idle = [0.0] * numberVariables_gamma_idle
    objcoefVariables_alpha_idle = [1.0] * numberVariables_alpha_idle

    objcoefVariables = objcoefVariables_gamma + objcoefVariables_alpha + objcoefVariables_x + objcoefVariables_y + objcoefVariables_gamma_idle + objcoefVariables_alpha_idle

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

    # Add constraints-4, first part of feasible set X:
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

    # Add constraints-5, second part of feasible set X:
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

    # Add constraints-6, first part of feasible set Y:
    valtemp = [0.0] * numberVariables
    name_y_temp = "y_1"
    valtemp[
        variable_index(
            variable_name=name_y_temp
        )
    ] = 1.0

    constCoefMatrix.append(valtemp)

    # Add constraints-7, second part of feasible set Y:
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

    # Add constraints-8, third part of feasible set Y:
    valtemp = [0.0] * numberVariables
    name_y_temp = "y_%d" % I
    valtemp[
        variable_index(
            variable_name=name_y_temp
        )
    ] = 1.0

    constCoefMatrix.append(valtemp)
    
    
    # Add constraints-ext0:
    valtemp = [0.0] * numberVariables
    # append gamma
    for k in xrange(1, K + 1):
        name_gamma_idle_temp = "gamma_idle_%d" % k
        valtemp[
            variable_index(
                variable_name=name_gamma_idle_temp
            )
        ] = p[k - 1]
    constCoefMatrix.append(valtemp)
        
    # Add constraints-ext1:
    valtemp = [0.0] * numberVariables
    # append gamma
    for k in xrange(1, K + 1):
        name_gamma_idle_temp = "gamma_idle_%d" % k
        valtemp[
            variable_index(
                variable_name=name_gamma_idle_temp
            )
        ] = -1.0
        name_alpha_idle_temp = "alpha_idle_%d" % 1
        valtemp[
            variable_index(
                variable_name=name_alpha_idle_temp
            )
        ] = -1.0
        constCoefMatrix.append(valtemp)
    
    # Add constraints-ext2:
    for k in xrange(1, K + 1):
        valtemp = [0.0] * numberVariables
        # append gamma
        name_gamma_idle_temp = "gamma_idle_%d" % k
        valtemp[
            variable_index(
                variable_name=name_gamma_idle_temp
            )
        ] = -1.0
        # append x
        for i in xrange(1, I + 1):
            for j in xrange(1, J + 1):
                name_x_temp = "x_%d_%d" % (i, j)
                valtemp[
                    variable_index(
                        variable_name=name_x_temp
                    )
                ] = -s[k - 1][i - 1][j - 1]

        constCoefMatrix.append(valtemp)

    # Add constraints-ext3-p2:
    for t in xrange(2, I + 1):
        for k in xrange(1, K + 1):
            valtemp = [0.0] * numberVariables

            # append x
            for i in xrange(1, t):
                for j in xrange(1, J + 1):
                    name_x_temp = "x_%d_%d" % (i, j)
                    valtemp[
                        variable_index(
                            variable_name=name_x_temp
                        )
                    ] = -s[k - 1][i - 1][j - 1]
            # append y
            name_y_temp = "y_%d" % t
            valtemp[
                variable_index(
                    variable_name=name_y_temp
                )
            ] = 1.0

            # append gamma
            name_gamma_idle_temp = "gamma_idle_%d" % k
            valtemp[
                variable_index(
                    variable_name=name_gamma_idle_temp
                )
            ] = -1.0

            constCoefMatrix.append(valtemp)
            
    # Add constraints-ext3-p1:
    for k in xrange(1, K + 1):
        valtemp = [0.0] * numberVariables

        # append y
        name_y_temp = "y_%d" % 1
        valtemp[
            variable_index(
                variable_name=name_y_temp
            )
        ] = 1.0

        # append gamma
        name_gamma_idle_temp = "gamma_idle_%d" % k
        valtemp[
            variable_index(
                variable_name=name_gamma_idle_temp
            )
        ] = -1.0

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
        model.addLConstr(
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
        
        


if __name__ == '__main__':

    I = 10  # Number of appointment positions
    J = 2  # Number of user types
    K = 1000  # Training sample size
    L = 20  # Total session length of the planning horizon
    N = [3, 7]  # Number of category users
    N = np.asarray(a=N, dtype=int)
    p = [1 / K] * K
    p = np.asarray(a=p, dtype=float)
    N_T = 1000  # Testing sample size
    s_all = read_simulation_samples_from_excel(filename="..//data//Simulation Samples.xlsx")
    s_train = s_all[: K, :, :]
    s_test = s_all[-N_T:, :, :]
    
    #Tolerance of idle time
    tau = {
        "-1.5": 1.5
        # "-2": 2,
        # "-2.5": 2.5,
        # "-3": 3,
        # "-3.5": 3.5,
        # "-4": 4
    }
    tolerance = {
        "(1-1)": [1, 1],
        "(1-1.5)": [1, 1.5],
        "(1.5-1)": [1.5, 1]
    }
    
    for quantile_index in list(tolerance.keys()):
        tolerance[quantile_index] = np.asarray(a=tolerance[quantile_index], dtype=float)
        toleranceStr = quantile_index
        

        for tau_index in list(tau.keys()):
            tau[tau_index] = np.asarray(a=tau[tau_index], dtype=float)
            tauStr = tau_index


            # TAD model
            try:
                solutionsDict_TAD, objValue_TAD = tad_with_service_time_dist_idle(
                    I=I,
                    J=J,
                    K=K,
                    L=L,
                    N=N,
                    p=p,
                    r=tolerance[toleranceStr],
                    s=s_train,
                    tau=tau[tauStr]
                )

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
                
