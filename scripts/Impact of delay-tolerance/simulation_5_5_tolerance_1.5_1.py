#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:21:51 2023

@author: lijun
"""

import math
import sys
import gurobipy
import time
import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment
from six.moves import xrange
from generateSample import read_simulation_samples_from_excel
from model_ED import model_ED
from model_DUM import model_DUM
from model_TAD import tad_with_service_time_dist



if __name__ == '__main__':

    I = 10  # Number of appointment positions
    J = 2  # Number of user types
    K = 1000  # Training sample size
    L = 20  # Total session length of the planning horizon
    N = [5, 5]  # Number of category users
    N = np.asarray(a=N, dtype=int)
    p = [1 / K] * K
    p = np.asarray(a=p, dtype=float)

    tolerance = {
        "(1-0.5)": [1, 0.5],
        "(1-1)": [1, 1],
        "(1-1.5)": [1, 1.5],
        "(1-2)": [1, 2],
        "(1.5-1)": [1.5, 1]
    }
    
    for quantile_index in list(tolerance.keys()):
        tolerance[quantile_index] = np.asarray(a=tolerance[quantile_index], dtype=float)

    s_all = read_simulation_samples_from_excel(filename="..//data//Simulation Samples.xlsx")
    s_train = s_all[: K, :, :]
    s_test = s_all[-1000:, :, :]

    # toleranceStr = "(1-0.5)" # *
    # toleranceStr = "(1-1)" # *
    # toleranceStr = "(1-1.5)" # *
    # toleranceStr = "(1-2)" # *
    toleranceStr = "(1.5-1)" # *


    # ED model
    try:
        start_time_ED = time.time()
        solutionsDict_ED, objValue_ED = model_ED(
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
        solutionsDict_TAD, objValue_TAD = tad_with_service_time_dist(
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

    Lambda_L = 0.005
    Lambda_U = 1.0

    try:
        start_time_DUM = time.time()
        solutionsDict_DUM_L, objValue_DUM_L = model_DUM(
            I=I,
            J=J,
            K=K,
            L=L,
            Lambda=Lambda_L,
            N=N,
            p=p,
            r=tolerance[toleranceStr],
            s=s_train
        )
        solutionsDict_DUM_U, objValue_DUM_U = model_DUM(
            I=I,
            J=J,
            K=K,
            L=L,
            Lambda=Lambda_U,
            N=N,
            p=p,
            r=tolerance[toleranceStr],
            s=s_train
        )
        while ((objValue_DUM_L - objValue_DUM_U) > error_bound):
            Lambda_C = (Lambda_L + Lambda_U) / 2

            solutionsDict_DUM_C, objValue_DUM_C = model_DUM(
                I=I,
                J=J,
                K=K,
                L=L,
                Lambda=Lambda_C,
                N=N,
                p=p,
                r=tolerance[toleranceStr],
                s=s_train
            )
            if (objValue_DUM_C <= 0):
                Lambda_U = Lambda_C
            else:
                Lambda_L = Lambda_C
            solutionsDict_DUM_L, objValue_DUM_L = model_DUM(
                I=I,
                J=J,
                K=K,
                L=L,
                Lambda=Lambda_L,
                N=N,
                p=p,
                r=tolerance[toleranceStr],
                s=s_train
            )
            solutionsDict_DUM_U, objValue_DUM_U = model_DUM(
                I=I,
                J=J,
                K=K,
                L=L,
                Lambda=Lambda_U,
                N=N,
                p=p,
                r=tolerance[toleranceStr],
                s=s_train
            )
            # print("Current objValue_DUM_L : %f" % objValue_DUM_L)
            # print("Current objValue_DUM_U : %f" % objValue_DUM_U)
            # print("Current Lambda_L : %f" % Lambda_L)
            # print("Current Lambda_U : %f" % Lambda_U)
            # print("Current objValue_DUM_L - objValue_DUM_U : %f" % (objValue_DUM_L - objValue_DUM_U))
        end_time_DUM = time.time()
        time_cost_DUM = end_time_DUM - start_time_DUM
        print("Final results:")
        print("Final objValue_DUM_L : %f" % objValue_DUM_L)
        print("Final objValue_DUM_U : %f" % objValue_DUM_U)
        print("Final Lambda_L : %f" % Lambda_L)
        print("Final Lambda_U : %f" % Lambda_U)
        print("Final objValue_L - objValue_U : %f" % (objValue_DUM_L - objValue_DUM_U))

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

