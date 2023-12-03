#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 09:00:18 2023

@author: lijun
"""

import sys
import gurobipy
import time
import numpy as np
from model.model_ED import model_ED
from model.model_DUM import model_DUM
from model.model_TAD import tad_with_service_time_dist
import xlrd
from six.moves import xrange

def load_samples_of_visit_types_from_excel(filename="..//..//data//Sample Matrix Based on Visit Type.xlsx"):
    wb = xlrd.open_workbook(filename=filename)

    sample_tensor = []
    for ws_index in xrange(len(wb.sheets())):
        ws = wb.sheet_by_index(sheetx=ws_index)
        for sample_index in xrange(1000):
            sample_matrix = []
            for position_index in xrange(10):
                v = []
                for type_index in xrange(2):
                    v.append(
                        ws.cell_value(
                            rowx=11 * sample_index + 1 + position_index,
                            colx=type_index
                        )
                    )
                sample_matrix.append(v)
            sample_tensor.append(sample_matrix)
    sample_tensor = np.asarray(a=sample_tensor, dtype=int)
    
    return sample_tensor

if __name__ == '__main__':
    BasicServiceTimeInputPath = "..//data//"

    ServiceTimeTensorFileName = "Service Time Matrix Based on Visit Type1.xlsx"

    quanrtileValueName = "0.2" # *


    I = 10      # Number of appointment positions
    J = 2       # Number of user types
    K = 300     # Training sample size
    L = 170     # Total session length of the planning horizon
    N = [7, 3]  # Number of category users
    N = np.asarray(a=N, dtype=int)
    p = [1 / K] * K
    p = np.asarray(a=p, dtype=float)

    r = {
        "0.2": [8, 15], # quantile: 0.25
        "0.3": [12, 22], # quantile: 0.32
        "0.4": [18, 25] # quantile: 0.40
    }
    for quantile_index in list(r.keys()):
        r[quantile_index] = np.asarray(a=r[quantile_index], dtype=int)

    s_all = load_samples_of_visit_types_from_excel(
        filename=BasicServiceTimeInputPath + ServiceTimeTensorFileName
    )

    s_train = s_all[: K, :, :]
    s_test = s_all[-1000: , :, :]

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


    
    try:
        start_time_TAD = time.time()
        solutionsDict_TAD, objValue_TAD = tad_with_service_time_dist(
            I=I,
            J=J,
            K=K,
            L=L,
            N=N,
            p=p,
            r=r[quanrtileValueName],
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
            r=r[quanrtileValueName],
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
            r=r[quanrtileValueName],
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
                r=r[quanrtileValueName],
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
                r=r[quanrtileValueName],
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
                r=r[quanrtileValueName],
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
    
    
