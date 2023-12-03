#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 15:15:07 2023

@author: lijun
"""

import time
import sys
import gurobipy
import numpy as np
from model.model_ED import model_ED
from model.model_DUM import model_DUM
from model.model_TAD import tad_with_service_time_dist
import xlrd
from six.moves import xrange

def load_service_time_tensor(filename, sheetname, numMatrixs):
    """Load service time matrixs (tensor) from an excel file.

    Parameters:
    ----------
    :param filename: str
    :param sheetname: str
    :param numMatrixs: int

    Return:
    ------
    :return: np.ndarray
        A tensor in shape of (k, 10, 3)
    """
    wb = xlrd.open_workbook(filename=filename)
    ws = wb.sheet_by_name(sheet_name=sheetname)

    s = []
    workline = 1
    for i in xrange(numMatrixs):
        m = []
        for row in xrange(workline, workline + 10):
            v = []
            for col in xrange(3):
                v.append(ws.cell_value(rowx=row, colx=col))
            m.append(v)
        workline += 11
        s.append(m)
    s = np.asarray(a=s, dtype=int)
    return s

if __name__ == '__main__':
    BasicInputPath = "..//data//"

    ServiceTimeTensorFileName = "Service Time Matrix Samples.xlsx"

    quanrtileValueName = "0.4" # *

    I = 10      # Number of appointment positions
    J = 3       # Number of user types
    K = 500     # Training sample size
    L = 160     # Total session length of the planning horizon
    N = [4, 3, 3]   # Number of category users
    N = np.asarray(a=N, dtype=int)
    p = [1 / K] * K
    p = np.asarray(a=p, dtype=float)

    r = {
        # "0.22": [5, 10, 9],  # quantile: 0.22
        "0.2": [7, 11, 10],  # quantile: 0.25
        "0.3": [12, 15, 13],  # quantile: 0.32
        "0.4": [18, 21, 20]  # quantile: 0.40
        # "0.45": [22, 25, 23]  # quantile: 0.45
    }
    for quantile_index in list(r.keys()):
        r[quantile_index] = np.asarray(a=r[quantile_index], dtype=int)

    s_all = load_service_time_tensor(
        filename=BasicInputPath + ServiceTimeTensorFileName,
        sheetname="Sheet1",
        numMatrixs=2000
    )  # in shape of (2000, 10, 3)

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
    
    
