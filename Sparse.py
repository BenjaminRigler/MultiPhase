import numpy as np
import copy


class SparseMatrix:
    def __init__(self, A):
        self.val = A[np.where(A!=0)].flatten()
        self.rowPtr = np.cumsum(np.count_nonzero(A, axis=1))
        self.rowPtr = np.append(self.rowPtr, self.rowPtr[-1])
        self.rowPtr[1:-1] = self.rowPtr[:-2]
        self.rowPtr[0] = 0
        self.colInd = np.nonzero(A)[1]



class gausseidel:
    def solve(A, b, alpha, tol, maxIter, T0):    
        T = copy.deepcopy(T0)
        n = T.shape[0]
        iter = 0
        
        n_1 = n-1
        while True:
            iter += 1

            Told = copy.deepcopy(T)

            for i in range(n-1):
                index = A.rowPtr[i:i+2]
                values = A.val[index[0]:index[1]]
                colId = A.colInd[index[0]:index[1]]
                T[i] = T[i] + alpha * ((-np.sum(values[colId!=i] * T[colId[colId!=i]]) + b[i]) / values[colId==i][0] - T[i]) 
            
            index = A.rowPtr[n-1:n+1]
            values = A.val[index[0]:]
            colId = A.colInd[index[0]:]
            T[n_1] = T[n_1] + alpha * ((-np.sum(values[colId!=n_1] * T[colId[colId!=n_1]]) + b[n_1]) / values[colId==n_1][0] - T[n_1])
            
            
            norm = np.linalg.norm(T - Told)
            if norm < tol:
                print("Tolerance reached")
                break
            if iter == maxIter:
                print("max Iterations reached")
                break
            if iter%100 == 0:
                print(f"Res {norm}")
        
        
            
        return T
