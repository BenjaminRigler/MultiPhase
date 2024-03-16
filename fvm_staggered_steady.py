import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab

#ub = [top, east, south, west]

def gaussSeidel(n, tol, maxIter, rel, T0, A, b):
    T = copy.deepcopy(T0)
    iter = 0
    while True:
        iter += 1

        Told = copy.deepcopy(T)

        for i in range(n):
            sum = np.sum([A[i,j] * T[j] for j in range(n) if (A[i,j] != 0 and i != j)])
            T[i] =  T[i] + rel*((-sum + b[i]) / A[i,i] - T[i])
        
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

def solveMomU(n, dx, rho, gamma, alphau, ub, phi0, u, v, p):
    numVol = n * (n+1)
    A = np.zeros((numVol,numVol))
    b = np.zeros((numVol))

    Fe = (u[:,1:-1] + u[:,2:]) * rho / 2 * dx
    Fw = (u[:,:-2] + u[:,1:-1]) * rho / 2 * dx
    Fs = (v[:-2,:-1] + v[:-2,1:]) * rho / 2 * dx
    Fn = (v[1:-1,:-1] + v[1:-1,1:]) * rho / 2 * dx

    Dw = gamma
    De = gamma
    Ds = gamma
    Dn = gamma
    
    index_west = np.arange(0, (n+1)*(n-1)+1, n+1)
    A[index_west, index_west] = 1
    b[index_west] = ub[3]

    index_east = np.arange(n, (n+1)*n+1, n+1)
    A[index_east, index_east] = 1
    b[index_east] = ub[1]
    
    index = np.array(range(n*(n+1)))
    index = np.delete(index, np.concatenate((index_east, index_west)))
    
    aW = Dw + np.maximum(Fw.flatten(),0)
    aE = De + np.maximum(-Fe.flatten(),0)
    A[index, index-1] = -aW
    A[index, index+1] = -aE

    index_north = np.arange((n+1)*(n-1)+1, (n+1)*(n-1)+n, 1)
    index_exc_north = np.delete(index, np.ravel([np.where(index == i) for i in index_north]))
    aN = Dn + np.maximum(-Fn.flatten(),0)
    A[index_exc_north, index_exc_north + n + 1] = -aN
    vnorth = (v[-1,1:] + v[-1,:-1]) / 2
    A[index_north, index_north] += np.maximum(rho*vnorth*dx, 0) + 2*gamma
    b[index_north] = rho * np.maximum(-vnorth,0) * ub[0] * dx + 2 * ub[0] * gamma
    
    index_south = np.arange(1, n)
    index_exc_south = np.delete(index, np.ravel([np.where(index == i) for i in index_south]))
    aS = Ds + np.maximum(Fs.flatten(),0)
    A[index_exc_south, index_exc_south - n - 1] = -aS
    vsouth = (v[0,1:] + v[0,:-1]) / 2
    A[index_south, index_south] += np.maximum(-rho*vsouth*dx,0) + 2*gamma
    b[index_south] = rho * np.maximum(vsouth, 0) * ub[2] * dx + 2 * ub[2] * gamma
      
    aP = aE + aW + Fe.flatten() - Fw.flatten()
    A[index, index] += aP
    
    A[index_exc_north, index_exc_north] += aN + Fn.flatten()
    A[index_exc_south, index_exc_south] += aS - Fs.flatten()

    dp = (p[:,:-1] - p[:,1:]) * dx
    b[index] += dp.flatten()
    
    b[index] += (1-alphau) * A[index, index] / alphau * u.flatten()[index]
    A[index, index] /= alphau

    u, exitcode = bicgstab(csr_matrix(A), b, atol=1e-5)

    if exitcode == 0:
        print("Converged")
    else:
        print("Diverged")

    #u = gaussSeidel(numVol, 1e-06, 10000, 1, phi0.flatten(), A, b)
    
    return u.reshape(n, n+1), A[range(numVol),range(numVol)]

def solveMomV(n, dx, rho, gamma, alphav, vb, phi0, u, v, p):
    numVol = n * (n+1)
    A = np.zeros((numVol,numVol))
    b = np.zeros((numVol))

    Fe = (u[:-1,1:-1] + u[1:,1:-1]) * rho / 2 * dx
    Fw = (u[:-1,1:-1] + u[1:,1:-1]) * rho / 2 * dx
    Fs = (v[:-2,:] + v[1:-1,:]) * rho / 2 * dx
    Fn = (v[1:-1,:] + v[2:,:]) * rho / 2 * dx

    Dw = gamma
    De = gamma
    Ds = gamma
    Dn = gamma
    
    index_north = np.arange(n*n, (n+1)*n, 1)
    A[index_north, index_north] = 1
    b[index_north] = vb[0]
   
    index_south = np.arange(0, n)
    A[index_south, index_south] = 1
    b[index_south] = vb[2]
    
    index = np.array(range(n*(n+1)))
    index = np.delete(index, np.concatenate((index_north, index_south)))

    index_west = np.arange(n, n*n, n)
    index_exc_west = np.delete(index, np.ravel([np.where(index == i) for i in index_west]))
    aW = Dw + np.maximum(Fw.flatten(),0)
    A[index_exc_west, index_exc_west - 1] = -aW
    uwest = (u[1:,0] + u[:-1,0]) / 2
    A[index_west, index_west] += np.maximum(-rho*uwest*dx, 0) + 2*gamma
    b[index_west] = rho * np.maximum(uwest,0) * vb[3] * dx + 2 * vb[3] * gamma
    
    index_east = np.arange(2*n-1, n*(n+1)-1, n)
    index_exc_east = np.delete(index, np.ravel([np.where(index == i) for i in index_east]))
    aE = De + np.maximum(-Fe.flatten(),0)
    A[index_exc_east, index_exc_east + 1] = -aE
    ueast = (u[1:,-1] + u[:-1,-1]) / 2
    A[index_east, index_east] += np.maximum(rho*ueast*dx, 0) + 2*gamma
    b[index_east] = rho * np.maximum(-ueast,0) * vb[1] * dx + 2 * vb[1] * gamma
    
    aN = Dn + np.maximum(-Fn.flatten(),0)
    aS = Ds + np.maximum(Fs.flatten(),0)
    A[index, index+n] = -aN
    A[index, index-n] = -aS
  
    aP = aN + aS + Fn.flatten() - Fs.flatten()
    A[index, index] += aP

    A[index_exc_east, index_exc_east] += aE + Fe.flatten()
    A[index_exc_west, index_exc_west] += aW - Fw.flatten()

    dp = (p[:-1,:] - p[1:,:]) * dx
    b[index] += dp.flatten()

    b[index] += (1-alphav) * A[index, index] / alphav * v.flatten()[index]
    A[index, index] /= alphav

    #v = gaussSeidel(numVol, 1e-06, 10000, 1, phi0.flatten(), A, b)
    
    v, exitcode = bicgstab(csr_matrix(A), b, atol=1e-5)

    if exitcode == 0:
        print("Converged")
    else:
        print("Diverged")

    return v.reshape(n+1,n), A[range(numVol), range(numVol)]

def solvePCorr(n, dx, rho, aCU, aCV, ustar, vstar, p):
    
    b = (ustar[:,:-1] - ustar[:,1:] + vstar[:-1,:] - vstar[1:,:]) * rho * dx
    
    A = np.zeros((n*n,n*n))
    p_index_global = np.arange(0, n*n)
    u_index_global = np.arange(0, n*(n+1))
    u_index_west = np.arange(0, (n+1)*n, n+1)
    u_index_east = np.arange(n, (n+1)*n+1, n+1)
    
    u_index_center = np.delete(u_index_global, np.ravel([np.where(u_index_global == i) for i in np.concatenate((u_index_east, u_index_west))]))
    p_index_exc_east = np.delete(p_index_global, np.ravel([np.where(p_index_global == i) for i in np.arange(n-1, n-1 + n*n, n)]))
    
    aE = rho * dx / aCU[u_index_center] * dx
    A[p_index_exc_east, p_index_exc_east+1] = -aE
    A[p_index_exc_east, p_index_exc_east] += aE

    aW = rho * dx / aCU[u_index_center] * dx
    p_index_exc_west = np.delete(p_index_global, np.ravel([np.where(p_index_global == i) for i in np.arange(0, n*n, n)]))
    A[p_index_exc_west, p_index_exc_west-1] = -aW
    A[p_index_exc_west, p_index_exc_west] += aW

    v_index_north = np.arange(n*n, n*(n+1), 1)
    v_index_south = np.arange(0, n, 1)
    v_index_center = np.delete(u_index_global, np.ravel([np.where(u_index_global == i) for i in np.concatenate((v_index_north, v_index_south))]))
    
    p_index_exc_north = np.delete(p_index_global, np.ravel([np.where(p_index_global == i) for i in np.arange(n*(n-1), n*n, 1)]))
    aN = rho * dx / aCV[v_index_center] * dx
    A[p_index_exc_north, p_index_exc_north+n] = -aN
    A[p_index_exc_north, p_index_exc_north] += aN

    p_index_exc_south = np.delete(p_index_global, np.ravel([np.where(p_index_global == i) for i in np.arange(0, n, 1)]))
    aS = rho * dx / aCV[v_index_center] * dx
    A[p_index_exc_south, p_index_exc_south-n] = -aS
    A[p_index_exc_south, p_index_exc_south] += aS

    p, exitcode = bicgstab(csr_matrix(A), b.flatten(), atol=1e-5)

    if exitcode == 0:
        print("Converged")
    else:
        print("Diverged")

    return p.reshape(n,n)

def simple(n, dx, rho, gamma, alphap):
    pstar = np.zeros((n,n))
    ustar = np.zeros((n, n+1))
    vstar = np.zeros((n+1, n))

    u = np.zeros((n, n+1))
    v = np.zeros((n+1, n))

    maxIter = 10000
    iter = 0
    while True:
        iter += 1
        print("Start new simple iteration")
        print("Solve u momentum equation")
        ustar, aCU = solveMomU(n, dx, rho, gamma, 0.5, [1, 0, 0, 0], ustar, u, v, pstar)
        
        print("Solve v momentum equation")# ustar here is not corrected
        vstar, aCV = solveMomV(n, dx, rho, gamma, 0.5, [0, 0, 0, 0], vstar, u, v, pstar)

        print("Solve pressure correction equatoin")
        pcor = solvePCorr(n, dx, rho, aCU, aCV, ustar, vstar, pstar)

        pt = pcor.reshape(n,n)

        #u_index_global = np.arange(0, n*(n+1))
        #u_index_west = np.arange(0, (n+1)*n, n+1)
        #u_index_east = np.arange(n, (n+1)*n+1, n+1)
        #u_index_center = np.delete(u_index_global, np.ravel([np.where(u_index_global == i) for i in np.concatenate((u_index_east, u_index_west))]))
        #ucor = dx / aCU[u_index_center] * (pt[:,:-1] - pt[:,1:]).flatten()
        u = copy.deepcopy(ustar)
        ucor = dx / aCU.reshape(n, n+1)[:,1:-1] * (pt[:,:-1] - pt[:,1:])
        u[:,1:-1] += ucor

        
        #v_index_north = np.arange(n*n, n*(n+1), 1)
        #v_index_south = np.arange(0, n, 1)
        #v_index_center = np.delete(u_index_global, np.ravel([np.where(u_index_global == i) for i in np.concatenate((v_index_north, v_index_south))]))
        v = copy.deepcopy(vstar)
        vcor = dx / aCV.reshape(n+1, n)[1:-1,:] * (pt[:-1,:] - pt[1:,:])
        v[1:-1,:] += vcor

        p = pstar + alphap * pcor


        divu = np.linalg.norm((u[:,:-1] - u[:,1:] + v[:-1,:] - v[1:,:]) * rho * dx)
        tol = 1e-06
        nu = np.linalg.norm(ucor.flatten())
        nv = np.linalg.norm(vcor.flatten())
        npc = np.linalg.norm(pcor.flatten())
        print(f"u: {nu} v: {nv} np: {npc}")
        print(f"Mass imbalance: {divu}")
        #if nu < tol and nv < tol and npc < tol:
        if divu < tol or (nu < tol and nv < tol and npc < tol):
            print(f"Simple converged after {iter} iterations")
            break
        elif maxIter == iter:
            print("Simple reached max iterations")
            break

        ustar = copy.deepcopy(u)
        vstar = copy.deepcopy(v)
        pstar = copy.deepcopy(p)

    plt.figure()
    plt.contourf(p)
    plt.show()

    plt.figure()
    plt.contourf(u)
    plt.show()

    plt.figure()
    plt.contourf(v)
    plt.show()

def main():
    n = 50
    L = 1
    dx = L/n

    rho = 10
    gamma = 1

    simple(n, dx, rho, gamma, 0.4)

if __name__ == '__main__':
    main()