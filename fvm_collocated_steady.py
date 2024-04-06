import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab
from scipy.interpolate import RegularGridInterpolator

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

def solveMom(n, dx, rho, gamma, alphau, ub, vb, u, v, p):
    numVol = n * n
    A = np.zeros((numVol,numVol))
    bu = np.zeros((numVol,))
    bv = np.zeros((numVol,))

    Fv = (u[:,1:] + u[:,:-1]) * rho / 2 * dx
    Fh = (v[1:,:] + v[:-1,:]) * rho / 2 * dx
    

    Dw = gamma
    De = gamma
    Ds = gamma
    Dn = gamma
    
    index = np.array(range(numVol))
    
    index_north = np.arange(n*(n-1), n*n, 1)
    index_exc_north = np.delete(index, np.ravel([np.where(index == i) for i in index_north]))
    aN = Dn + np.maximum(-Fh.flatten(),0)
    A[index_exc_north, index_exc_north + n] += -aN
    A[index_north, index_north] += np.maximum(rho*vb[0]*dx, 0) + 2*gamma
    bu[index_north] = rho * np.maximum(-vb[0],0) * ub[0] * dx + 2 * ub[0] * gamma
    bv[index_north] = rho * np.maximum(-vb[0],0) * vb[0] * dx + 2 * vb[0] * gamma

    index_south = np.arange(0, n)
    index_exc_south = np.delete(index, np.ravel([np.where(index == i) for i in index_south]))
    aS = Ds + np.maximum(Fh.flatten(),0)
    A[index_exc_south, index_exc_south - n] += -aS
    A[index_south, index_south] += np.maximum(-rho*vb[2]*dx,0) + 2*gamma
    bu[index_south] = rho * np.maximum(vb[2], 0) * ub[2] * dx + 2 * ub[2] * gamma
    bv[index_south] = rho * np.maximum(vb[2], 0) * vb[2] * dx + 2 * vb[2] * gamma

    index_east = np.arange(n-1, n*(n+1)-1, n)
    index_exc_east = np.delete(index, np.ravel([np.where(index == i) for i in index_east]))
    aE = De + np.maximum(-Fv.flatten(),0)
    A[index_exc_east, index_exc_east + 1] += -aE
    A[index_east, index_east] += np.maximum(rho*ub[1]*dx, 0) + 2*gamma
    bu[index_east] = rho * np.maximum(-ub[1],0) * ub[1] * dx + 2 * ub[1] * gamma
    bv[index_east] = rho * np.maximum(-ub[1],0) * vb[1] * dx + 2 * vb[1] * gamma

    index_west = np.arange(0, n*n, n)
    index_exc_west = np.delete(index, np.ravel([np.where(index == i) for i in index_west]))
    aW = Dw + np.maximum(Fh.flatten(),0)
    A[index_exc_west, index_exc_west - 1] += -aW
    A[index_west, index_west] += np.maximum(-rho*ub[3]*dx,0) + 2*gamma
    bu[index_west] = rho * np.maximum(ub[3], 0) * ub[3] * dx + 2 * ub[3] * gamma
    bu[index_west] = rho * np.maximum(ub[3], 0) * vb[3] * dx + 2 * vb[3] * gamma
    
    A[index_exc_north, index_exc_north] += aN + Fh.flatten()
    A[index_exc_south, index_exc_south] += aS - Fh.flatten()
    A[index_exc_east, index_exc_east] += aE + Fv.flatten()
    A[index_exc_west, index_exc_west] += aW - Fv.flatten()

    dp = np.zeros((n,n))
    dp[:,1:-1] = (p[:,:-2] - p[:,2:]) * dx
    dp[:,0] = -(p[:,1] - p[:,0]) * dx / 2
    dp[:,-1] = -(p[:,-1] - p[:,-2]) * dx / 2
    bu[index] += dp.flatten()
    
    dp = np.zeros((n,n))
    dp[1:-1,:] = (p[:-2,:] - p[2:,:]) * dx / 2
    dp[0,:] = -(p[1,:] - p[0,:]) * dx / 2
    dp[-1,:] = -(p[-1,:] - p[-2,:]) * dx / 2
    bv[index] += dp.flatten()

    bu[index] += (1-alphau) * A[index, index] / alphau * u.flatten()[index]
    bv[index] += (1-alphau) * A[index, index] / alphau * v.flatten()[index]
    A[index, index] /= alphau
    
    u, exitcodeu = bicgstab(csr_matrix(A), bu, atol=1e-5)
    
    if exitcodeu == 0:
        print("U-mom Converged")
    else:
        print("U-mom Diverged")

    v, exitcodev = bicgstab(csr_matrix(A), bv, atol=1e-5)
    
    if exitcodev == 0:
        print("V-mom Converged")
    else:
        print("V-mom Diverged")
    
    #u = gaussSeidel(numVol, 1e-06, 10000, 1, phi0.flatten(), A, b)
    
    return u.reshape(n, n), v.reshape(n, n),  A[range(numVol),range(numVol)]


def solvePCorr(n, dx, rho, alphau, aC, ub, vb, ustar, vstar, p, ustarold, vstarold, ufold, vfold):
    
    dpv = (p[:,1:] - p[:,:-1]) / dx
    dph = (p[1:,:] - p[:-1,:]) / dx
    aCr = aC.reshape(n,n)
    aCv  = (aCr[:,1:] + aCr[:,:-1]) / 2
    aCh  = (aCr[1:,:] + aCr[:-1,:]) / 2

    dpdx = np.zeros((n,n))
    dpdx[:,1:-1] = (p[:,2:] - p[:,:-2]) / (2 * dx)
    dpdx[:,0] = (p[:,1] - p[:,0]) / (2 * dx)
    dpdx[:,-1] = (p[:,-1] - p[:,-2]) / (2 * dx)

    dpdy = np.zeros((n,n))
    dpdy[1:-1,:] = (p[2:,:] - p[:-2,:]) / (2 * dx)
    dpdy[0,:] = (p[1,:] - p[0,:]) / (2 * dx)
    dpdy[-1,:] = (p[-1,:] - p[-2,:]) / (2 * dx)

    b = np.zeros((n,n))

    urel = (1 - alphau) * (ufold - (ustarold[:,1:] + ustarold[:,:-1]) / 2)
    vrel = (1 - alphau) * (vfold - (vstarold[1:,:] + vstarold[:-1,:]) / 2)

    uf = ((ustar[:,1:] + ustar[:,:-1]) / 2 ) - dx*dx / aCv *  (dpv - (dpdx[:,:-1] + dpdx[:,1:])/2) + urel
    vf = ((vstar[1:,:] + vstar[:-1,:]) / 2 ) - dx*dx / aCh *  (dph - (dpdy[:-1,:] + dpdy[1:,:])/2) + vrel
    # does ac contain under relaxation?
    b[:,:-1] -= dx * uf * rho
    b[:,1:] += dx * uf * rho
    b[:-1,:] -= dx * vf * rho
    b[1:,:] += dx * vf * rho
    b[:,0] += rho * dx * ub[3]
    b[:,-1] -= rho * dx * ub[1]
    b[-1,:] -= rho * dx * vb[0]
    b[0,:] += rho * dx * vb[2]
    #b = (ustar[:,:-1] - ustar[:,1:] + vstar[:-1,:] - vstar[1:,:]) * rho * dx
    
    A = np.zeros((n*n,n*n))
    p_index_global = np.arange(0, n*n)
    
    p_index_exc_east = np.delete(p_index_global, np.ravel([np.where(p_index_global == i) for i in np.arange(n-1, n-1 + n*n, n)]))
    aE = rho * dx / (aC[p_index_exc_east] + aC[p_index_exc_east+1]) * dx * 2
    A[p_index_exc_east, p_index_exc_east+1] = -aE
    A[p_index_exc_east, p_index_exc_east] += aE

    p_index_exc_west = np.delete(p_index_global, np.ravel([np.where(p_index_global == i) for i in np.arange(0, n*n, n)]))
    aW = rho * dx / (aC[p_index_exc_west] + aC[p_index_exc_west-1]) * dx * 2
    A[p_index_exc_west, p_index_exc_west-1] = -aW
    A[p_index_exc_west, p_index_exc_west] += aW
    
    p_index_exc_north = np.delete(p_index_global, np.ravel([np.where(p_index_global == i) for i in np.arange(n*(n-1), n*n, 1)]))
    aN = rho * dx / (aC[p_index_exc_north] + aC[p_index_exc_north + n]) * dx * 2
    A[p_index_exc_north, p_index_exc_north+n] = -aN
    A[p_index_exc_north, p_index_exc_north] += aN

    p_index_exc_south = np.delete(p_index_global, np.ravel([np.where(p_index_global == i) for i in np.arange(0, n, 1)]))
    aS = rho * dx / (aC[p_index_exc_south] + aC[p_index_exc_south - n])* dx * 2
    A[p_index_exc_south, p_index_exc_south-n] = -aS
    A[p_index_exc_south, p_index_exc_south] += aS

    p, exitcode = bicgstab(csr_matrix(A), b.flatten(), atol=1e-5)
    if exitcode == 0:
        print("Pressure correction converged")
    else:
        print("Pressure correction diverged")

    return p.reshape(n,n), uf, vf

def simple(n, dx, rho, gamma, alphap):
    pstar = np.zeros((n,n))
    ustar = np.zeros((n, n))
    vstar = np.zeros((n, n))

    u = np.zeros((n, n))
    v = np.zeros((n, n))
    
    ub = [1, 0, 0, 0]
    vb = [0, 0, 0, 0]

    ustarold = np.zeros((n, n))
    vstarold = np.zeros((n, n))
    ufold = np.zeros((n, n-1))
    vfold = np.zeros((n-1, n))

    alphau = 0.5

    maxIter = 1000
    iter = 0
    while True:
        iter += 1
        print("Start new simple iteration")
        print("Solve momentum equation")

        ustar, vstar, aC = solveMom(n, dx, rho, gamma, alphau, ub, vb, u, v, pstar)

        print("Solve pressure correction equatoin")
        pcor, ufold, vfold = solvePCorr(n, dx, rho, alphau, aC, ub, vb, ustar, vstar, pstar, ustarold, vstarold, ufold, vfold)
        
        pt = pcor.reshape(n,n)

        dpdx = np.zeros((n,n))
        dpdx[:,1:-1] = (pt[:,2:] - pt[:,:-2]) / (2 * dx)
        dpdx[:,0] = (pt[:,1] - pt[:,0]) / (2 * dx)
        dpdx[:,-1] = (pt[:,-1] - pt[:,-2]) / (2 * dx)

        dpdy = np.zeros((n,n))
        dpdy[1:-1,:] = (pt[2:,:] - pt[:-2,:]) / (2 * dx)
        dpdy[0,:] = (pt[1,:] - pt[0,:]) / (2 * dx)
        dpdy[-1,:] = (pt[-1,:] - pt[-2,:]) / (2 * dx)
       
        #is ac underrelaxed
        ucor = -dx * dx / aC.reshape(n, n) * dpdx  #* alphau
        
        u = ustar + ucor
       
        vcor = -dx * dx / aC.reshape(n, n) * dpdy #* alphau
        v = vstar + vcor

        p = pstar + alphap * pt


        #divu = np.linalg.norm((u[:,:-1] - u[:,1:] + v[:-1,:] - v[1:,:]) * rho * dx)
        tol = 1e-06
        nu = np.linalg.norm(ucor.flatten())
        nv = np.linalg.norm(vcor.flatten())
        npc = np.linalg.norm(pcor.flatten())
        print(f"u: {nu} v: {nv} np: {npc}")
        #print(f"Mass imbalance: {divu}")
        #if nu < tol and nv < tol and npc < tol:
        #if divu < tol or (nu < tol and nv < tol and npc < tol):
        if (nu < tol and nv < tol and npc < tol):
            print(f"Simple converged after {iter} iterations")
            break
        elif maxIter == iter:
            print("Simple reached max iterations")
            break
        
        ustarold = copy.deepcopy(ustar)
        vstarold = copy.deepcopy(vstar)

        ustar = copy.deepcopy(u)
        vstar = copy.deepcopy(v)
        pstar = copy.deepcopy(p)

    '''
    fig = plt.figure()
    cf = plt.contourf(pcor)
    fig.colorbar(cf)
    plt.show()

    plt.figure()
    cf = plt.contourf(ustar)
    fig.colorbar(cf)
    plt.show()

    plt.figure()
    cf = plt.contourf(vstar)
    fig.colorbar(cf)
    plt.show()
    '''
    fig = plt.figure()
    cf = plt.contourf(p)
    fig.colorbar(cf)
    plt.show()

    plt.figure()
    cf = plt.contourf(ucor)
    fig.colorbar(cf)
    plt.show()

    plt.figure()
    cf = plt.contourf(vcor)
    fig.colorbar(cf)
    plt.show()
    '''
    plt.figure()
    cf = plt.contourf(ucor)
    fig.colorbar(cf)
    plt.show()

    plt.figure()
    cf = plt.contourf(vcor)
    fig.colorbar(cf)
    plt.show()
    '''

    xx = np.linspace(dx/2, 1-dx/2, n)
    interpu = RegularGridInterpolator((xx, xx), u.T)
    interpv = RegularGridInterpolator((xx, xx), v.T)

    xi = 0.5 * np.ones((n,))

    plt.figure()
    plt.plot(interpu(np.stack((xi, xx), axis=1)), xx)
    plt.xlabel("u")
    plt.ylabel("y")
    plt.show()

    plt.figure()
    plt.plot(xx, interpv(np.stack((xx, xi), axis=1)))
    plt.xlabel("x")
    plt.ylabel("v")
    plt.show()

   
def main():
    n = 30
    L = 1
    dx = L/n

    rho = 100
    gamma = 1

    #ub = [1, 0, 0, 0]
    #vb = [0, 0, 0, 0]
    #u, v, ac = solveMom(n, dx, rho, gamma, 1, ub, vb, np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)),)
   
    #solvePCorr(n, dx, rho, 0.5, ac, ub, vb, u, v, np.zeros((n,n)))
    
    simple(n, dx, rho, gamma, 0.5)
if __name__ == '__main__':
    main()