import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab
from scipy.interpolate import RegularGridInterpolator
from Sparse import SparseMatrix
from Sparse import GaussSeidel


#ub = [top, east, south, west]
'''
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
'''


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

    #u = GaussSeidel.solveu(SparseMatrix(A), b, 0.2, 1e-5, 10000, u.flatten(), updateBU, n, dx, rho, gamma, ub, v)
    
    u, exitcode = bicgstab(csr_matrix(A), b, atol=1e-5)

    if exitcode == 0:
        print("Converged")
    else:
        print("Diverged")

    #u = gaussSeidel(numVol, 1e-06, 10000, 1, phi0.flatten(), A, b)
    
    return u.reshape(n, n+1), A[range(numVol),range(numVol)]

def updateBU(n, dx, rho, gamma, ub, u, v):
    Fe = (u[:,1:-1] + u[:,2:]) * rho / 2 * dx
    Fw = (u[:,:-2] + u[:,1:-1]) * rho / 2 * dx
    Fs = (v[:-2,:-1] + v[:-2,1:]) * rho / 2 * dx
    Fn = (v[1:-1,:-1] + v[1:-1,1:]) * rho / 2 * dx
    
    DWFp = 0.5
    DWFn = 0.5
    
    index_west = np.arange(0, (n+1)*(n-1)+1, n+1)
    index_east = np.arange(n, (n+1)*n+1, n+1)

    index = np.array(range(n*(n+1)))
    index = np.delete(index, np.concatenate((index_east, index_west)))

    b = np.zeros((n*(n+1),))
    b[index] += (np.maximum(Fe, 0) * u[:,1:-1] - np.maximum(-Fe, 0) * u[:,2:]).flatten()
    b[index] += (np.maximum(-Fw, 0) * u[:,1:-1] - np.maximum(Fw, 0) * u[:,:-2]).flatten()

    index_north = np.arange((n+1)*(n-1)+1, (n+1)*(n-1)+n, 1)
    index_exc_north = np.delete(index, np.ravel([np.where(index == i) for i in index_north]))
    b[index_exc_north] += (np.maximum(Fn, 0) * u[:-1,1:-1] - np.maximum(-Fn, 0) * u[1:,1:-1]).flatten()

    index_south = np.arange(1, n)
    index_exc_south = np.delete(index, np.ravel([np.where(index == i) for i in index_south]))
    b[index_exc_south] += (np.maximum(-Fs, 0) * u[1:,1:-1] - np.maximum(Fs, 0) * u[:-1,1:-1]).flatten()
    
    tol = 1e-10
    
    num = np.hstack((u[:,:1] - 2*ub[3] + u[:,1:2], u[:,1:-1] - u[:,0:-2]))
    den = np.hstack((2*u[:,1:2] - 2*ub[3], u[:,2:] - u[:,0:-2]))
    phiTildepv = grad(num, den, tol)

    num = np.hstack(((u[:,1:-1] - u[:,2:]), (u[:,-1:] - 2 * ub[1] + u[:,-2:-1])))
    den = np.hstack(((u[:,:-2] - u[:,2:]), (2*u[:,-2:-1] - 2*ub[1])))
    phiTildenv = grad(num, den, tol)

    num = np.vstack(((2 * u[0,1:-1] - 2 * ub[2]), (u[1:-1,1:-1] - u[:-2,1:-1])))
    den = np.vstack(((u[1,1:-1] + u[0,1:-1] - 2*ub[2]), (u[2:,1:-1] - u[:-2,1:-1])))
    phiTildeph = grad(num, den, tol)

    num = np.vstack(((u[1:-1,1:-1] - u[2:,1:-1]), (2 * u[-1,1:-1] - 2 * ub[0])))
    den = np.vstack(((u[0:-2, 1:-1] - u[2:,1:-1]), (u[-2,1:-1] - 2*ub[0] + u[-1,1:-1])))
    phiTildenh = grad(num, den, tol)
    
    #dwf = lambda phi : grad(phi, (2*(1-phi)), tol)
    #dwf = lambda phi: 0.5 + phi-phi
    
    Dwfpv = dwf(phiTildepv)
    Dwfnv = dwf(phiTildenv)
    Dwfph = dwf(phiTildeph)
    Dwfnh = dwf(phiTildenh)

    b[index_exc_north] -= (np.maximum(Fn, 0) * (Dwfph * u[1:,1:-1] + (1-Dwfph) * u[:-1,1:-1])).flatten()
    b[index_exc_north] += (np.maximum(-Fn, 0) * (Dwfnh * u[:-1,1:-1] + (1-Dwfnh) * u[1:,1:-1])).flatten()
    
    b[index_exc_south] -= (np.maximum(-Fs, 0) * (Dwfph * u[:-1,1:-1] + (1-Dwfph) * u[1:,1:-1])).flatten()
    b[index_exc_south] += (np.maximum(Fs, 0) * (Dwfnh * u[1:,1:-1] + (1-Dwfnh) * u[:-1,1:-1])).flatten()

    b[index] -= (np.maximum(Fe, 0) * (Dwfpv[:,1:] * u[:,2:] + (1-Dwfpv[:,1:]) * u[:,1:-1])).flatten()
    b[index] += (np.maximum(-Fe, 0) * (Dwfnv[:,1:] * u[:,1:-1] + (1-Dwfnv[:,1:]) * u[:,2:])).flatten()

    b[index] -= (np.maximum(-Fw, 0) * (Dwfpv[:,:-1] * u[:,:-2] + (1-Dwfpv[:,:-1]) * u[:,1:-1])).flatten()
    b[index] += (np.maximum(Fw, 0) * (Dwfnv[:,:-1] * u[:,1:-1] + (1-Dwfnv[:,:-1]) * u[:,:-2])).flatten()
    
    return b

def updateBV(n, dx, rho, gamma, vb, u, v):
    Fe = (u[:-1,1:-1] + u[1:,1:-1]) * rho / 2 * dx
    Fw = (u[:-1,1:-1] + u[1:,1:-1]) * rho / 2 * dx
    Fs = (v[:-2,:] + v[1:-1,:]) * rho / 2 * dx
    Fn = (v[1:-1,:] + v[2:,:]) * rho / 2 * dx
    
    DWFp = 0.5
    DWFn = 0.5
    
    index_north = np.arange(n*n, (n+1)*n, 1)
    index_south = np.arange(0, n)

    index = np.array(range(n*(n+1)))
    index = np.delete(index, np.concatenate((index_north, index_south)))
   
    b = np.zeros((n*(n+1),))
    b[index] += (np.maximum(Fn, 0) * v[1:-1,:] - np.maximum(-Fn, 0) * v[2:,:]).flatten()
    b[index] += (np.maximum(-Fs, 0) * v[1:-1,:] - np.maximum(Fs, 0) * v[:-2,:]).flatten()

    index_east = np.arange(2*n-1, n*(n+1)-1, n)
    index_exc_east = np.delete(index, np.ravel([np.where(index == i) for i in index_east]))
    b[index_exc_east] += (np.maximum(Fe, 0) * v[1:-1,:-1] - np.maximum(-Fe, 0) * v[1:-1,1:]).flatten()

    index_west = np.arange(n, n*n, n)
    index_exc_west = np.delete(index, np.ravel([np.where(index == i) for i in index_west]))
    b[index_exc_west] += (np.maximum(-Fw, 0) * v[1:-1,1:] - np.maximum(Fw, 0) * v[1:-1,:-1]).flatten()

    tol = 1e-10
    
    num = np.hstack((2*v[1:-1, 0:1] - 2*vb[3], v[1:-1, 1:-1] - v[1:-1,:-2]))
    den = np.hstack((v[1:-1,1:2] - 2*vb[3] + v[1:-1,0:1], v[1:-1,2:] - v[1:-1,:-2]))
    phiTildepv = grad(num, den, tol)
   
    num = np.hstack((v[1:-1, 1:-1] - v[1:-1, 2:], 2*v[1:-1,-1:] - 2*vb[1]))
    den = np.hstack((v[1:-1,:-2] - v[1:-1, 2:], v[1:-1,-2:-1] - 2*vb[1] + v[1:-1,-1:]))
    phiTildenv = grad(num, den, tol)
 
    num = np.vstack((v[0,:] - 2*vb[2] + v[1,:], v[1:-1, :] - v[:-2,:]))
    den = np.vstack((2*v[1,:] - 2*vb[2], v[2:,:] - v[:-2,:]))
    phiTildeph = grad(num, den, tol)
    
    num = np.vstack((v[1:-1,:] - v[2:,:], v[-1,:] - 2*vb[0] + v[-2,:]))
    den = np.vstack((v[:-2,:] - v[2:,:], 2*v[-2,:] - 2*vb[0]))
    phiTildenh = grad(num, den, tol)
    
    #dwf = lambda phi : grad(np.array([1]), np.array([2]), tol) 
    #dwf = lambda phi: 0.5 + phi-phi
    #dwf = lambda phi : grad(phi, (2*(1-phi)), tol)

    
    Dwfpv = dwf(phiTildepv)
    Dwfnv = dwf(phiTildenv)
    Dwfph = dwf(phiTildeph)
    Dwfnh = dwf(phiTildenh)


    b[index_exc_east] -= (np.maximum(Fe, 0) * (Dwfpv * v[1:-1,1:] + (1-Dwfpv) * v[1:-1,:-1])).flatten()
    b[index_exc_east] += (np.maximum(-Fe, 0) * (Dwfnv * v[1:-1,:-1] + (1-Dwfnv) * v[1:-1,1:])).flatten()
    
    b[index_exc_west] -= (np.maximum(-Fw, 0) * (Dwfpv * v[1:-1,:-1] + (1-Dwfpv) * v[1:-1,1:])).flatten()
    b[index_exc_west] += (np.maximum(Fw, 0) * (Dwfnv * v[1:-1,1:] + (1-Dwfnv) * v[1:-1,:-1])).flatten()

    b[index] -= (np.maximum(Fn, 0) * (Dwfph[1:,:] * v[2:,:] + (1-Dwfph[1:,:]) * v[1:-1,:])).flatten()
    b[index] += (np.maximum(-Fn, 0) * (Dwfnh[1:,:] * v[1:-1,:] + (1-Dwfnh[1:,:]) * v[2:,:])).flatten()

    b[index] -= (np.maximum(-Fs, 0) * (Dwfph[:-1,:] * v[:-2,:] + (1-Dwfph[-1:,:]) * v[1:-1,:])).flatten()
    b[index] += (np.maximum(Fs, 0) * (Dwfnh[:-1,:] * v[1:-1,:] + (1-Dwfnh[-1:,:]) * v[:-2,:])).flatten()

    return b

def grad(n, d, tol):
    n[(n<tol) & (n>=0)] = tol
    n[(n>-tol) & (n<=0)] = -tol

    d[(d<tol) & (d>=0)] = tol
    d[(d>-tol) & (d<=0)] = -tol
    #d[d==0] = tol
   
    return n / d
    
def dwf(phi):
        d = np.zeros(phi.shape)
        d[np.logical_and(phi <=1,phi >= 0)] = 1/2
        return d



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

    #v = GaussSeidel.solvev(SparseMatrix(A), b, 0.2, 1e-5, 1000, v.flatten(), updateBV, n, dx, rho, gamma, vb, u)
    #v = GaussSeidel.solve(SparseMatrix(A), b, 1, 1e-5, 1000, v.flatten())
    
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

    maxIter = 2000
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

        u = copy.deepcopy(ustar)
        ucor = dx / aCU.reshape(n, n+1)[:,1:-1] * (pt[:,:-1] - pt[:,1:])
        u[:,1:-1] += ucor
        
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
        if nu < tol and nv < tol and npc < tol:
        #if divu < tol or (nu < tol and nv < tol and npc < tol):
            print(f"Simple converged after {iter} iterations")
            break
        elif maxIter == iter:
            print("Simple reached max iterations")
            break

        ustar = copy.deepcopy(u)
        vstar = copy.deepcopy(v)
        pstar = copy.deepcopy(p)

    print(np.max(u), np.min(u), np.max(v), np.min(v))

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
    cf = plt.contourf(u)
    fig.colorbar(cf)
    plt.show()

    plt.figure()
    cf = plt.contourf(v)
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

    xx = np.linspace(0, 1, n+1)
    yy = np.linspace(dx/2, 1-dx/2, n)
    
    interpu = RegularGridInterpolator((xx, yy), u.T)
    interpv = RegularGridInterpolator((yy, xx), v.T)

    xi = 0.5 * np.ones((n,))
    
    plt.figure()
    plt.plot(interpu(np.stack((xi, yy), axis=1)), yy)
    plt.xlabel("y")
    plt.ylabel("u")
    plt.show()

    plt.figure()
    plt.plot(yy, interpv(np.stack((yy, xi), axis=1)))
    plt.xlabel("x")
    plt.ylabel("v")
    plt.show()
    print(np.max(interpv(np.stack((yy, xi), axis=1))))

def main():
    n = 10
    L = 1
    dx = L/n

    rho = 100
    gamma = 1
    u = np.ones((n, n+1))
    u[:,0] = 0
    u[:,1] = 1
    u[:,2] = 2
    u[:,3] = 3

    simple(n, dx, rho, gamma, 0.5)
    
    
    rng = np.random.default_rng(0)
    t = rng.integers(-10, 10, (n+1, n))
    #t[:,0] = 0
    #t[:,-1] = 0
    t[0,:] = 0
    t[-1,:] = 0
    t = t.astype(np.double)
    #updateBV(n, dx, rho, gamma, [0, 0, 0, 0], np.zeros((n,n+1)), t)
    
if __name__ == '__main__':
    main()

