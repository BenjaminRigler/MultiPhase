import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab
from scipy.interpolate import RegularGridInterpolator
from vtkWriter import vtkWriter


# solve u-momentum equation
def solveMomU(n, dx, rho, gamma, dt, alphau, ub, u, v, p, u0, u00, FluxC, FluxCt, FluxV):
    numVol = n * (n+1)
    A = np.zeros((numVol,numVol))
    b = np.zeros((numVol))

    # Fluxes
    Fe = (u[:,1:-1] + u[:,2:]) * rho / 2 * dx
    Fw = (u[:,:-2] + u[:,1:-1]) * rho / 2 * dx
    Fs = (v[:-2,:-1] + v[:-2,1:]) * rho / 2 * dx
    Fn = (v[1:-1,:-1] + v[1:-1,1:]) * rho / 2 * dx

    # Diffusion
    Dw = gamma
    De = gamma
    Ds = gamma
    Dn = gamma
    
    # Build A matrix using hybrid differencing scheme for convection and central differencing scheme for diffusion
    # For boundary faces use upwind scheme for convection and a pseudo ghost layer for diffusion
    # west
    index_west = np.arange(0, (n+1)*(n-1)+1, n+1)
    A[index_west, index_west] = 1
    b[index_west] = ub[3]

    # east
    index_east = np.arange(n, (n+1)*n+1, n+1)
    A[index_east, index_east] = 1
    b[index_east] = ub[1]
    
    index = np.array(range(n*(n+1)))
    index = np.delete(index, np.concatenate((index_east, index_west)))
    
    aW = np.maximum(np.maximum(Fw.flatten(),0), Dw+Fw.flatten()/2)
    aE = np.maximum(np.maximum(-Fe.flatten(),0), De-Fe.flatten()/2)
    A[index, index-1] = -aW
    A[index, index+1] = -aE

    # north
    index_north = np.arange((n+1)*(n-1)+1, (n+1)*(n-1)+n, 1)
    index_exc_north = np.delete(index, np.ravel([np.where(index == i) for i in index_north]))
    aN = np.maximum(np.maximum(-Fn.flatten(),0), Dn-Fn.flatten()/2)
    A[index_exc_north, index_exc_north + n + 1] = -aN
    vnorth = (v[-1,1:] + v[-1,:-1]) / 2
    A[index_north, index_north] += np.maximum(rho*vnorth*dx, 0) + 2*gamma
    b[index_north] = rho * np.maximum(-vnorth,0) * ub[0] * dx + 2 * ub[0] * gamma
    
    # south
    index_south = np.arange(1, n)
    index_exc_south = np.delete(index, np.ravel([np.where(index == i) for i in index_south]))
    aS = np.maximum(np.maximum(Fs.flatten(),0), Ds+Fs.flatten()/2)
    A[index_exc_south, index_exc_south - n - 1] = -aS
    vsouth = (v[0,1:] + v[0,:-1]) / 2
    A[index_south, index_south] += np.maximum(-rho*vsouth*dx,0) + 2*gamma
    b[index_south] = rho * np.maximum(vsouth, 0) * ub[2] * dx + 2 * ub[2] * gamma
      
    # center  
    aP = aE + aW + Fe.flatten() - Fw.flatten()
    A[index, index] += aP
    
    A[index_exc_north, index_exc_north] += aN + Fn.flatten()
    A[index_exc_south, index_exc_south] += aS - Fs.flatten()

    # add pressure gradient to source term
    dp = (p[:,:-1] - p[:,1:]) * dx
    b[index] += dp.flatten()
    
    # add time stepping
    A[index, index] += rho * dx * dx * FluxC
    b[index] += rho * dx * dx * u0.flatten()[index] * FluxCt + rho * dx * dx * u00.flatten()[index] * FluxV

    # add implicit underrelaxation
    b[index] += (1-alphau) * A[index, index] / alphau * u.flatten()[index]
    A[index, index] /= alphau
    
    # solve
    u, exitcode = bicgstab(csr_matrix(A), b, atol=1e-5, x0=u.flatten())

    if exitcode == 0:
        print("Converged")
    else:
        print("Diverged")
    
    return u.reshape(n, n+1), A[range(numVol),range(numVol)]

def solveMomV(n, dx, rho, gamma, dt, alphav, vb, u, v, p, v0, v00, FluxC, FluxCt, FluxV):
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
    
    aW = np.maximum(np.maximum(Fw.flatten(),0), Dw+Fw.flatten()/2)
    A[index_exc_west, index_exc_west - 1] = -aW
    uwest = (u[1:,0] + u[:-1,0]) / 2
    A[index_west, index_west] += np.maximum(-rho*uwest*dx, 0) + 2*gamma
    b[index_west] = rho * np.maximum(uwest,0) * vb[3] * dx + 2 * vb[3] * gamma
    
    index_east = np.arange(2*n-1, n*(n+1)-1, n)
    index_exc_east = np.delete(index, np.ravel([np.where(index == i) for i in index_east]))
    aE = np.maximum(np.maximum(-Fe.flatten(),0), De-Fe.flatten()/2)
    A[index_exc_east, index_exc_east + 1] = -aE
    ueast = (u[1:,-1] + u[:-1,-1]) / 2
    A[index_east, index_east] += np.maximum(rho*ueast*dx, 0) + 2*gamma
    b[index_east] = rho * np.maximum(-ueast,0) * vb[1] * dx + 2 * vb[1] * gamma
    
    aN = np.maximum(np.maximum(-Fn.flatten(),0), Dn-Fn.flatten()/2)
    aS = np.maximum(np.maximum(Fs.flatten(),0), Ds+Fs.flatten()/2)
    A[index, index+n] = -aN
    A[index, index-n] = -aS
  
    aP = aN + aS + Fn.flatten() - Fs.flatten()
    A[index, index] += aP

    A[index_exc_east, index_exc_east] += aE + Fe.flatten()
    A[index_exc_west, index_exc_west] += aW - Fw.flatten()

    dp = (p[:-1,:] - p[1:,:]) * dx
    b[index] += dp.flatten()

    # time
    A[index, index] += rho * dx * dx * FluxC
    b[index] += rho * dx * dx * v0.flatten()[index] * FluxCt + rho * dx * dx * v00.flatten()[index] * FluxV

    b[index] += (1-alphav) * A[index, index] / alphav * v.flatten()[index]
    A[index, index] /= alphav

    v, exitcode = bicgstab(csr_matrix(A), b, atol=1e-5, x0=v.flatten())

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

def simple(n, dx, rho, gamma, dt, alphap, u0, v0, u00, v00, FluxC, FluxCt, FluxV):
    pstar = np.zeros((n,n))
    ustar = copy.deepcopy(u0)
    vstar = copy.deepcopy(v0)

    u = copy.deepcopy(u0)
    v = copy.deepcopy(v0)

    u0 = copy.deepcopy(u0)
    v0 = copy.deepcopy(v0)

    maxIter = 2000
    iter = 0
    while True:
        iter += 1
        print("Start new simple iteration")
        print("Solve u momentum equation")
        ustar, aCU = solveMomU(n, dx, rho, gamma, dt, 0.5, [1, 0, 0, 0], u, v, pstar, u0, u00, FluxC, FluxCt, FluxV)
        
        print("Solve v momentum equation")# ustar here is not corrected
        vstar, aCV = solveMomV(n, dx, rho, gamma, dt, 0.5, [0, 0, 0, 0], u, v, pstar, v0, v00, FluxC, FluxCt, FluxV)

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

 

    return u, v, p, u0, v0

def runTimeLoop(n, dx, rho, gamma, CFL, numdt, u0, v0, p0):
    u0 = copy.deepcopy(u0)
    v0 = copy.deepcopy(v0)
    p0 = copy.deepcopy(p0)
    u00 = np.zeros(u0.shape)
    v00 = np.zeros(v0.shape)
    dt = computeDt(dx, u0, v0, CFL)
    FluxC = 1 / dt
    FluxCt = 1 / dt
    FluxV = 0
    time = dt
    time_ls = [0]

    vtk = vtkWriter("sim")
    vtk.writeVTK(0, n, 1, u0, v0, p0)
    u0, v0, p0, u00, v00 = simple(n, dx, rho, gamma, dt, 0.5, u0, v0, u00, v00, FluxC, FluxCt, FluxV)
    
    for i in range(1,numdt):
        dto = dt    
        dt = computeDt(dx, u0, v0, CFL)
     
        FluxC = 1/dt + 1/(dt + dto) #3/2 / dt
        FluxCt = 1/dt + 1/dto #2 / dt
        FluxV = -dt / dto / (dt + dto) #-1/2 / dt   

        u0, v0, p0, u00, v00 = simple(n, dx, rho, gamma, dt, 0.5, u0, v0, u00, v00, FluxC, FluxCt, FluxV)
        time += dt
        
        if np.linalg.norm(u0 - u00) < 1e-05 and np.linalg.norm(v0 - v00) < 1e-05:
            vtk.writeVTK(i, n, 1, u0, v0, p0)
            time_ls.append(time)
            break

        if i%50==0:
            print(np.linalg.norm(u0-u00))
            vtk.writeVTK(i, n, 1, u0, v0, p0)
            time_ls.append(time)
    
    vtk.writeSeries(time_ls)

    fig = plt.figure()
    cf = plt.contourf(u0)
    fig.colorbar(cf)
    plt.show()

    plt.figure()
    cf = plt.contourf(v0)
    fig.colorbar(cf)
    plt.show()

    plt.figure()
    cf = plt.contourf(p0)
    fig.colorbar(cf)
    plt.show()

    xx = np.linspace(0, 1, n+1)
    yy = np.linspace(dx/2, 1-dx/2, n)
    
    interpu = RegularGridInterpolator((xx, yy), u0.T)
    interpv = RegularGridInterpolator((yy, xx), v0.T)

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

def computeDt(dx, u, v, CFL):
    u = (u[:,1:] + u[:,:-1]) / 2
    v = (v[1:,:] + v[:-1,:]) / 2
    return dx * CFL / np.max(u+v) 



def main():
    n = 30
    L = 1
    dx = L/n
    CFL = 0.5
    rho = 100
    gamma = 1
    
    u0 = np.zeros((n, n+1))
    u0[-1,:] = 1
    runTimeLoop(n, dx, rho, gamma, CFL, 1001, u0, np.zeros((n+1,n)), np.zeros((n,n)))
    
if __name__ == '__main__':
    main()

