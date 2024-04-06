import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab
import copy
from vtkWriter import vtkWriter
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

class FVM:
    # set simulation related parameter
    def __init__(self, n, L, rho, gamma, CFL, ttotal, ub, vb):
        self._n = n
        self._L = L
        self._rho = rho
        self._gamma = gamma
        self._CFL = CFL
        self._ub = ub
        self._vb = vb
        self._ttotal = ttotal
        
        self._dx = L / n

    # set numerical related parameter
    def setNumPar(self, alphau, alphap, tolMom, tolP, tolSimple, maxIterSimple, tolSS, scheme):
        self._alphau = alphau
        self._alphap = alphap
        self._tolMom = tolMom
        self._tolP = tolP
        self._tolSimple = tolSimple
        self._maxIterSimple = maxIterSimple
        self._tolSS = tolSS
        self._scheme = scheme

        if self._scheme == 'hybrid':
            self._coef_func = lambda F, D: np.maximum(np.maximum(-F.flatten(),0), D-F.flatten()/2)
        elif self._scheme == 'upwind':
            self._coef_func = lambda F, D: D + np.maximum(-F.flatten(), 0)
        elif self._scheme == 'central':
            self._coef_func = lambda F, D: D - F.flatten() / 2

    # set initial fields
    def initFields(self, u0, v0, p0, dtInit):
        self._u0 = u0
        self._v0 = v0
        self._p0 = p0
        self._dtInit = dtInit

    # set output parameter
    def setOutput(self, name, outputIter):
        self._name = name
        self._outputIter = outputIter

    # solve u-momentum equation
    def solveMomU(self, u, v, p, u0, u00, FluxC, FluxCt, FluxV):
        n = self._n
        gamma = self._gamma
        rho = self._rho
        dx = self._dx 
        ub = self._ub
        alphau = self._alphau

        numVol = n * (n+1)
        A = np.zeros((numVol,numVol))
        b = np.zeros((numVol))

        coef_func = self._coef_func

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
        # west bnd
        index_west = np.arange(0, (n+1)*(n-1)+1, n+1)
        A[index_west, index_west] = 1
        b[index_west] = ub[3]

        # east bnd
        index_east = np.arange(n, (n+1)*n+1, n+1)
        A[index_east, index_east] = 1
        b[index_east] = ub[1]
        
        index = np.array(range(n*(n+1)))
        index = np.delete(index, np.concatenate((index_east, index_west)))
        
        # west/east coefficients
        aW = coef_func(-Fw, Dw)
        aE = coef_func(Fe, De)
        A[index, index-1] = -aW
        A[index, index+1] = -aE

        # north
        index_north = np.arange((n+1)*(n-1)+1, (n+1)*(n-1)+n, 1)
        index_exc_north = np.delete(index, np.ravel([np.where(index == i) for i in index_north]))
        aN = coef_func(Fn, Dn)
        A[index_exc_north, index_exc_north + n + 1] = -aN
        vnorth = (v[-1,1:] + v[-1,:-1]) / 2
        A[index_north, index_north] += np.maximum(rho*vnorth*dx, 0) + 2*gamma
        b[index_north] = rho * np.maximum(-vnorth,0) * ub[0] * dx + 2 * ub[0] * gamma
        
        # south
        index_south = np.arange(1, n)
        index_exc_south = np.delete(index, np.ravel([np.where(index == i) for i in index_south]))
        aS = coef_func(-Fs, Ds)
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
        u, exitcode = bicgstab(csr_matrix(A), b, atol=self._tolMom, x0=u.flatten())

        if exitcode == 0:
            print("u-momentum converged")
        else:
            print("u-momentum diverged")
        
        return u.reshape(n, n+1), A[range(numVol),range(numVol)]
    
    # solve v-momentum equation
    def solveMomV(self, u, v, p, v0, v00, FluxC, FluxCt, FluxV):
        n = self._n
        gamma = self._gamma
        rho = self._rho
        dx = self._dx 
        vb = self._vb
        alphav = self._alphau
        
        numVol = n * (n+1)
        A = np.zeros((numVol,numVol))
        b = np.zeros((numVol))
        coef_func = self._coef_func

        # Fluxes
        Fe = (u[:-1,1:-1] + u[1:,1:-1]) * rho / 2 * dx
        Fw = (u[:-1,1:-1] + u[1:,1:-1]) * rho / 2 * dx
        Fs = (v[:-2,:] + v[1:-1,:]) * rho / 2 * dx
        Fn = (v[1:-1,:] + v[2:,:]) * rho / 2 * dx

        # Diffusion
        Dw = gamma
        De = gamma
        Ds = gamma
        Dn = gamma
        
        # Build A matrix using hybrid differencing scheme for convection and central differencing scheme for diffusion
        # For boundary faces use upwind scheme for convection and a pseudo ghost layer for diffusion
        # north bnd
        index_north = np.arange(n*n, (n+1)*n, 1)
        A[index_north, index_north] = 1
        b[index_north] = vb[0]
    
        # south bnd
        index_south = np.arange(0, n)
        A[index_south, index_south] = 1
        b[index_south] = vb[2]
        
        index = np.array(range(n*(n+1)))
        index = np.delete(index, np.concatenate((index_north, index_south)))

        # west coefficient
        index_west = np.arange(n, n*n, n)
        index_exc_west = np.delete(index, np.ravel([np.where(index == i) for i in index_west]))
        aW = coef_func(-Fw, Dw)
        A[index_exc_west, index_exc_west - 1] = -aW
        uwest = (u[1:,0] + u[:-1,0]) / 2
        A[index_west, index_west] += np.maximum(-rho*uwest*dx, 0) + 2*gamma
        b[index_west] = rho * np.maximum(uwest,0) * vb[3] * dx + 2 * vb[3] * gamma
        
        # east coefficient
        index_east = np.arange(2*n-1, n*(n+1)-1, n)
        index_exc_east = np.delete(index, np.ravel([np.where(index == i) for i in index_east]))
        aE = coef_func(Fw, De)
        A[index_exc_east, index_exc_east + 1] = -aE
        ueast = (u[1:,-1] + u[:-1,-1]) / 2
        A[index_east, index_east] += np.maximum(rho*ueast*dx, 0) + 2*gamma
        b[index_east] = rho * np.maximum(-ueast,0) * vb[1] * dx + 2 * vb[1] * gamma
        
        # north/south coefficient
        aN = coef_func(Fn, Dn)
        aS = coef_func(-Fs, Ds)
        A[index, index+n] = -aN
        A[index, index-n] = -aS
    
        # center coefficient
        aP = aN + aS + Fn.flatten() - Fs.flatten()
        A[index, index] += aP

        A[index_exc_east, index_exc_east] += aE + Fe.flatten()
        A[index_exc_west, index_exc_west] += aW - Fw.flatten()

        # add pressure gradient to source term
        dp = (p[:-1,:] - p[1:,:]) * dx
        b[index] += dp.flatten()

        # add time stepping
        A[index, index] += rho * dx * dx * FluxC
        b[index] += rho * dx * dx * v0.flatten()[index] * FluxCt + rho * dx * dx * v00.flatten()[index] * FluxV

        # add implicit under-relaxation
        b[index] += (1-alphav) * A[index, index] / alphav * v.flatten()[index]
        A[index, index] /= alphav

        # solve 
        v, exitcode = bicgstab(csr_matrix(A), b, atol=self._tolMom, x0=v.flatten())

        if exitcode == 0:
            print("v-momentum converged")
        else:
            print("v-momentum diverged")
        
        return v.reshape(n+1,n), A[range(numVol), range(numVol)]
    
    # solve pressure correction equation
    def solvePCorr(self, aCU, aCV, ustar, vstar, p):
        n = self._n
        rho = self._rho
        dx = self._dx 

        # add div(u) to source term
        b = (ustar[:,:-1] - ustar[:,1:] + vstar[:-1,:] - vstar[1:,:]) * rho * dx
        
        ## build A matrix
        A = np.zeros((n*n,n*n))
        p_index_global = np.arange(0, n*n)
        u_index_global = np.arange(0, n*(n+1))
        u_index_west = np.arange(0, (n+1)*n, n+1)
        u_index_east = np.arange(n, (n+1)*n+1, n+1)
        u_index_center = np.delete(u_index_global, np.ravel([np.where(u_index_global == i) for i in np.concatenate((u_index_east, u_index_west))]))
        
        # east coefficient
        aE = rho * dx / aCU[u_index_center] * dx
        p_index_exc_east = np.delete(p_index_global, np.ravel([np.where(p_index_global == i) for i in np.arange(n-1, n-1 + n*n, n)]))
        A[p_index_exc_east, p_index_exc_east+1] = -aE
        A[p_index_exc_east, p_index_exc_east] += aE

        # west coefficient
        aW = rho * dx / aCU[u_index_center] * dx
        p_index_exc_west = np.delete(p_index_global, np.ravel([np.where(p_index_global == i) for i in np.arange(0, n*n, n)]))
        A[p_index_exc_west, p_index_exc_west-1] = -aW
        A[p_index_exc_west, p_index_exc_west] += aW

        v_index_north = np.arange(n*n, n*(n+1), 1)
        v_index_south = np.arange(0, n, 1)
        v_index_center = np.delete(u_index_global, np.ravel([np.where(u_index_global == i) for i in np.concatenate((v_index_north, v_index_south))]))

        # north coefficient        
        p_index_exc_north = np.delete(p_index_global, np.ravel([np.where(p_index_global == i) for i in np.arange(n*(n-1), n*n, 1)]))
        aN = rho * dx / aCV[v_index_center] * dx
        A[p_index_exc_north, p_index_exc_north+n] = -aN
        A[p_index_exc_north, p_index_exc_north] += aN

        # south coefficient
        p_index_exc_south = np.delete(p_index_global, np.ravel([np.where(p_index_global == i) for i in np.arange(0, n, 1)]))
        aS = rho * dx / aCV[v_index_center] * dx
        A[p_index_exc_south, p_index_exc_south-n] = -aS
        A[p_index_exc_south, p_index_exc_south] += aS

        # solve
        p, exitcode = bicgstab(csr_matrix(A), b.flatten(), atol=self._tolP)

        if exitcode == 0:
            print("Pressure correction converged")
        else:
            print("Pressure correction diverged")

        return p.reshape(n,n)
    
    # solve NS using simple algorithm
    def simple(self, u0, v0, u00, v00, FluxC, FluxCt, FluxV):
        # set up all initial values
        n = self._n
        rho = self._rho
        dx = self._dx 
        alphap = self._alphap

        pstar = np.zeros((n,n))
        ustar = copy.deepcopy(u0)
        vstar = copy.deepcopy(v0)

        u = copy.deepcopy(u0)
        v = copy.deepcopy(v0)

        u0 = copy.deepcopy(u0)
        v0 = copy.deepcopy(v0)

        maxIter = self._maxIterSimple
        iter = 0
        # do simple iterations
        while True:
            iter += 1
            print("Start new simple iteration")
            print("Solve u momentum equation")
            ustar, aCU = self.solveMomU(u, v, pstar, u0, u00, FluxC, FluxCt, FluxV)
            
            print("Solve v momentum equation")# ustar here is not corrected
            vstar, aCV = self.solveMomV(u, v, pstar, v0, v00, FluxC, FluxCt, FluxV)

            print("Solve pressure correction equation")
            pcor = self.solvePCorr(aCU, aCV, ustar, vstar, pstar)

            pt = pcor.reshape(n,n)

            # velocity correction
            u = copy.deepcopy(ustar)
            ucor = dx / aCU.reshape(n, n+1)[:,1:-1] * (pt[:,:-1] - pt[:,1:])
            u[:,1:-1] += ucor
            
            
            v = copy.deepcopy(vstar)
            vcor = dx / aCV.reshape(n+1, n)[1:-1,:] * (pt[:-1,:] - pt[1:,:])
            v[1:-1,:] += vcor

            # pressure correction
            p = pstar + alphap * pcor

            # continuity
            divu = np.linalg.norm((u[:,:-1] - u[:,1:] + v[:-1,:] - v[1:,:]) * rho * dx)
            
            # compute Res
            tol = self._tolSimple
            nu = np.linalg.norm(ucor.flatten())
            nv = np.linalg.norm(vcor.flatten())
            npc = np.linalg.norm(pcor.flatten())
            print(f"Res: u-mom = {nu} v-mom: {nv} p: {npc}")
            print(f"Mass imbalance: {divu}")
            if nu < tol and nv < tol and npc < tol:
                print(f"Simple converged after {iter} iterations")
                break
            elif maxIter == iter:
                print("Simple reached max iterations")
                break

            ustar = copy.deepcopy(u)
            vstar = copy.deepcopy(v)
            pstar = copy.deepcopy(p)

        return u, v, p, u0, v0

    # run unsteady simulation
    def runTimeLoop(self):
        n = self._n
        tol = self._tolSS

        # get init values
        u0 = copy.deepcopy(self._u0)
        v0 = copy.deepcopy(self._v0)
        p0 = copy.deepcopy(self._p0)
        u00 = np.zeros(u0.shape)
        v00 = np.zeros(v0.shape)
        
        # do backwart euler step
        dt = self._dtInit
        FluxC = 1 / dt
        FluxCt = 1 / dt
        FluxV = 0
        time = dt
        time_ls = [0]
        vtk = vtkWriter(self._name)
        vtk.writeVTK(0, n, self._L, u0, v0, p0)
        u0, v0, p0, u00, v00 = self.simple(u0, v0, u00, v00, FluxC, FluxCt, FluxV)
        
        # do time loop
        i = 0
        while True:
            i += 1

            # set up Adams-Moulton scheme for time stepping
            dto = dt    
            dt = self.computeDt(u0, v0)
        
            FluxC = 1/dt #1/dt + 1/(dt + dto) #3/2 / dt
            FluxCt = 1/dt #1/dt + 1/dto #2 / dt
            FluxV = 0 #-dt / dto / (dt + dto) #-1/2 / dt   

            u0, v0, p0, u00, v00 = self.simple(u0, v0, u00, v00, FluxC, FluxCt, FluxV)
            time += dt
            
            # check for steady state solution
            if np.linalg.norm(u0 - u00) < tol and np.linalg.norm(v0 - v00) < tol:
                vtk.writeVTK(i, n, self._L, u0, v0, p0)
                time_ls.append(time)
                print(f"Simulation reached steady state after {i+1} iterations / {time} s")
                break

            # check if maximum time reached
            if time >= self._ttotal:
                vtk.writeVTK(i, n, self._L, u0, v0, p0)
                time_ls.append(time)
                break

            # write solution file
            if i%self._outputIter==0:
                vtk.writeVTK(i, n, self._L, u0, v0, p0)
                time_ls.append(time)
        
        vtk.writeSeries(time_ls)

    # compute timestep for given CFL number
    def computeDt(self, u, v):
        u = (u[:,1:] + u[:,:-1]) / 2
        v = (v[1:,:] + v[:-1,:]) / 2
        return self._dx * self._CFL / np.max(u+v) 
    
    # run unsteady simulation
    def runSteadyState(self):
        n = self._n

        # get init values
        u0 = copy.deepcopy(self._u0)
        v0 = copy.deepcopy(self._v0)
        p0 = copy.deepcopy(self._p0)
        u00 = np.zeros(u0.shape)
        v00 = np.zeros(v0.shape)

        u0, v0, p0, u00, v00 = self.simple(u0, v0, u00, v00, 0, 0, 0)
        
        vtk = vtkWriter(self._name)
        vtk.writeVTK(0, n, self._L, u0, v0, p0)       
        