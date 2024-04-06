This project solves the Navier-Stokes equations using the Finite Volume method for the lid driven cavity problem. For meshing a staggered cartesian grid is used. The simple algorithm is used to deal with the pressure-velocity coupling. The upwind, central and hybrid differencing schemes are supported. Both steady state and unsteady problems can be simulated. For time integration a multistep method with variable time stepping is used.

Files: The FVM class implements the solver. Furthermore there is a vtkWriter class to write results in the legacy vtk format for post processing in paraView.

The workflow of a simulation is presented in the sim.ipynb notebook.

Please note that the scripts fvm_staggered_steady.py, fvm_staggered_unsteady.py and fvm_collocated_steady.py are only prototypes and might not work!

Some results for a simulation with Re = 100 are shown next.

![](https://github.com/BenjaminRigler/MultiPhase/blob/main/post_Re100_u.png?raw=true)

![](https://github.com/BenjaminRigler/MultiPhase/blob/main/post_Re100_v.png?raw=true)

![](https://github.com/BenjaminRigler/MultiPhase/blob/main/animation_re100.gif)

