import os
import numpy as np
import shutil

# write results in legacy vtk format
class vtkWriter:
    # initialize writer object
    def __init__(self, name):
        self._name = name
        rel_path = os.path.join(name, name+'_')
        script_dir = os.path.dirname(__file__)
        self._abs_file_path = os.path.join(script_dir, rel_path)

        if name not in os.listdir():
            os.mkdir(os.path.join(script_dir, name))
        else:
            shutil.rmtree(os.path.join(script_dir, name))
            os.mkdir(os.path.join(script_dir, name))

    # write results file
    def writeVTK(self, i, n, L, u, v, p):
        f = open(self._abs_file_path+str(i)+".vtk", "w")
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Sim data\n")
        f.write("ASCII\n\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {n+1} {n+1} {1}\n")
        f.write(f"POINTS {(n+1)*(n+1)} double\n")  

        xx = np.linspace(0, L, n+1)
        for x in xx:
            for y in xx:
                f.write(f"{x} {y} {0}\n")

        u = ((u[:,:-1] + u[:,1:]) / 2).flatten()
        v = ((v[:-1,:] + v[1:,:]) / 2).flatten()
        p = p.flatten()

        f.write("\n")
        f.write(f"CELL_DATA {n*n}")
        f.write("SCALARS p double\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(n):
            for j in range(n):
                f.write(f"{p[i + j * n]}\n")

        f.write("VECTORS U double\n")
        for i in range(n):
            for j in range(n):
                f.write(f"{u[i+j*n]} {v[i+j*n]} {0}\n")

        f.close()

    # write series file
    def writeSeries(self, time):
        f = open(self._abs_file_path[:-1] + ".vtk.series", "w")
        f.write("{\n")
        f.write("\t\"file-series-version\" : \"1.0\",\n")
        f.write("\t\"files\" : [\n")

        files = (os.listdir(os.path.join(os.path.dirname(__file__))+"/" + self._name))
        files = [f for f in files if ".series" not in f]
        files = sorted(files, key = lambda x:int(x[len(self._name)+1:-4]))

        for file, t in zip(files[:-1], time[:-1]):
            f.write("\t\t{ \"name\" : \"" + file + "\", \"time\" : " + str(round(t, 2)) + " },\n")

        f.write("\t\t{ \"name\" : \"" + files[-1] + "\", \"time\" : " + str(round(time[-1], 2)) + " }\n")

        f.write("\t]\n")
        f.write("}")
    