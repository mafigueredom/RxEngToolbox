"""
Created on Sat May 22 20:38:46 2020
@author: SantiagoOrtiz
"""

from dolfin import XDMFFile, Mesh
import pygmsh
import meshio
import numpy as np
import os

def Generate_PBRpygmsh(xlim, ylim, f_ids, meszf=0.003, folder='pygmeshio_data'):

    # meszf = mesh element size factor
    # rectangle geometry
    xmin, xmax = xlim
    ymin, ymax = ylim
    WAid, INid = f_ids  # Wall and inlet boundaries ids.

    # Pygmsh files
    pygmsh_geo = os.path.join(folder, "PBR.geo")
    pygmsh_msh = os.path.join(folder, "PBR.msh")
    pygmsh_mesh = os.path.join(folder, "PBR_mesh.xdmf")
    pygmsh_facets = os.path.join(folder, "PBR_facets.xdmf")

    geom = pygmsh.built_in.Geometry()
    rect = geom.add_rectangle(xmin, xmax, ymin, ymax, 0.0, lcar=meszf)
    geom.add_physical([rect.line_loop.lines[0]], 10)
    geom.add_physical([rect.line_loop.lines[1]], 11)
    geom.add_physical([rect.line_loop.lines[2]], WAid)  # Wall boundary
    geom.add_physical([rect.line_loop.lines[3]], INid)  # Inlet boundary
    geom.add_physical([rect.surface], 20)

    mesh = pygmsh.generate_mesh(geom, dim=2, prune_z_0=True,
                                geo_filename=pygmsh_geo,
                                msh_filename=pygmsh_msh)

    cells = np.vstack(np.array([cells.data for cells in mesh.cells
                                if cells.type == "triangle"]))
    triangle_mesh = meshio.Mesh(points=mesh.points, cells=[("triangle", cells)])

    facet_cells = np.vstack(np.array([cells.data for cells in mesh.cells
                                      if cells.type == "line"]))
    facet_data = mesh.cell_data_dict["gmsh:physical"]["line"]
    facet_mesh = meshio.Mesh(points=mesh.points,
                             cells=[("line", facet_cells)],
                             cell_data={"name_to_read": [facet_data]})

    # Write mesh
    meshio.xdmf.write(pygmsh_mesh, triangle_mesh)
    meshio.xdmf.write(pygmsh_facets, facet_mesh)

    return pygmsh_mesh, pygmsh_facets


if __name__ == "__main__":

    xlim = 0.0, 3.0
    ylim = 0.0, 0.0127
    WAid = 12
    INid = 13
    BND_ids = WAid, INid
    meszf = 0.001 # mesh element size factor

    Domain, Facets = Generate_PBRpygmsh(xlim, ylim, BND_ids,
                                        meszf, folder='pygmeshio_test')

    mesh_D = Mesh()
    with XDMFFile(Domain) as infile:
        infile.read(mesh_D)

    mesh_F = Mesh()
    with XDMFFile(Facets) as infile:
        infile.read(mesh_F)

    print('num_cells', mesh_D.num_cells())
    print('num_vertices', mesh_D.num_vertices())
    #print('num_facets', mesh_F.num_facets())
    #print('num_edges', mesh_F.num_edges())
