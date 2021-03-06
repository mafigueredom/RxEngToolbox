"""

CASE STUDY 1: Selective oxidation of o-xylene to phthalic anhydride.

Highlights:
-Convection-diffusion-reaction simulation in single packed-bed tube.
-Two-dimensional pseudo-homogeneous mathematical model.
-Five PDEs accounting for species mass- and energy-transport equations.
-One ODE accounting for the cooling jacket model.
-Physical properties assumed to be constant.
-Catalytic rate model for V2O5 assumed of pseudo-first-order for each reagent.

A + 3.B --> C + 3.D

A: o-Xylene
B: Oxygen
C: Phthalic anhydride
D: Water

"""

# FEniCS Project Packages
from fenics import *
from dolfin import XDMFFile, Mesh, FiniteElement, MixedElement, \
                   FunctionSpace, Measure, Constant, TestFunctions, \
                   Function, split, Expression, project, dot, grad, dx, \
                   solve, as_vector, MeshValueCollection, cpp, MPI, \
                   TimeSeries, as_backend_type, assemble \

from ufl.operators import exp

# Mesh Generation and format conversion
from pygmeshio import Generate_PBRpygmsh

# General-purpose packages
import os
import datetime as dtm
import numpy as np
import math as mt

# Generation of report
from Simulation_report import Execution_time


Start_St = dtm.datetime.now()  # The start time of execution---------------------
# Time-stepping
t = 0.0
tf = 5.0                     # Final time, sec
num_steps = 500              # Number of time steps
delta_t = tf / num_steps     # Time step size, sec
dt = Constant(delta_t)

R_path = 'Results'          # Path for data storage
Data_postprocessing = True  # Data storage for later visualization & FEniCS computations.

PBR_L = 3.0       # Reactor length, m
PBR_R = 0.0127    # Reactor radius, m
meszf = 0.001     # mesh element size factor

# Mesh generation and formats conversion
Wallboundary_id = 12
Inletboundary_id = 13
Boundary_ids = Wallboundary_id, Inletboundary_id
pygmsh_mesh, pygmsh_facets = Generate_PBRpygmsh((0., PBR_L), (0., PBR_R),
                                                Boundary_ids, meszf)
# Reading mesh data stored in .xdmf files.
mesh = Mesh()
with XDMFFile(pygmsh_mesh) as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile(pygmsh_facets) as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# Define function spaces for PDEs variational formulation.
P1 = FiniteElement('P', mesh.ufl_cell(), 1)  # Lagrange 1-order polynomials family
element = MixedElement([P1, P1, P1, P1])
V = FunctionSpace(mesh, element)

# Splitting test and trial functions
v_A, v_B, v_C, v_T = TestFunctions(V)  # Test functions
u = Function(V)
u_A, u_B, u_C, u_T = split(u)          # Trial functions, time step = n+1
u_n = Function(V)                      # Trial functions, time step = n
_3u_n = Function(V)

# Retrieve boundaries marks for Robin boundary conditions.
ds_in = Measure("ds", domain=mesh,
                subdomain_data=mf, subdomain_id=Inletboundary_id)
ds_wall = Measure("ds", domain=mesh,
                  subdomain_data=mf, subdomain_id=Wallboundary_id)

"""Initial values (t == 0.0)"""
CAo = Constant(0.0)          # Initial composition for A
CBo = Constant(0.0)          # Initial composition for A
CCo = Constant(0.0)          # Initial composition for A
To = Constant(625.15)        # Initial packed-bed Temperature
Twall = Constant(625.15)    # Initial wall temperature

u_0 = Expression(('CA_init', 'CB_init', 'CC_init', 'T_init'),
                 degree=0, CA_init=CAo, CB_init=CBo, CC_init=CCo, T_init=To)
u_n = project(u_0, V)
u_An, u_Bn, u_Cn, u_Tn = split(u_n)

# Define expressions used in variational forms
R = Constant(8.314)          # Gas constant

# Velocity vector over axial coordinate.
Vz = Constant((4.0*1.6)/(3.14159*np.power(0.0254, 2)*3600.0))
w = as_vector([Vz, 0.0])

"""Inlet values"""
PA_in = Constant(0.011e5)    # Inlet pressure of A (Limiting reagent)
PB_in = Constant(0.211e5)    # Inlet pressure of B  (Excess reagent)
PC_in = Constant(0.0)        # Inlet pressure of C  (Product)

T_in = Constant(625.15)      # Reageants inlet Temperature
CA_in = PA_in/(R*T_in)       # Inlet composition of A,
CB_in = PB_in/(R*T_in)       # Inlet composition of B,
CC_in = PC_in/(R*T_in)       # Inlet composition of B,
Tcool_in = Constant(625.15)  # Coolant inlet Temperature

"""Transport properties"""
eps = Constant(0.35)         # epsilon
knt = 0.6                    # Average Conductivity R-Z
D = 0.01                     # Average diffusivity R-Z

"""Kinetic parameters"""
alfa = Constant(19.837)
beta = Constant(13636.0)
Uc1 = Constant(3.60E+10)
deltaH = Constant(-307000.0*4.183)
rob = Constant(1300.0)

"""Stoichiometry coefficients"""
cofA = Constant(-1.0)
cofB = Constant(-3.0)
cofC = Constant(1.0)

"""Coolant properties"""
rof = Constant(1.293)
cpf = Constant(992.0)
hw = Constant(96.0)          # Wall heat transfer coefficient,

""" Cooling Jacket model """
Length = Constant(3.0)
radius = Constant(0.0127)
A = 2.0*mt.pi*radius*Length
fw = Constant(0.1)
Cpw = Constant(4200.0)
roW = Constant(1000.0)
Vw = Constant(3.0*(np.power(0.0254, 2))*(4.0-(np.power(mt.pi, 2))/4.0))
Denom = ((roW*Vw*Cpw)/dt + fw*Cpw + hw*A*Length)


def Kinetic_oxy(Temperature):
    """mathematical expression of
       kinetic temperature-dependent constant"""
    k_oxy = exp(alfa) * exp(-beta/Temperature)
    return k_oxy

if Data_postprocessing:
    # Storing results data for each finite element function
    fA_out = XDMFFile(os.path.join(R_path, 'A.xdmf'))
    fB_out = XDMFFile(os.path.join(R_path, 'B.xdmf'))
    fC_out = XDMFFile(os.path.join(R_path, 'C.xdmf'))
    fD_out = XDMFFile(os.path.join(R_path, 'D.xdmf'))
    fT_out = XDMFFile(os.path.join(R_path, 'T.xdmf'))

# Variational problem definition

F_A = ((u_A - u_An)/dt)*v_A*dx + dot(w, grad(u_A))*v_A*dx + \
     eps*D*dot(grad(u_A), grad(v_A))*dx - \
     cofA*rob*Kinetic_oxy(u_Tn)*u_A*u_B*(pow(R*u_T, 2)/Uc1)*v_A*dx + \
     Vz*(u_A - CA_in)*v_A*ds_in

F_B = ((u_B - u_Bn)/dt)*v_B*dx + dot(w, grad(u_B))*v_B*dx + \
     eps*D*dot(grad(u_B), grad(v_B))*dx - \
     cofB*rob*Kinetic_oxy(u_Tn)*u_A*u_B*(pow(R*u_T, 2)/Uc1)*v_B*dx + \
     Vz*(u_B - CB_in)*v_B*ds_in

F_C = ((u_C - u_Cn)/dt)*v_C*dx + dot(w, grad(u_C))*v_C*dx + \
     eps*D*dot(grad(u_C), grad(v_C))*dx - \
     cofC*rob*Kinetic_oxy(u_Tn)*u_A*u_B*(pow(R*u_T, 2)/Uc1)*v_C*dx + \
     Vz*(u_C - CC_in)*v_C*ds_in

F_T = (rof*cpf*(u_T - u_Tn)/dt)*v_T*dx + \
      rof*cpf*dot(w, grad(u_T))*v_T*dx + \
      knt*dot(grad(u_T), grad(v_T))*dx + \
      deltaH*rob*Kinetic_oxy(u_Tn)*u_A*u_B*(pow(R*u_T, 2)/Uc1)*v_T*dx + \
      (rof*cpf*Vz)*(u_T - T_in)*v_T*ds_in + \
      hw*(u_T - Twall)*v_T*ds_wall

F = F_A + F_B + F_C + F_T


for n in range(num_steps):
    print('{} out of {}'.format(n, num_steps))
    t += delta_t  # Update current time

    # Solve variational problem for time step
    solve(F == 0, u, solver_parameters={"newton_solver": {
            "relative_tolerance": 1e-6}, "newton_solver": {
                    "maximum_iterations": 10}})
    print('solver done')

    # Save solution to files for visualization and data postprocessing
    _u_A, _u_B, _u_C, _u_T = u_n.split()

    Uvector = 3*as_backend_type(u_n.vector()).get_local()
    _3u_n.vector().set_local(Uvector)
    _3u_A, _3u_B, _3u_C, _3u_T = _3u_n.split()
    _u_D = _3u_C

    if Data_postprocessing:
        fA_out.write_checkpoint(_u_A, "A", t,  XDMFFile.Encoding.HDF5, True)
        fB_out.write_checkpoint(_u_B, "B", t,  XDMFFile.Encoding.HDF5, True)
        fC_out.write_checkpoint(_u_C, "C", t,  XDMFFile.Encoding.HDF5, True)
        fD_out.write_checkpoint(_u_D, "D", t,  XDMFFile.Encoding.HDF5, True)
        fT_out.write_checkpoint(_u_T, "T", t,  XDMFFile.Encoding.HDF5, True)

    Twall_n = Twall
    ufl_integral = assemble(_u_T*ds_wall)
    Numer = (roW*Vw*Cpw)/dt*Twall_n + fw*Cpw*Tcool_in + hw*A*ufl_integral
    Twall_new = Numer/Denom
    print('Wall temperature {} K'.format(np.around(float(Twall_new), 4)))

    Twall.assign(Twall_new)
    u_n.assign(u)

fA_out.close(); fB_out.close(); fC_out.close(); fD_out.close(); fT_out.close()
End_St = dtm.datetime.now()  # The end time of execution----------------------
Execution_time(Start_St, End_St, 'Reports', 'PhtAnh_PBR', True)
# _______________END_______________ #
