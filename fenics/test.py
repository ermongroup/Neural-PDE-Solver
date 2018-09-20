from fenics import *
import numpy as np
import time
from dolfin.fem.solving import *
import matplotlib.pyplot as plt

# Create classes for defining parts of the boundaries
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)


def main():
  # Parameters
  bc_vec = [1.0, 5.0, 10.0, 20.0]  # boundary conditions in order left, top, right, bottom
  save_solution = True
  n_iter = 200  # range of iterations range(0,n_iter)
  n_repeat = 20  # repetitions for runtime measurement (arithmetic mean)
  n_mesh = 50  # governs grid resolution
  para_solver = 'gmres'  # specify solver: gmres, cg, bicgstab
  para_precon = 'none'  # specify preconditioner

  list_linear_solver_methods()
  list_krylov_solver_preconditioners()


  # Initialize sub-domain instances
  left = Left()
  top = Top()
  right = Right()
  bottom = Bottom()

  # Create mesh and define function space
  # Note: Fenics matplotlib support does not work with quadrilateral cells.
  mesh = UnitSquareMesh.create(n_mesh, n_mesh, CellType.Type.quadrilateral)

  # Initialize mesh function for boundary domains
  boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
  boundaries.set_all(0)
  left.mark(boundaries, 1)
  top.mark(boundaries, 2)
  right.mark(boundaries, 3)
  bottom.mark(boundaries, 4)

  V = FunctionSpace(mesh, 'P', 1)

  # Define Dirichlet boundary conditions at four boundaries
  bc = [DirichletBC(V, bc_vec[0], boundaries, 1),
        DirichletBC(V, bc_vec[1], boundaries, 2),
        DirichletBC(V, bc_vec[2], boundaries, 3),
        DirichletBC(V, bc_vec[3], boundaries, 4)]

  # Define variational problem
  u = TrialFunction(V)
  v = TestFunction(V)
  f = Constant(0.0)
  a = dot(grad(u), grad(v))*dx
  L = f*v*dx

  # Define solver
  u = Function(V)
  problem = LinearVariationalProblem(a, L, u, bc)
  solver = LinearVariationalSolver(problem)

  solver.parameters["krylov_solver"]["monitor_convergence"] = False
  solver.parameters['krylov_solver']['error_on_nonconvergence'] = False
  solver.parameters["krylov_solver"]["maximum_iterations"] = n_iter
  solver.parameters['krylov_solver']['report'] = True
  solver.parameters['linear_solver'] = para_solver
  solver.parameters['preconditioner'] = para_precon

  solver.solve()
  x = u.compute_vertex_values(mesh)
  x = x.reshape((n_mesh + 1, n_mesh + 1))
  plt.imshow(x)
  plt.colorbar()
  plt.show()


if __name__ == '__main__':
  main()
