from fenics import *
import numpy as np
import os
import time
from dolfin.fem.solving import *

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

def setup_solver(bc_vec, n_mesh):
  '''
  Set up the solver.
  '''
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
  return mesh, u, solver

def run_fenics(mesh, u, solver, n_iter, para_solver, para_precon):
  '''
  Run fenics for n_iter iterations.
  Return the output x and the runtime t.
  '''
  solver.parameters['krylov_solver']['monitor_convergence'] = False
  solver.parameters['krylov_solver']['error_on_nonconvergence'] = False
  solver.parameters['krylov_solver']['maximum_iterations'] = n_iter
  solver.parameters['krylov_solver']['report'] = True
  solver.parameters['linear_solver'] = para_solver
  solver.parameters['preconditioner'] = para_precon

  start = time.time()
  # Solve
  solver.solve()
  end = time.time()
  t = end - start

  x = u.compute_vertex_values(mesh)
  return x, t

def get_data(dset_path, max_temp):
  '''
  Get data.
  '''
  bc = np.load(os.path.join(dset_path, 'bc.npy'))[0] / max_temp
  frames = np.load(os.path.join(dset_path, 'frames', '0000.npy')) / max_temp
  data = {'bc': bc, 'x': frames[:, 0], 'final': frames[:, 1]}
  return data

def l2_error(x, gt):
  return ((x - gt) ** 2).mean()

def runtime(data):
  '''
  Calculate runtime.
  '''
  n_mesh = 1024
  n_evaluation_steps = 200
  threshold = 0.001
  # Parameters
  para_solver = 'gmres'  # specify solver: gmres, cg, bicgstab
  para_precon = 'default'  # specify preconditioner

  batch_size = data['bc'].shape[0]
  for i in range(batch_size):
    bc = data['bc'][i]
    gt = data['final'][i]
    x = data['x'][i]
    # Initialize with 0
    x[1:-1, 1:-1] = 0
    starting_error = l2_error(x, gt)

    mesh, u, solver = setup_solver(bc, n_mesh)
    for n_iter in range(100, n_evaluation_steps, 10):
      x, t = run_fenics(mesh, u, solver, n_iter, para_solver, para_precon)
      x = x.reshape((n_mesh + 1, n_mesh + 1))
      e = l2_error(x, gt) / starting_error
      print('error:', e)
      print('Time:', t)
      if e < threshold:
        print(t)
        exit(0)

def main():
  dset_path = os.path.join(os.environ['HOME'], 'slowbro/PDE/heat/square/1025x1025')
  data = get_data(dset_path, 100)

  list_linear_solver_methods()
  list_krylov_solver_preconditioners()

  runtime(data)

if __name__ == '__main__':
  main()
