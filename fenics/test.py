from fenics import *
import numpy as np
import os
import time
from dolfin.fem.solving import *

from setup import *
from utils import *

np.random.seed(666)

def run_fenics(bc_vec, mesh, boundaries, n_iter, para_solver, para_precon):
  '''
  Run fenics for n_iter iterations.
  Return the output x and the runtime t.
  '''
  V = FunctionSpace(mesh, 'P', 1)

  # Define Dirichlet boundary conditions at four boundaries
  bc = []
  for i in range(len(bc_vec)):
    bc.append(DirichletBC(V, bc_vec[i], boundaries, i))

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
  x = u.compute_vertex_values(mesh)
  print('# points:', len(x))

  # Solver parameters
  solver.parameters['krylov_solver']['monitor_convergence'] = False
  solver.parameters['krylov_solver']['error_on_nonconvergence'] = False
  solver.parameters['krylov_solver']['maximum_iterations'] = n_iter
  solver.parameters['krylov_solver']['report'] = True
  solver.parameters['krylov_solver']['relative_tolerance'] = 0.00001
  solver.parameters['krylov_solver']['absolute_tolerance'] = 0.00001

  solver.parameters['linear_solver'] = para_solver
  solver.parameters['preconditioner'] = para_precon

  start = time.time()
  # Solve
  solver.solve()
  end = time.time()
  t = end - start

  x = u.compute_vertex_values(mesh)
  return x, t

def runtime(geometry, boundary_conditions, n_mesh, threshold, para_solver, para_precon):
  '''
  Calculate runtime.
  '''
  n_evaluation_steps = 200

  if para_precon == 'default':
    n_iters = np.arange(40, 200, 5)
  elif para_precon == 'amg':
    n_iters = np.arange(1, 11)
  else:
    n_iters = np.arange(1, 21)

  batch_size = boundary_conditions.shape[0]
  times = []
  for i in range(batch_size):
    print('\n###################################################################')
    print(i)
    bc = boundary_conditions[i]

    # Set up
    if geometry == 'square':
      mesh, boundaries = setup_grid(n_mesh)
    else:
      mesh, boundaries = setup_geometry(geometry, n_mesh)

    # Get ground truth
    gt, _ = run_fenics(bc, mesh, boundaries, 1000, para_solver, 'amg')

    starting_error = rms(gt, 0)
    for n_iter in n_iters:
      x, t = run_fenics(bc, mesh, boundaries, n_iter, para_solver, para_precon)
      e = rms(x, gt) / starting_error
      print('Iters: {}, error: {:.4f}, time: {:.3f}'.format(n_iter, e, t))
      print('')
      if e < threshold:
        times.append(t)
        break
  return times

def main():
#  geometry = 'centered_cylinders'
  geometry = 'centered_Lshape'
  num = 100

  if geometry == 'square':
    n_mesh = 256
  elif geometry == 'centered_cylinders':
    n_mesh = 180
  elif geometry == 'centered_Lshape':
    n_mesh = 230

  size_str = '{}x{}'.format(n_mesh + 1, n_mesh + 1)
  dset_path = os.path.join(os.environ['HOME'], 'slowbro/PDE/heat/', geometry, size_str)
  boundary_conditions = get_boundary_conditions(geometry, num, dset_path, max_temp=100)

  list_linear_solver_methods()
  list_krylov_solver_preconditioners()
  print('')

  # Parameters
  threshold = 0.01
  para_solver = 'gmres'  # specify solver: gmres, cg, bicgstab
  para_precon = 'amg'  # specify preconditioner

  times = runtime(geometry, boundary_conditions, n_mesh, threshold, para_solver, para_precon)
  print('Threshold: {}'.format(threshold))
  print('{} examples, {:.3f} sec'.format(len(times), sum(times)))

if __name__ == '__main__':
  main()
