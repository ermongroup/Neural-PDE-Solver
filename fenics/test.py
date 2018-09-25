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

def setup_mesh(bc_vec, n_mesh):
  '''
  Setup the mesh and boundaries.
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

  # Note: Ordering is important!!
  bottom.mark(boundaries, 1)
  top.mark(boundaries, 2)
  left.mark(boundaries, 3)
  right.mark(boundaries, 4)

  return mesh, boundaries

def run_fenics(bc_vec, mesh, boundaries, n_iter, para_solver, para_precon):
  '''
  Run fenics for n_iter iterations.
  Return the output x and the runtime t.
  '''
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
  size = np.sqrt(len(x)).astype(int)
  x = x.reshape((size, size))
  return x, t

def get_data(dset_path, max_temp):
  '''
  Get data.
  '''
  num = 50
  n = num // 16 + 1
  bc = np.load(os.path.join(dset_path, 'bc.npy'))[:n] / max_temp
  bc = bc.reshape((-1, 4))[:num]

  all_frames = []
  for i in range(n):
    frames = np.load(os.path.join(dset_path, 'frames', '{:04d}.npy'.format(i))) / max_temp
    all_frames.append(frames)
  frames = np.concatenate(all_frames, axis=0)[:num]
  data = {'bc': bc, 'x': frames[:, 0], 'final': frames[:, 1]}
  return data

def rms(x, gt):
  return np.sqrt(((x - gt) ** 2).mean())

def runtime(data, n_mesh, threshold, para_solver, para_precon):
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

  batch_size = data['bc'].shape[0]
  times = []
  for i in range(batch_size):
    bc = data['bc'][i]
    gt = data['final'][i]
    x = data['x'][i]
    # Initialize with 0
    x[1:-1, 1:-1] = 0

    # Set up
    mesh, boundaries = setup_mesh(bc, n_mesh)

    # Get ground truth
    gt, _ = run_fenics(bc, mesh, boundaries, 1000, para_solver, 'amg')

    starting_error = rms(x[1:-1, 1:-1], gt[1:-1, 1:-1])
    print('\n###################################################################')
    print(i)
    print('Starting error:', starting_error)
    for n_iter in n_iters:
      x, t = run_fenics(bc, mesh, boundaries, n_iter, para_solver, para_precon)
      e = rms(x[1:-1, 1:-1], gt[1:-1, 1:-1]) / starting_error
      print('Iters: {}, error: {:.4f}, time: {:.3f}'.format(n_iter, e, t))
      print('')
      if e < threshold:
        times.append(t)
        break
  return times

def main():
  n_mesh = 256
  size_str = '{}x{}'.format(n_mesh + 1, n_mesh + 1)
  dset_path = os.path.join(os.environ['HOME'], 'slowbro/PDE/heat/square', size_str)
  data = get_data(dset_path, 100)

  list_linear_solver_methods()
  list_krylov_solver_preconditioners()
  print('')

  # Parameters
  threshold = 0.01
  para_solver = 'gmres'  # specify solver: gmres, cg, bicgstab
  para_precon = 'default'  # specify preconditioner

  times = runtime(data, n_mesh, threshold, para_solver, para_precon)
  print('Threshold: {}'.format(threshold))
  print('{} examples, {:.3f} sec'.format(len(times), sum(times)))

if __name__ == '__main__':
  main()
