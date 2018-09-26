import math
import numpy as np
from fenics import *
from dolfin.fem.solving import *
from mshr import Circle, Rectangle, generate_mesh

def setup_grid(n_mesh):
  '''
  Quadrilateral cells.
  '''
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

def setup_geometry(geometry, finesse):
  '''
  Setup different geometries.
  '''
  if geometry == 'centered_cylinders':
    tol = 0.02  # tolerance for boundary definition
    radii = [2.0, 0.5]

    class Outer_circle(SubDomain):
        def inside(self, x, on_boundary):
            r = math.sqrt(x[0]*x[0]+x[1]*x[1])
            return near(r, radii[0], tol)

    class Inner_circle(SubDomain):
        def inside(self, x, on_boundary):
            r = math.sqrt(x[0]*x[0]+x[1]*x[1])
            return near(r, radii[1], tol)

    outer_circle = Outer_circle()
    inner_circle = Inner_circle()

    circ_large = Circle(Point(0, 0), radii[0])
    circ_small = Circle(Point(0, 0), radii[1])
    domain = circ_large - circ_small
    mesh = generate_mesh(domain, finesse)

    boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    outer_circle.mark(boundaries, 1)
    inner_circle.mark(boundaries, 2)

    return mesh, boundaries

  elif geometry == 'centered_Lshape':
    tol = 0.02  # tolerance for boundary definition

    class Hor1(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 1)

    class Vert1(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.5) and x[1] >= 0.5

    class Hor2(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.5) and x[0] <= 0.5

    class Vert2(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0)

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1)

    hor1 = Hor1()
    hor2 = Hor2()
    vert1 = Vert1()
    vert2 = Vert2()
    bottom = Bottom()
    right = Right()

    rectangle_large = Rectangle(Point(0, 0), Point(1, 1))
    rectangle_small = Rectangle(Point(0, 1), Point(0.5, 0.5))
    domain = rectangle_large - rectangle_small
    mesh = generate_mesh(domain, finesse)

    boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    hor1.mark(boundaries, 1)
    vert1.mark(boundaries, 1)
    hor2.mark(boundaries, 2)
    vert2.mark(boundaries, 2)
    bottom.mark(boundaries, 3)
    right.mark(boundaries, 4)

    return mesh, boundaries

  else:
    raise NotImplementedError
