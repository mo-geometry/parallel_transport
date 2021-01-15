import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from bushings_surface import *
import random


class Spinor:
  def __init__(self, args):
      # 1) initialize
      self.t = np.linspace(0, 2 * np.pi, args.time_resolution)  # time vector
      self.θ0, self.φ0, dt = np.pi * np.random.rand(1), 2 * np.random.rand(1) * np.pi, self.t[1] - self.t[0] # initial state
      self.I, self.i, self.j, self.k = self.cayley(args)               # cayley matrices
      # 2) unitary - quaternion - rotor - surface - moving frame
      self.U, self.U_label, cigar, pancake, args = self.select_unitary(args)
      self.surface = Bushings(args, cigar=cigar, pancake=pancake)
      self.Q = self.unitary2quaternion(self.U)
      self.rotor = self.quaternion2rotor(self.Q)
      # 3) hopf coordinates from a random initial state
      self.ω, self.θ, self.φ, self.dωdt, self.dθdt, self.dφdt, self.γ, self.ξ, self.H = self.quaternion2hopf()
      self.global_grid, self.mobius = self.get_global_grid(args)
      self.r, self.drdθ, self.d2rdθ2, self.shape = self.surface.surface(self.θ, η=args.eta, λ=args.Lambda, nF=args.nF,
                                                                        cigar=cigar, pancake=pancake, derivatives=True)
      self.drdt = self.gradient0(self.r, dt)
      # 4) moving frames + parallel transport + the geometric phase
      self.R, self.dRdθ, self.dRdφ, self.dRdt, self.d2Rdt2, self.d2Rdθ2, self.d2Rdθdφ, self.d2Rdφ2 = self.path_derivatives()
      self.tangent, self.darboux, self.frenet = self.moving_frames()
      self.pt_tangent, self.pt_darboux, self.pt_frenet, self.pt_tan_plane_vec = self.parallel_transport(args)


  def quaternion2hopf(self):
      dt = self.t[1] - self.t[0]
      # HAMILTONIAN
      # quaternion shorthand
      a = self.Q[:, 0]
      b = self.Q[:, 1]
      c = self.Q[:, 2]
      d = self.Q[:, 3]
      # gradients
      a_dt = self.gradient0(self.Q[:, 0], dt)
      b_dt = self.gradient0(self.Q[:, 1], dt)
      c_dt = self.gradient0(self.Q[:, 2], dt)
      d_dt = self.gradient0(self.Q[:, 3], dt)
      # hamiltonian
      H = 2 * np.array([a * b_dt - a_dt * b - c_dt * d + c * d_dt,
                        a * c_dt - a_dt * c + b_dt * d - b * d_dt,
                        a * d_dt - a_dt * d - b_dt * c + b * c_dt]).T
      # BLOCH VECTOR
      # initial state
      b0 = np.array([np.cos(self.θ0), np.sin(self.θ0) * np.sin(self.φ0), np.sin(self.θ0) * np.cos(self.φ0)])
      B = np.matmul(self.rotor, b0).squeeze()  # evolve the bloch vector
      # hopf coordinates
      dωdt = (H[:, 1] * B[:, 1] + H[:, 2] * B[:, 2]) / (B[:, 1] ** 2 + B[:, 2] ** 2)
      dθdt = (H[:, 2] * B[:, 1] - H[:, 1] * B[:, 2]) / np.sqrt(B[:, 1] ** 2 + B[:, 2] ** 2)
      dφdt = dωdt * B[:, 0] - H[:, 0]
      # extrinsic parameters
      ω = np.cumsum(dωdt * dt, axis=0) # global phase
      θ = np.cumsum(dθdt * dt, axis=0) # polar angle
      φ = np.cumsum(dφdt * dt, axis=0) # azimuthal angle
      # initialize
      ω = ω - ω[0]
      θ = θ - θ[0] + self.θ0
      φ = φ - φ[0] + self.φ0
      # intrinsic parameters
      # S1 fibre bundle
      dγdt = dφdt * B[:, 0]
      dξdt = (H[:, 0] * B[:, 0] + H[:, 1] * B[:, 1] + H[:, 2] * B[:, 2])
      γ = np.cumsum(dγdt * dt, axis=0) - dγdt[0] * dt  # geometric phase
      ξ = np.cumsum(dξdt * dt, axis=0) - dξdt[0] * dt  # dynamic phase
      return ω, θ, φ, dωdt, dθdt, dφdt, γ, ξ, H#, Bloch


  def path_derivatives(self):
      # shorthand
      r, drdθ, d2rdθ2, θ, φ, dθdt, dφdt = self.r, self.drdθ, self.d2rdθ2, self.θ, self.φ, self.dθdt, self.dφdt
      #################
      # path
      R = np.array([r * np.cos(θ), r * np.sin(θ) * np.sin(φ), r * np.sin(θ) * np.cos(φ)])
      # first derivatives
      dRdθ = np.array([drdθ * np.cos(θ) - r * np.sin(θ),
                       (drdθ * np.sin(θ) + r * np.cos(θ)) * np.sin(φ),
                       (drdθ * np.sin(θ) + r * np.cos(θ)) * np.cos(φ)])
      dRdφ = np.array([np.zeros((len(r),)), r * np.sin(θ) * np.cos(φ), -r * np.sin(θ) * np.sin(φ)])
      dRdt = dRdφ * dφdt + dRdθ * dθdt
      # second derivatives
      d2Rdθ2 = np.array([d2rdθ2 * np.cos(θ) - 2 * drdθ * np.sin(θ) - r * np.cos(θ),
                         (d2rdθ2 * np.sin(θ) + 2 * drdθ * np.cos(θ) - r * np.sin(θ)) * np.sin(φ),
                         (d2rdθ2 * np.sin(θ) + 2 * drdθ * np.cos(θ) - r * np.sin(θ)) * np.cos(φ)])
      d2Rdθdφ = np.array([np.zeros((len(r),)),
                          (drdθ * np.sin(θ) + r * np.cos(θ)) * np.cos(φ),
                          -(drdθ * np.sin(θ) + r * np.cos(θ)) * np.sin(φ)])
      d2Rdφ2 = np.array([np.zeros((len(r),)), -r * np.sin(θ) * np.sin(φ), -r * np.sin(θ) * np.cos(φ)])
      # in time
      # d2θdt2, d2φdt2 = self.gradient0(dθdt, self.t[1] - self.t[0]), self.gradient0(dφdt, self.t[1] - self.t[0])
      # d2Rdt2 = dRdφ * d2φdt2 + d2Rdφ2 * dφdt ** 2 + 2 * d2Rdθdφ * dθdt * dφdt + d2Rdθ2 * dθdt ** 2 + dRdθ * d2θdt2
      d2Rdt2 = self.gradient1(dRdt, self.t[1] - self.t[0])
      return R, dRdθ, dRdφ, dRdt, d2Rdt2, d2Rdθ2, d2Rdθdφ, d2Rdφ2


  def moving_frames(self):
      dt = self.t[1] - self.t[0]
      # initialize frames
      tangent = np.zeros((3, 3, len(self.t)))
      darboux = np.zeros((3, 3, len(self.t)))
      frenet = np.zeros((3, 3, len(self.t)))
      # shorthand
      r, drdθ, d2rdθ2 = self.r, self.drdθ, self.d2rdθ2
      θ, φ, dθdt, dφdt = self.θ, self.φ, self.dθdt, self.dφdt
      R, dRdθ, dRdφ, dRdt, d2Rdt2 = self.R, self.dRdθ, self.dRdφ, self.dRdt, self.d2Rdt2
      # normalization coefficients
      nθ, nφ = np.sqrt(r ** 2 + drdθ ** 2), r * np.sin(θ)
      nv = np.sqrt(nθ ** 2 * dθdt ** 2 + nφ ** 2 * dφdt ** 2)   # normalize velocity
      # na = np.sqrt(np.sum(d2Rdt2 ** 2, axis=0)) # normalize acceleration
      ################
      # BASIS VECTORS
      # azimuthal vector
      eφ = np.array([np.zeros((len(φ))), np.cos(φ), -np.sin(φ)])
      # polar vector
      eθ = np.array([drdθ * np.cos(θ) - r * np.sin(θ),
                     (drdθ * np.sin(θ) + r * np.cos(θ)) * np.sin(φ),
                     (drdθ * np.sin(θ) + r * np.cos(θ)) * np.cos(φ)]) / nθ
      # surface normal: en = eφ x eθ
      en = np.array([(drdθ * np.sin(θ) + r * np.cos(θ)),
                     -(drdθ * np.cos(θ) - r * np.sin(θ)) * np.sin(φ),
                     -(drdθ * np.cos(θ) - r * np.sin(θ)) * np.cos(φ)]) / nθ
      # unit velocity: et = dRdt / nt
      ev = dRdt / nv
      # tangent normal: es = et x en
      es = (nθ / (nφ * nv)) * dθdt * dRdφ - (nφ / (nθ * nv)) * dφdt * dRdθ
      # center of force: ea = d2Rdt2 / na
      ea = self.gradient1(ev, dt) / np.sqrt(np.sum(self.gradient1(ev, dt) ** 2, axis=0))
      # binormal vector: eb = et x ea
      eb = np.cross(ev.T, ea.T).T
      ################
      # TANGENT FRAME: [azimuthal, polar, surface-normal]
      tangent[0, :, :] = eφ
      tangent[1, :, :] = eθ
      tangent[2, :, :] = en
      # DARBOUX FRAME: [velocity, bi-normal, surface normal]
      darboux[0, :, :] = ev
      darboux[1, :, :] = es
      darboux[2, :, :] = en
      ######################
      # FRENET SERRET FRAME: [velocity, acceleration, bi-normal FS]
      frenet[0, :, :] = ev
      frenet[1, :, :] = ea
      frenet[2, :, :] = eb
      #####################
      return tangent, darboux, frenet


  def parallel_transport(self, args, method="integrate"):
      # initialize parallel transport vectors
      pt_tangent = np.zeros((args.time_resolution, 3))
      pt_darboux = np.zeros((args.time_resolution, 3))
      pt_frenet = np.zeros((args.time_resolution, 3))
      pt_tan_plane_vec = np.zeros((args.time_resolution, 2))
      # initialize random vector
      init_vec = (0.5 - np.random.rand(3, )) / 2
      init_vec = init_vec / np.sqrt(np.sum(init_vec ** 2))
      tangent_vec = (0.5 - np.random.rand(2, )) / 2
      tangent_vec = tangent_vec / np.sqrt(np.sum(tangent_vec ** 2))
      # project onto the 3 moving frames
      pt_tangent[0, :] = np.matmul(self.tangent[:, :, 0], init_vec)
      pt_darboux[0, :] = np.matmul(self.darboux[:, :, 0], init_vec)
      pt_frenet[0, :] = np.matmul(self.frenet[:, :, 0], init_vec)
      pt_tan_plane_vec[0, :] = tangent_vec
      # resolve vector coefficients
      if method is "integrate" or args.moving_frame is "GEOMETRIC PHASE":  # integrate
          # time step
          dt = self.t[1] - self.t[0]
          # hamiltonians: differential basis
          tangent_matrix, G = self.tangent2exponential_matrix(self.tangent, dt)
          darboux_matrix = self.basis2exponential_matrix(self.darboux, dt)
          frenet_matrix = self.basis2exponential_matrix(self.frenet, dt)
          # generalized geometric phase for surfaces of revolution
          self.γ = -(np.cumsum(G[2, :] * dt, axis=0) - G[2, 0] * dt)
          geometric_matrix = self.parallel_transport_geometric_phase()
          for idx in range(1, args.time_resolution):
              pt_tangent[idx, :] = np.matmul(tangent_matrix[:, :, idx], pt_tangent[idx - 1, :]).T
              pt_darboux[idx, :] = np.matmul(darboux_matrix[:, :, idx], pt_darboux[idx - 1, :]).T
              pt_frenet[idx, :] = np.matmul(frenet_matrix[:, :, idx], pt_frenet[idx - 1, :]).T
              pt_tan_plane_vec[idx, :] = np.matmul(geometric_matrix[:, :, idx], tangent_vec).T
      elif method is "projection":  # projection
          for idx in range(1, args.time_resolution):
              # fill pt vectors
              pt_tangent[idx, :] = np.matmul(np.matmul(self.tangent[:, :, idx],
                                                       self.tangent[:, :, idx - 1].T), pt_tangent[idx - 1, :])
              pt_darboux[idx, :] = np.matmul(np.matmul(self.darboux[:, :, idx],
                                                       self.darboux[:, :, idx - 1].T), pt_darboux[idx - 1, :])
              pt_frenet[idx, :] = np.matmul(np.matmul(self.frenet[:, :, idx],
                                                      self.frenet[:, :, idx - 1].T), pt_frenet[idx - 1, :])
              pt_tan_plane_vec[idx, :] = np.matmul(np.matmul(self.tangent[:2, :, idx],
                                                             self.tangent[:2, :, idx - 1].T), pt_tan_plane_vec[idx - 1, :])
      return pt_tangent, pt_darboux, pt_frenet, pt_tan_plane_vec


  def parallel_transport_geometric_phase(self):
      # initialize
      rotation_matrix = np.zeros((2, 2, len(self.γ)))
      rotation_matrix[0, 0, :], rotation_matrix[0, 1, :] = np.cos(self.γ), -np.sin(self.γ)
      rotation_matrix[1, 0, :], rotation_matrix[1, 1, :] = np.sin(self.γ), np.cos(self.γ)
      return rotation_matrix


  def get_global_grid(self, args, n_pts=77, time_steps=2**12):
      t = np.linspace(0, 2*np.pi, time_steps)
      dt = t[1] - t[0]
      dim = args.global_grid_size
      H = np.zeros((time_steps, 3))
      H[:, 0] = np.interp(t, self.t, self.H[:, 0])
      H[:, 1] = np.interp(t, self.t, self.H[:, 1])
      H[:, 2] = np.interp(t, self.t, self.H[:, 2])
      # quaternion shorthand
      a = np.interp(t, self.t, self.Q[:, 0])
      b = np.interp(t, self.t, self.Q[:, 1])
      c = np.interp(t, self.t, self.Q[:, 2])
      d = np.interp(t, self.t, self.Q[:, 3])
      # GLOBAL PHASE GRID:
      θ0, φ0 = np.meshgrid(np.linspace(0, np.pi, dim[0]), np.linspace(0, 2 * np.pi, dim[1]))
      θ0, φ0 = θ0.flatten(), φ0.flatten()
      R0 = np.array([np.cos(θ0), np.sin(θ0) * np.sin(φ0), np.sin(θ0) * np.cos(φ0)]).T
      # evolve the state
      R1 = np.matmul(self.quaternion2rotor(np.array([a,b,c,d]).T), R0.T)  # evolve the bloch vector
      np.seterr(divide='ignore', invalid='ignore')  # suppress divide by 0 warning
      OMEGA = (H[:, 1].reshape(len(H[:, 1]), 1) * R1[:, 1, :] + H[:, 2].reshape(len(H[:, 1]), 1) * R1[:, 2, :]) / (
                  R1[:, 1, :] ** 2 + R1[:, 2, :] ** 2)
      O = np.sum(OMEGA * dt, axis=0)
      global_grid = O.reshape(dim[0], dim[1])
      global_grid[:, 0], global_grid[:, -1] = np.nan, np.nan  # global_grid[0, :], global_grid[-1, :] = np.nan, np.nan
      global_grid = global_grid / np.pi
      # MOBIUS band
      ω1 = np.interp(np.linspace(0, 2 * np.pi, n_pts), self.t, self.ω)  # global phase
      t1 = np.linspace(0, 2 * np.pi, n_pts)  # time
      p1 = np.linspace(-1, 1, int(args.display_frames / 4))  # band
      ω2, p2 = np.meshgrid(ω1, p1)
      t2, _ = np.meshgrid(t1, p1)
      ω2, p2, t2 = ω2.ravel(), p2.ravel(), t2.ravel()
      # delaunay
      x = (1 + 0.25 * p2 * np.cos(ω2 / 2)) * np.cos(t2)
      y = (1 + 0.25 * p2 * np.cos(ω2 / 2)) * np.sin(t2)
      z = 0.25 * p2 * np.sin(ω2 / 2)
      tri = Triangulation(np.ravel(t2), np.ravel(p2))
      # MOBIUS arrow
      arrow_start = np.array([(1 + 0.25 * p1[0] * np.cos(ω1 / 2)) * np.cos(t1),
                              (1 + 0.25 * p1[0] * np.cos(ω1 / 2)) * np.sin(t1),
                              0.25 * p1[0] * np.sin(ω1 / 2)])
      arrow_end = np.array([(1 + 0.25 * p1[-1] * np.cos(ω1 / 2)) * np.cos(t1),
                            (1 + 0.25 * p1[-1] * np.cos(ω1 / 2)) * np.sin(t1),
                            0.25 * p1[-1] * np.sin(ω1 / 2)])
      t_full = np.linspace(0, 2 * np.pi, args.display_frames)
      t_part = np.linspace(0, 2 * np.pi, n_pts)
      start_x = np.interp(t_full, t_part, arrow_start[0, :])
      start_y = np.interp(t_full, t_part, arrow_start[1, :])
      start_z = np.interp(t_full, t_part, arrow_start[2, :])
      end_x = np.interp(t_full, t_part, arrow_end[0, :])
      end_y = np.interp(t_full, t_part, arrow_end[1, :])
      end_z = np.interp(t_full, t_part, arrow_end[2, :])
      arrow = np.array([[start_x, end_x], [start_y, end_y], [start_z, end_z]])
      return global_grid, [np.ravel(x), np.ravel(y), np.ravel(z), tri, arrow]


  def select_unitary(self, args):
      # select surface
      cigar, pancake = self.select_bushings(args)
      # select moving frame
      if args.moving_frame is "random":
          args.moving_frame = random.choice(['TANGENT', 'DARBOUX', 'FRENET SERRET', 'GEOMETRIC PHASE'])
      # select unitary
      if args.select_unitary == 'random':
          x = ['equation (27)', 'equation (28)', 'equation (29)', 'equation (30)']
          U_label = random.choice(x)
      else: U_label = args.select_unitary
      if U_label == 'equation (27)':
          U = np.matmul(self.expmatrix(-self.t, self.i),
                        np.matmul(self.expmatrix(self.t / 2, self.k), self.expmatrix(self.t, self.i)))
      elif U_label == 'equation (28)':
          U = np.matmul(self.expmatrix(-self.t / 2, self.i), np.matmul(self.expmatrix(-self.t, self.j),
                                                                       self.expmatrix(-self.t, self.i)))
      elif U_label == 'equation (29)':
          U = np.matmul(self.expmatrix(-self.t, self.i), np.matmul(self.expmatrix(self.t / 2, self.k),
                                                                  np.matmul(self.expmatrix(self.t / 2, self.i),
                                                                            np.matmul(self.expmatrix(-self.t, self.j),
                                                                                      self.expmatrix(-self.t,
                                                                                                     self.i)))))
      else: # U_label == 'equation (30)':
          U = np.matmul(self.expmatrix(self.t / 2, self.i), self.expmatrix(self.t / 2, self.j))
      return U, U_label, cigar, pancake, args


  @staticmethod
  def exponential_matrix(operator):
      matrix = np.zeros(operator.shape)
      hz = operator[1, 0, :]
      hy = operator[0, 2, :]
      hx = operator[2, 1, :]
      α = np.sqrt(hx ** 2 + hy ** 2 + hz ** 2)
      # fill operator
      matrix[0, 0, :] = 2 * (hx ** 2 + (hy ** 2 + hz ** 2) * np.cos(α))
      matrix[1, 1, :] = 2 * (hy ** 2 + (hx ** 2 + hz ** 2) * np.cos(α))
      matrix[2, 2, :] = 2 * (hz ** 2 + (hx ** 2 + hy ** 2) * np.cos(α))
      matrix[1, 0, :] = 4 * np.sin(α / 2) * (hz * α * np.cos(α / 2) + hx * hy * np.sin(α / 2))
      matrix[0, 1, :] = 4 * np.sin(α / 2) * (-hz * α * np.cos(α / 2) + hx * hy * np.sin(α / 2))
      matrix[2, 0, :] = 4 * np.sin(α / 2) * (-hy * α * np.cos(α / 2) + hx * hz * np.sin(α / 2))
      matrix[0, 2, :] = 4 * np.sin(α / 2) * (hy * α * np.cos(α / 2) + hx * hz * np.sin(α / 2))
      matrix[2, 1, :] = 4 * np.sin(α / 2) * (hx * α * np.cos(α / 2) + hy * hz * np.sin(α / 2))
      matrix[1, 2, :] = 4 * np.sin(α / 2) * (-hx * α * np.cos(α / 2) + hy * hz * np.sin(α / 2))
      return matrix / (2 * α.reshape(1,1,len(α)) ** 2)


  def basis2exponential_matrix(self, basis, dt):
      operator = np.zeros(basis.shape)
      e0 = basis[0, :, :]
      e1 = basis[1, :, :]
      e2 = basis[2, :, :]
      # skew vector
      hz = np.sum(self.gradient1(e0, dt) * e1, axis=0)
      hy = np.sum(self.gradient1(e2, dt) * e0, axis=0)
      hx = np.sum(self.gradient1(e1, dt) * e2, axis=0)
      # fill operator
      operator[0, 1, :], operator[1, 0, :] = -hz, hz
      operator[0, 2, :], operator[2, 0, :] = hy, -hy
      operator[1, 2, :], operator[2, 1, :] = -hx, hx
      return self.exponential_matrix(-operator*dt)


  def tangent2exponential_matrix(self, basis, dt):
      operator = np.zeros(basis.shape)
      # operator elements
      hz = -(self.drdθ * np.sin(self.θ) + self.r * np.cos(self.θ)) * self.dφdt / np.sqrt(self.r ** 2 + self.drdθ ** 2)
      hy = -(self.drdθ * np.cos(self.θ) - self.r * np.sin(self.θ)) * self.dφdt / np.sqrt(self.r ** 2 + self.drdθ ** 2)
      hx = -(self.r ** 2 + 2 * self.drdθ ** 2 - self.r * self.d2rdθ2) * self.dθdt / (self.r ** 2 + self.drdθ ** 2)
      # fill operator
      operator[0, 1, :], operator[1, 0, :] = -hz, hz
      operator[0, 2, :], operator[2, 0, :] = hy, -hy
      operator[1, 2, :], operator[2, 1, :] = -hx, hx
      return self.exponential_matrix(-operator*dt), np.array([hx,hy,hz])


  @staticmethod
  def unitary2quaternion(U, cayley_basis='left'):
      if cayley_basis == 'left':
          q = U[:, :, 0]
      else:  # 'right'
          q = U[:, 0, :]
      return q

  @staticmethod
  def quaternion2rotor(Q):
      # quaternion shorthand
      a = Q[:, 0]
      b = Q[:, 1]
      c = Q[:, 2]
      d = Q[:, 3]
      # rotation matrix
      rotor = np.zeros((len(a), 3, 3))
      rotor[:, 0, 0] = a ** 2 + b ** 2 - c ** 2 - d ** 2
      rotor[:, 1, 1] = a ** 2 - b ** 2 + c ** 2 - d ** 2
      rotor[:, 2, 2] = a ** 2 - b ** 2 - c ** 2 + d ** 2
      rotor[:, 0, 1] = 2 * (b * c - a * d)
      rotor[:, 1, 0] = 2 * (b * c + a * d)
      rotor[:, 0, 2] = 2 * (b * d + a * c)
      rotor[:, 2, 0] = 2 * (b * d - a * c)
      rotor[:, 1, 2] = 2 * (c * d - a * b)
      rotor[:, 2, 1] = 2 * (c * d + a * b)
      return rotor

  @staticmethod
  def gradient0(q, dt):
      q_long = np.concatenate((q, q, q), axis=0)
      q_long_dt = np.gradient(q_long, dt, axis=0)
      return q_long_dt[len(q):2 * len(q)]

  @staticmethod
  def gradient1(v1, dt):
      v1_long_dt = np.gradient(np.concatenate((v1, v1, v1), axis=1), dt, axis=1)
      return v1_long_dt[:, v1.shape[1]:2 * v1.shape[1]]

  @staticmethod
  def expmatrix(theta, operator):
      theta = theta.reshape(len(theta), 1, 1)
      return np.cos(theta) * np.identity(4) + np.sin(theta) * operator

  @staticmethod
  def select_bushings(args):
      if args.sphere is True:
          return False, False
      if np.random.rand(1) < 0.5:
          return True, False
      else:
          return False, True

  @staticmethod
  def cayley(args, cayley_basis='left'):
      I = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
      if cayley_basis == 'left':  # left cayley matrices
          i = np.array([[0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, -1],
                        [0, 0, 1, 0]])
          j = np.array([[0, 0, -1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, -1, 0, 0]])
          k = np.array([[0, 0, 0, -1],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]])
      else:  # projection == 'right'
          i = np.array([[0, 1, 0, 0],
                        [-1, 0, 0, 0],
                        [0, 0, 0, -1],
                        [0, 0, 1, 0]])
          j = np.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0]])
          k = np.array([[0, 0, 0, 1],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [-1, 0, 0, 0]])
      return I, i, j, k