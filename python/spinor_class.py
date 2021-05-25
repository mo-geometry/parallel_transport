import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from bushings_surface import *
import random


class Spinor:
  def __init__(self, args):
      # 1) initialize
      self.t = np.linspace(0, 2 * np.pi, args.time_resolution)          # time vector
      self.φ0, self.θ0, self.ω0, self.dt = self.initialize_spinor(args)
      self.I, self.i, self.j, self.k = self.cayley()                    # cayley matrices
      self.πI, self.πi, self.πj, self.πk = self.lie_algebra()           # Lie algebra matrices
      # 2) unitary - quaternion - rotor - surface
      self.U, self.U_label, cigar, pancake, args = self.select_unitary(args)
      self.surface = Bushings(args, cigar=cigar, pancake=pancake)
      self.Q = self.unitary2quaternion()
      self.rotor = self.quaternion2rotor(self.Q)
      # 3) spinor
      self.Ψ, self.H, self.φ, self.θ, self.ω, self.dφdt, self.dθdt, self.dωdt = self.fill_spinor()
      self.ν0, self.μ0, self.ν, self.μ, self.Ω, self.dνdt, self.dμdt, self.dΩdt = self.project_bundle(args)
      self.global_grid, self.mobius = self.get_global_grid(args)
      self.r, self.drdμ, self.d2rdμ2, \
      self.surface_shape = self.surface.surface(self.μ, cigar=cigar, pancake=pancake,
                                                η=args.eta, λ=args.Lambda, nF=args.nF,
                                                derivatives=True)
      self.drdt = self.gradient0(self.r, self.dt)
      # 4) moving frames + parallel transport + the geometric phase
      self.R, self.dRdμ, self.dRdν, self.dRdt, \
      self.d2Rdt2, self.d2Rdμ2, self.d2Rdμdν, self.d2Rdν2 = self.path_derivatives()
      self.tangent, self.darboux, self.frenet = self.moving_frames()
      self.pt_tangent, self.pt_darboux, self.pt_frenet, self.pt_tan_plane_vec, \
      self.γ, self.ξ = self.parallel_transport(args)


  def project_bundle(self, args):
      if args.projection=='random':
          args.projection = random.choice(['i', 'j', 'k'])
      if args.projection=='i':
          Ri, Rj, Rk = self.Ψ[:, 0, 0], self.Ψ[:, 1, 0], self.Ψ[:, 2, 0]
      elif args.projection == 'j':
          Ri, Rj, Rk = self.Ψ[:, 0, 1], self.Ψ[:, 1, 1], self.Ψ[:, 2, 1]
      elif args.projection == 'k':
          Ri, Rj, Rk = self.Ψ[:, 0, 2], self.Ψ[:, 1, 2], self.Ψ[:, 2, 2]
      Hi, Hj, Hk = self.H[:, 0], self.H[:, 1], self.H[:, 2]  # bloch vector
      # hopf coordinates
      dΩdt = (Hj * Rj + Hk * Rk) / (Rj ** 2 + Rk ** 2)
      dμdt = (Hk * Rj - Hj * Rk) / np.sqrt(Rj ** 2 + Rk ** 2)
      dνdt = dΩdt * Ri - Hi
      μ0, ν0 = np.arccos(Ri[0]), np.arctan2(Rj[0], Rk[0])
      if ν0 < 0:  ν0 = ν0 + 2 * np.pi                       # range is [0, 2 * np.pi]
      Ω = np.cumsum(dΩdt * self.dt, axis=0) - dΩdt[0] * self.dt
      μ = np.cumsum(dμdt * self.dt, axis=0) - dμdt[0] * self.dt + μ0
      ν = np.cumsum(dνdt * self.dt, axis=0) - dνdt[0] * self.dt + ν0
      # # check
      # R1 = np.array([np.cos(μ), np.sin(μ) * np.sin(ν), np.sin(μ) * np.cos(ν)])
      # R2 = R1 - np.array([Ri, Rj, Rk])
      return ν0, μ0, ν, μ, Ω, dνdt, dμdt, dΩdt


  def fill_spinor(self):
      # HAMILTONIAN
      # quaternion shorthand
      a = self.Q[:, 0]
      b = self.Q[:, 1]
      c = self.Q[:, 2]
      d = self.Q[:, 3]
      # gradients
      a_dt = self.gradient0(self.Q[:, 0], self.dt)
      b_dt = self.gradient0(self.Q[:, 1], self.dt)
      c_dt = self.gradient0(self.Q[:, 2], self.dt)
      d_dt = self.gradient0(self.Q[:, 3], self.dt)
      # hamiltonian
      H = 2 * np.array([a * b_dt - a_dt * b - c_dt * d + c * d_dt,
                        a * c_dt - a_dt * c + b_dt * d - b * d_dt,
                        a * d_dt - a_dt * d - b_dt * c + b * c_dt]).T
      # initial state
      φ0, θ0, ω0 = self.φ0, self.θ0, self.ω0
      # SPINOR
      Ψ, Ψ0 = np.zeros(self.rotor.shape), np.zeros((3, 3))
      Ψ0[0, 0], Ψ0[0, 1], Ψ0[0, 2] = np.cos(θ0), -np.sin(θ0) * np.sin(ω0), -np.sin(θ0) * np.cos(ω0)
      Ψ0[1, 0], Ψ0[2, 0] = np.sin(θ0) * np.sin(φ0), np.sin(θ0) * np.cos(φ0)
      Ψ0[1, 1] = np.cos(φ0) * np.cos(ω0) + np.sin(φ0) * np.cos(θ0) * np.sin(ω0)
      Ψ0[2, 1] = -np.sin(φ0) * np.cos(ω0) + np.cos(φ0) * np.cos(θ0) * np.sin(ω0)
      Ψ0[1, 2] = -np.cos(φ0) * np.sin(ω0) + np.sin(φ0) * np.cos(θ0) * np.cos(ω0)
      Ψ0[2, 2] = np.sin(φ0) * np.sin(ω0) + np.cos(φ0) * np.cos(θ0) * np.cos(ω0)
      # VALIDATE PROJECTIONS
      Ψ1 = np.matmul(self.rotor, Ψ0).squeeze()
      # BLOCH VECTOR
      Ii, Ij, Ik, Hi, Hj, Hk = Ψ1[:, 0, 0], Ψ1[:, 1, 0], Ψ1[:, 2, 0], H[:, 0], H[:, 1], H[:, 2]  # bloch vector
      # hopf coordinates
      dωdt = (Hj * Ij + Hk * Ik) / (Ij ** 2 + Ik ** 2)
      dθdt = (Hk * Ij - Hj * Ik) / np.sqrt(Ij ** 2 + Ik ** 2)
      dφdt = dωdt * Ii - Hi
      # extrinsic parameters
      ω = np.cumsum(dωdt * self.dt, axis=0)  # omega
      θ = np.cumsum(dθdt * self.dt, axis=0)  # theta
      φ = np.cumsum(dφdt * self.dt, axis=0)  # phi
      # as per initial state
      φ = φ - φ[0] + φ0
      θ = θ - θ[0] + θ0
      ω = ω - ω[0] + ω0
      # # time dependent
      # Ψ[:, 0, 0], Ψ[:, 0, 1], Ψ[:, 0, 2] = np.cos(θ), -np.sin(θ) * np.sin(ω), -np.sin(θ) * np.cos(ω)
      # Ψ[:, 1, 0], Ψ[:, 2, 0] = np.sin(θ) * np.sin(φ), np.sin(θ) * np.cos(φ)
      # Ψ[:, 1, 1] = np.cos(φ) * np.cos(ω) + np.sin(φ) * np.cos(θ) * np.sin(ω)
      # Ψ[:, 2, 1] = -np.sin(φ) * np.cos(ω) + np.cos(φ) * np.cos(θ) * np.sin(ω)
      # Ψ[:, 1, 2] = -np.cos(φ) * np.sin(ω) + np.sin(φ) * np.cos(θ) * np.cos(ω)
      # Ψ[:, 2, 2] = np.sin(φ) * np.sin(ω) + np.cos(φ) * np.cos(θ) * np.cos(ω)
      return Ψ1, H, φ, θ, ω, dφdt, dθdt, dωdt


  def initialize_spinor(self, args):
      dt = self.t[1] - self.t[0]
      if args.spinor_initial_state_deg is None:
          φ0, θ0, ω0 = 2 * np.pi * np.random.rand(1), np.pi * np.random.rand(1), 2 * np.pi * np.random.rand(1)
      else:
          v = args.spinor_initial_state_deg
          φ0, θ0, ω0 = v[0] * np.pi / 180, v[1] * np.pi / 180, v[2] * np.pi / 180
      return φ0, θ0, ω0, dt


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
      ω = ω - ω[0] + self.ω0
      θ = θ - θ[0] + self.θ0
      φ = φ - φ[0] + self.φ0
      # intrinsic parameters
      # S1 fibre bundle
      dγdt = dφdt * np.cos(θ)
      Rx, Ry, Rz = np.cos(θ), np.sin(θ) * np.sin(φ), np.sin(θ) * np.cos(φ)
      dξdt = (H[:, 0] * Rx + H[:, 1] * Ry + H[:, 2] * Rz)
      γ = np.cumsum(dγdt * dt, axis=0) - dγdt[0] * dt  # geometric phase
      ξ = np.cumsum(dξdt * dt, axis=0) - dξdt[0] * dt  # dynamic phase
      return ω, θ, φ, dωdt, dθdt, dφdt, γ, ξ, H #, Bloch


  def path_derivatives(self):
      # shorthand
      r, drdμ, d2rdμ2, μ, ν, dμdt, dνdt = self.r, self.drdμ, self.d2rdμ2, self.μ, self.ν, self.dμdt, self.dνdt
      #################
      # path
      R = np.array([r * np.cos(μ), r * np.sin(μ) * np.sin(ν), r * np.sin(μ) * np.cos(ν)])
      # first derivatives
      dRdμ = np.array([drdμ * np.cos(μ) - r * np.sin(μ),
                       (drdμ * np.sin(μ) + r * np.cos(μ)) * np.sin(ν),
                       (drdμ * np.sin(μ) + r * np.cos(μ)) * np.cos(ν)])
      dRdν = np.array([np.zeros((len(r),)), r * np.sin(μ) * np.cos(ν), -r * np.sin(μ) * np.sin(ν)])
      dRdt = dRdν * dνdt + dRdμ * dμdt
      # second derivatives
      d2Rdμ2 = np.array([d2rdμ2 * np.cos(μ) - 2 * drdμ * np.sin(μ) - r * np.cos(μ),
                         (d2rdμ2 * np.sin(μ) + 2 * drdμ * np.cos(μ) - r * np.sin(μ)) * np.sin(ν),
                         (d2rdμ2 * np.sin(μ) + 2 * drdμ * np.cos(μ) - r * np.sin(μ)) * np.cos(ν)])
      d2Rdμdν = np.array([np.zeros((len(r),)),
                          (drdμ * np.sin(μ) + r * np.cos(μ)) * np.cos(ν),
                          -(drdμ * np.sin(μ) + r * np.cos(μ)) * np.sin(ν)])
      d2Rdν2 = np.array([np.zeros((len(r),)), -r * np.sin(μ) * np.sin(ν), -r * np.sin(μ) * np.cos(ν)])
      # in time
      # d2μdt2, d2νdt2 = self.gradient0(dμdt, self.t[1] - self.t[0]), self.gradient0(dνdt, self.t[1] - self.t[0])
      # d2Rdt2 = dRdν * d2νdt2 + d2Rdν2 * dνdt ** 2 + 2 * d2Rdμdν * dμdt * dνdt + d2Rdμ2 * dμdt ** 2 + dRdμ * d2μdt2
      d2Rdt2 = self.gradient1(dRdt, self.t[1] - self.t[0])
      return R, dRdμ, dRdν, dRdt, d2Rdt2, d2Rdμ2, d2Rdμdν, d2Rdν2


  def moving_frames(self):
      dt = self.t[1] - self.t[0]
      # initialize frames
      tangent = np.zeros((3, 3, len(self.t)))
      darboux = np.zeros((3, 3, len(self.t)))
      frenet = np.zeros((3, 3, len(self.t)))
      # shorthand
      r, drdμ, d2rdμ2 = self.r, self.drdμ, self.d2rdμ2
      μ, ν, dμdt, dνdt = self.μ, self.ν, self.dμdt, self.dνdt
      R, dRdμ, dRdν, dRdt, d2Rdt2 = self.R, self.dRdμ, self.dRdν, self.dRdt, self.d2Rdt2
      # normalization coefficients
      nμ, nν = np.sqrt(r ** 2 + drdμ ** 2), r * np.sin(μ)
      nv = np.sqrt(nμ ** 2 * dμdt ** 2 + nν ** 2 * dνdt ** 2)   # normalize velocity
      # na = np.sqrt(np.sum(d2Rdt2 ** 2, axis=0)) # normalize acceleration
      ################
      # BASIS VECTORS
      # azimuthal vector
      eν = np.array([np.zeros((len(ν))), np.cos(ν), -np.sin(ν)])
      # polar vector
      eμ = np.array([drdμ * np.cos(μ) - r * np.sin(μ),
                     (drdμ * np.sin(μ) + r * np.cos(μ)) * np.sin(ν),
                     (drdμ * np.sin(μ) + r * np.cos(μ)) * np.cos(ν)]) / nμ
      # surface normal: en = eν x eμ
      en = np.array([(drdμ * np.sin(μ) + r * np.cos(μ)),
                     -(drdμ * np.cos(μ) - r * np.sin(μ)) * np.sin(ν),
                     -(drdμ * np.cos(μ) - r * np.sin(μ)) * np.cos(ν)]) / nμ
      # unit velocity: et = dRdt / nt
      ev = dRdt / nv
      # tangent normal: es = et x en
      es = (nμ / (nν * nv)) * dμdt * dRdν - (nν / (nμ * nv)) * dνdt * dRdμ
      # center of force: ea = d2Rdt2 / na
      ea = self.gradient1(ev, dt) / np.sqrt(np.sum(self.gradient1(ev, dt) ** 2, axis=0))
      # binormal vector: eb = et x ea
      eb = np.cross(ev.T, ea.T).T
      ################
      # TANGENT FRAME: [azimuthal, polar, surface normal]
      tangent[0, :, :] = eν
      tangent[1, :, :] = eμ
      tangent[2, :, :] = en
      ################
      # DARBOUX FRAME: [velocity, bi-normal, surface normal]
      darboux[0, :, :] = ev
      darboux[1, :, :] = es
      darboux[2, :, :] = en
      ######################
      # FRENET SERRET FRAME: [velocity, acceleration, bi-normal FS]
      frenet[0, :, :] = ev
      frenet[1, :, :] = ea
      frenet[2, :, :] = eb
      ###############################
      return tangent, darboux, frenet


  def parallel_transport(self, args, integrate=True):
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
      if integrate is True or args.moving_frame is "GEOMETRIC PHASE":  # integrate
          # time step
          dt = self.t[1] - self.t[0]
          # hamiltonians: differential basis
          tangent_matrix, γ, ξ = self.tangent2exponential_matrix(self.tangent, dt, args)
          darboux_matrix = self.basis2exponential_matrix(self.darboux, dt)
          frenet_matrix = self.basis2exponential_matrix(self.frenet, dt)
          geometric_matrix = self.rotor2d(γ)
          for idx in range(1, args.time_resolution):
              pt_tangent[idx, :] = np.matmul(tangent_matrix[:, :, idx], pt_tangent[idx - 1, :]).T
              pt_darboux[idx, :] = np.matmul(darboux_matrix[:, :, idx], pt_darboux[idx - 1, :]).T
              pt_frenet[idx, :] = np.matmul(frenet_matrix[:, :, idx], pt_frenet[idx - 1, :]).T
              pt_tan_plane_vec[idx, :] = np.matmul(geometric_matrix[:, :, idx], tangent_vec).T
      else:  # projection
          for idx in range(1, args.time_resolution):
              # fill pt vectors
              pt_tangent[idx, :] = np.matmul(np.matmul(self.tangent[:, :, idx],
                                                       self.tangent[:, :, idx - 1].T), pt_tangent[idx - 1, :])
              pt_darboux[idx, :] = np.matmul(np.matmul(self.darboux[:, :, idx],
                                                       self.darboux[:, :, idx - 1].T), pt_darboux[idx - 1, :])
              pt_frenet[idx, :] = np.matmul(np.matmul(self.frenet[:, :, idx],
                                                      self.frenet[:, :, idx - 1].T), pt_frenet[idx - 1, :])
              pt_tan_plane_vec[idx, :] = np.matmul(np.matmul(self.tangent[:2, :, idx],
                                                             self.tangent[:2, :, idx - 1].T),
                                                   pt_tan_plane_vec[idx - 1, :])
      return pt_tangent, pt_darboux, pt_frenet, pt_tan_plane_vec, γ, ξ


  @staticmethod
  def rotor2d(γ):
      rotation_matrix = np.zeros((2, 2, len(γ)))
      rotation_matrix[0, 0, :], rotation_matrix[0, 1, :] = np.cos(γ), -np.sin(γ)
      rotation_matrix[1, 0, :], rotation_matrix[1, 1, :] = np.sin(γ), np.cos(γ)
      return rotation_matrix


  def get_global_grid(self, args, n_pts=77, time_steps=2**12):
      t = np.linspace(0, 2*np.pi, time_steps)
      dt = t[1] - t[0]
      dim = args.global_grid_size
      # hamiltonian shorthand
      H = np.zeros((time_steps, 3))
      H[:, 0] = np.interp(t, self.t, self.H[:, 0])
      H[:, 1] = np.interp(t, self.t, self.H[:, 1])
      H[:, 2] = np.interp(t, self.t, self.H[:, 2])
      # quaternion shorthand
      a = np.interp(t, self.t, self.Q[:, 0])
      b = np.interp(t, self.t, self.Q[:, 1])
      c = np.interp(t, self.t, self.Q[:, 2])
      d = np.interp(t, self.t, self.Q[:, 3])
      # SO(3) rotor
      rotor = self.quaternion2rotor(np.array([a, b, c, d]).T)
      # GLOBAL PHASE GRID:
      θ0, φ0 = np.meshgrid(np.linspace(0, np.pi, dim[0]), np.linspace(0, 2 * np.pi, dim[1]))
      θ0, φ0 = θ0.flatten(), φ0.flatten()

      # bloch vector
      if args.projection=='i':
          Ri, Rj, Rk = np.cos(θ0), np.sin(θ0) * np.sin(φ0), np.sin(θ0) * np.cos(φ0)
      elif args.projection == 'j':
          Ri = -np.sin(θ0) * np.sin(self.ω0)
          Rj = np.cos(φ0) * np.cos(self.ω0) + np.sin(φ0) * np.cos(θ0) * np.sin(self.ω0)
          Rk = -np.sin(φ0) * np.cos(self.ω0) + np.cos(φ0) * np.cos(θ0) * np.sin(self.ω0)
      elif args.projection == 'k':
          Ri = -np.sin(θ0) * np.cos(self.ω0)
          Rj = -np.cos(φ0) * np.sin(self.ω0) + np.sin(φ0) * np.cos(θ0) * np.cos(self.ω0)
          Rk = np.sin(φ0) * np.sin(self.ω0) + np.cos(φ0) * np.cos(θ0) * np.cos(self.ω0)
      R0 = np.array([Ri, Rj, Rk]).T
      # evolve the state
      R1 = np.matmul(rotor, R0.T)  # evolve the bloch vector
      np.seterr(divide='ignore', invalid='ignore')  # suppress divide by 0 warning
      Hj, Hk, Rj, Rk = H[:, 1].reshape(len(H[:, 1]), 1), H[:, 2].reshape(len(H[:, 1]), 1), R1[:, 1, :], R1[:, 2, :]
      OMEGA = (Hj * Rj + Hk * Rk) / (Rj ** 2 + Rk ** 2)
      # global grid
      global_grid = np.sum(OMEGA * dt, axis=0).reshape(dim[0], dim[1])
      global_grid[:, 0], global_grid[:, -1] = np.nan, np.nan  # global_grid[0, :], global_grid[-1, :] = np.nan, np.nan
      global_grid = global_grid / np.pi
      # MOBIUS band
      Ω1 = np.interp(np.linspace(0, 2 * np.pi, n_pts), self.t, self.Ω)  # global phase
      t1 = np.linspace(0, 2 * np.pi, n_pts)  # time
      p1 = np.linspace(-1, 1, int(args.display_frames / 4))  # band
      Ω2, p2 = np.meshgrid(Ω1, p1)
      t2, _ = np.meshgrid(t1, p1)
      Ω2, p2, t2 = Ω2.ravel(), p2.ravel(), t2.ravel()
      # delaunay
      x = (1 + 0.25 * p2 * np.cos(Ω2 / 2)) * np.cos(t2)
      y = (1 + 0.25 * p2 * np.cos(Ω2 / 2)) * np.sin(t2)
      z = p2 * np.sin(Ω2 / 2)
      tri = Triangulation(np.ravel(t2), np.ravel(p2))
      # MOBIUS arrow
      arrow_start = np.array([(1 + 0.25 * p1[0] * np.cos(Ω1 / 2)) * np.cos(t1),
                              (1 + 0.25 * p1[0] * np.cos(Ω1 / 2)) * np.sin(t1),
                              p1[0] * np.sin(Ω1 / 2)])
      arrow_end = np.array([(1 + 0.25 * p1[-1] * np.cos(Ω1 / 2)) * np.cos(t1),
                            (1 + 0.25 * p1[-1] * np.cos(Ω1 / 2)) * np.sin(t1),
                            p1[-1] * np.sin(Ω1 / 2)])
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
      if args.moving_frame is 'random':
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
      elif U_label == 'x_axis':
          U = self.expmatrix(self.t/2, self.i)
      elif U_label == 'y_axis':
          U = self.expmatrix(self.t/2, self.j)
      elif U_label == 'z_axis':
          U = self.expmatrix(self.t/2, self.k)
      else:  # U_label == 'equation (30)':
          U = np.matmul(self.expmatrix(self.t / 2, self.i), self.expmatrix(self.t / 2, self.j))
      return U, U_label, cigar, pancake, args #np.swapaxes(U,1,2)


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
      hz = np.sum(self.gradient1(e1, dt) * e0, axis=0)
      hy = np.sum(self.gradient1(e0, dt) * e2, axis=0)
      hx = np.sum(self.gradient1(e2, dt) * e1, axis=0)
      # fill operator
      operator[0, 1, :], operator[1, 0, :] = -hz, hz
      operator[0, 2, :], operator[2, 0, :] = hy, -hy
      operator[1, 2, :], operator[2, 1, :] = -hx, hx
      return self.exponential_matrix(operator*dt)


  def tangent2exponential_matrix(self, basis, dt, args):
      operator = np.zeros(basis.shape)
      nμ = np.sqrt(self.r ** 2 + self.drdμ ** 2)
      # operator elements
      Aν = (self.r ** 2 + 2 * self.drdμ ** 2 - self.r * self.d2rdμ2) * self.dμdt / nμ ** 2
      Aμ = (self.drdμ * np.cos(self.μ) - self.r * np.sin(self.μ)) * self.dνdt / nμ
      An = (self.drdμ * np.sin(self.μ) + self.r * np.cos(self.μ)) * self.dνdt / nμ
      # fill operator
      operator[0, 1, :], operator[1, 0, :] = -An, An
      operator[0, 2, :], operator[2, 0, :] = Aμ, -Aμ
      operator[1, 2, :], operator[2, 1, :] = -Aν, Aν
      # bloch vector
      if args.projection=='i':
          Ri, Rj, Rk = self.Ψ[:, 0, 0], self.Ψ[:, 1, 0], self.Ψ[:, 2, 0]
      elif args.projection == 'j':
          Ri, Rj, Rk = self.Ψ[:, 0, 1], self.Ψ[:, 1, 1], self.Ψ[:, 2, 1]
      elif args.projection == 'k':
          Ri, Rj, Rk = self.Ψ[:, 0, 2], self.Ψ[:, 1, 2], self.Ψ[:, 2, 2]
      # hamiltonian
      Hi, Hj, Hk = self.H[:, 0], self.H[:, 1], self.H[:, 2]
      # dynamic phase
      # dEdt = nμ
      if args.heuristic_dynamic_phase is True:
          dξdt = ((self.r / nμ) * (Hi * Ri + Hj * Rj + Hk * Rk) + (Hi / nμ) * self.drdμ * np.sin(self.μ)
                  + (nμ - self.r - Ri * self.drdμ * np.sin(self.μ)) * (self.dΩdt / nμ))
      else:
          dξdt = Hi * Ri + Hj * Rj + Hk * Rk
      ξ = (np.cumsum(dξdt * self.dt, axis=0) - dξdt[0] * self.dt)
      # geometric phase of a tangent vector to a surface of revolution
      γ = (np.cumsum(An * dt, axis=0) - An[0] * dt)
      return self.exponential_matrix(operator*dt), γ, ξ



  def unitary2quaternion(self, cayley_basis='left'):
      if cayley_basis == 'left':
          q = self.U[:, :, 0]
      else:  # 'right'
          q = self.U[:, 0, :]
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
      elif np.random.rand(1) < 0.5:
          return True, False
      else:
          return False, True

  @staticmethod
  def cayley(cayley_basis='left'):
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

  @staticmethod
  def lie_algebra():
      I = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
      k = np.array([[0,-1, 0],
                    [1, 0, 0],
                    [0, 0, 0]])
      j = np.array([[0, 0, 1],
                    [0, 0, 0],
                    [-1, 0, 0]])
      i = np.array([[0, 0, 0],
                    [0, 0,-1],
                    [0, 1, 0]])
      return I, i, j, k
