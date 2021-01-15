import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib import gridspec
import random
import sys


def randomColormap(choice=False, colorcode=False, intense=False, pattern=False):
    options=[choice,colorcode,intense,pattern]
    if np.any(options) is not True:
        options[random.choice([0, 1, 2, 3])] = True
    if options[0]:      # choice
        x=['gist_earth', 'gist_yarg', 'gist_gray', 'ocean', 'cubehelix', 'binary',  'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'viridis', 'plasma', 'inferno', 'magma', 'Spectral', 'coolwarm', 'seismic']
    elif options[1]:    # colorcode
        x=['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', ]
    elif options[2]:    # intense
        x=['gist_ncar', 'gist_rainbow', 'gist_stern', 'nipy_spectral', 'hsv', 'bwr', 'jet', 'rainbow', 'brg', 'terrain', 'gnuplot', 'gnuplot2', 'CMRmap',]
    else: # options[3]: pattern
        x=['Pastel1', 'Pastel2', 'Dark2', 'Accent', 'Set1', 'Set2', 'Set3', 'flag', 'Paired', 'prism', 'tab10', 'tab20', 'tab20b', 'tab20c',]
    return random.choice(x)


class Plot:
  def __init__(self, obj, args):
    # generate the path
    self.t = np.linspace(0, 2 * np.pi, args.display_frames)
    self.θ0 = obj.θ0
    self.φ0 = obj.φ0
    self.U_label = obj.U_label
    self.global_grid = obj.global_grid
    self.Q = np.zeros((args.display_frames, 4))
    self.r = np.zeros((args.display_frames,))
    self.θ = np.zeros((args.display_frames,))
    self.φ = np.zeros((args.display_frames,))
    self.drdt = np.zeros((args.display_frames,))
    self.dθdt = np.zeros((args.display_frames,))
    self.dφdt = np.zeros((args.display_frames,))
    # R3 path & parallel transported vector
    self.R = np.zeros((args.display_frames, 3))
    self.pt_vector = np.zeros((args.display_frames, 3))
    # initialize frames
    self.tangent = np.zeros((3, 3, args.display_frames))
    self.darboux = np.zeros((3, 3, args.display_frames))
    self.frenet = np.zeros((3, 3, args.display_frames))
    # parallel transport
    self.pt_tangent = np.zeros((args.display_frames, 3))
    self.pt_darboux = np.zeros((args.display_frames, 3))
    self.pt_frenet = np.zeros((args.display_frames, 3))
    self.pt_tan_plane_vec = np.zeros((args.display_frames, 2))
    # surface
    self.surface = obj.surface
    self.mobius = obj.mobius
    # intrinsics
    self.γ = np.zeros((args.display_frames,))
    self.ξ = np.zeros((args.display_frames,))
    self.ω = np.zeros((args.display_frames,))
    # update
    self.fill_stream(obj, args)
  # fill vector stream
  def fill_stream(self, obj, args):
      # frame skip
      skp = int(args.time_resolution / args.display_frames)
      # copy moving frames
      self.tangent = obj.tangent[:, :, ::skp]
      self.darboux = obj.darboux[:, :, ::skp]
      self.frenet = obj.frenet[:, :, ::skp]
      # copy pt vector coefficients
      self.pt_tangent = obj.pt_tangent[::skp, :]
      self.pt_darboux = obj.pt_darboux[::skp, :]
      self.pt_frenet = obj.pt_frenet[::skp, :]
      self.pt_tan_plane_vec = obj.pt_tan_plane_vec[::skp, :]
      # copy remainder
      self.t = obj.t[::skp]
      self.r = obj.r[::skp]
      self.θ = obj.θ[::skp]
      self.φ = obj.φ[::skp]
      self.drdt = obj.drdt[::skp]
      self.dθdt = obj.dθdt[::skp]
      self.dφdt = obj.dφdt[::skp]
      # path
      self.R = obj.R[:, ::skp]
      # intrinsics
      self.γ = obj.γ[::skp]
      self.ξ = obj.ξ[::skp]
      self.ω = obj.ω[::skp]


class Figure:
  def __init__(self, obj, args):
    self.obj = obj
    # initialize figure
    self.fig, self.aXes, self.text_string, self.fiGlobal = self.initialize_figure(args)


  # initialize the figure
  def initialize_figure(self, args, elev_init=12, string0='', string1=''):
      obj = self.obj  # shorthand
      path = obj.R[::-1, :].T
      plt.ion()
      # declare figure
      fig = plt.figure('unitary: ' + self.obj.U_label, figsize=(16.8, 9.2)) # , figsize=(5, 3)
      # SUBPLOT (3,3,1): global phase
      ax1 = fig.add_subplot(333)
      ax1.set_title('Global phase (units: \u03C0)')
      dim=args.global_grid_size
      θ, φ = np.meshgrid(np.linspace(0, 1, dim[0]), np.linspace(0, 2, dim[1]))
      plt.pcolormesh(φ, θ, np.round(obj.global_grid).astype('int16'))
      ax1.set_aspect('equal')
      plt.colorbar(orientation='vertical') # ,shrink=0.5
      ax1.set_ylabel("θ")
      ax1.set_xlabel("φ")
      ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
      ax1.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0])
      ax1.set_yticklabels(["0", " ", "0.5", " ", "1.0"])
      ax1.set_xticklabels(["0", " ", "0.5", " ", "1", " ", "1.5", " ", "2.0"])
      ax1.plot(self.obj.φ0 / np.pi, self.obj.θ0 / np.pi, 'red', marker="o", markersize=3)
      # SUBPLOT (3,3,6):  S1 fibre bundle
      ax2 = fig.add_subplot(336)
      ax2.set_title('S1 fibre bundle') # : global = geometric + dynmaic
      ax2.set_ylabel("\u03C0 radians")
      ax2.set_xlabel("t")
      ax2.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0])
      ax2.set_xticklabels(["0", " ", "0.5", " ", "1", " ", "1.5", " ", "2.0"])
      # global
      ax2.plot(obj.t / np.pi, obj.ω / np.pi, 'green', linewidth=4, label='global')
      f2a = ax2.plot(obj.t[0] / np.pi, obj.ω[0] / np.pi, 'green', marker="o", markersize=10)
      ax2.plot(obj.t / np.pi, obj.ξ / np.pi + obj.γ / np.pi, 'black', linestyle='dotted', linewidth=1)
      # dynamic
      ax2.plot(obj.t / np.pi, obj.ξ / np.pi, 'orange', linewidth=2, label='dynamic')
      f2c = ax2.plot(obj.t[0] / np.pi, obj.ξ[0] / np.pi, 'orange', marker="o", markersize=7)
      # geometric
      ax2.plot(obj.t / np.pi, obj.γ / np.pi, 'gold', linewidth=2, label='geometric')
      f2b = ax2.plot(obj.t[0] / np.pi, obj.γ[0] / np.pi, 'gold', marker="o", markersize=7)
      ax2.legend()
      # SUBPLOT (3,3,9): mobius band
      ax3 = fig.add_subplot(339, projection='3d')
      ax3.plot_trisurf(self.obj.mobius[0], self.obj.mobius[1], self.obj.mobius[2],
                       triangles=obj.mobius[3].triangles, linewidth=0.2, cmap='copper', antialiased=True)
      ax3.set_ylabel("y")
      ax3.set_xlabel("x")
      ax3.set_zlabel("z")
      arrow = obj.mobius[4]
      f3a = ax3.plot(arrow[0, :, 0], arrow[1, :, 0], arrow[2, :, 0], 'black', alpha=0.95, linewidth=3)
      f3b = ax3.plot([], [], [], 'orange', marker="o", markersize=5)
      f3c = ax3.plot([], [], [], 'orange', marker="o", markersize=3)
      ax3.plot(arrow[0, :, 0], arrow[1, :, 0], arrow[2, :, 0], 'black', alpha=0.95, linewidth=1.5)
      f3c[0].set_data(arrow[0, 0, 0], arrow[1, 0, 0])
      f3c[0].set_3d_properties(arrow[2, 0, 0])
      ax3.view_init(elev=40, azim=self.obj.t[0] * 180 / np.pi + 10)
      # SUBPLOT (1,3,(1,2)): Moving frames
      ax = fig.add_subplot(1,3,(1,2), projection='3d')
      # surface
      ax.plot_trisurf(obj.surface.x, obj.surface.y, obj.surface.z,
                      triangles=obj.surface.tri.triangles,
                      cmap=args.colormap, linewidths=0.2, alpha=0.7)
      # path
      alpha_grads = self.frontBack(path, args)
      s2 = [ax.plot(path[i - 1:i + 1, 0], path[i - 1:i + 1, 1], path[i - 1:i + 1, 2],
                    'white', alpha=alpha_grads[i], linewidth=2)
            for i in range(1, path.shape[0])]
      # point
      p1, = ax.plot(path[0:1, 0], path[0:1, 1], path[0:1, 2],
                    linestyle="", marker="o",
                    markersize=6)  # point
      # moving frame
      q0, = ax.plot([], [], [], 'black', linewidth=4)  # frame axis 0
      q1, = ax.plot([], [], [], 'black', linewidth=4)  # frame axis 1
      q2, = ax.plot([], [], [], 'black', linewidth=4)  # frame axis 2
      pt, = ax.plot([], [], [], 'black', linewidth=4)  # pt vector
      #############
      # camera view
      ax.view_init(elev=elev_init, azim=obj.φ[0] * 180 / np.pi)
      # axis ticks
      ax.set_xticklabels(["-1", " ", "-0.5", " ", "0", " ", "0.5", " ", "1.0"])
      ax.set_yticklabels(["-1", " ", "-0.5", " ", "0", " ", "0.5", " ", "1.0"])
      ax.set_zticklabels(["-1", " ", "-0.5", " ", "0", " ", "0.5", " ", "1.0"])
      # axis labels
      ax.set_ylabel("y")
      ax.set_xlabel("z")
      ax.set_zlabel("x")
      # figure title
      ax.set_title('Moving frames and parallel transport')
      # update text box info
      if args.moving_frame == 'GEOMETRIC PHASE':
          args.show_pt_vector = False
          str0 = 'GEOMETRIC PHASE \nframe: tangent plane    \n cmap: ' + args.colormap + ' '
      else:
          str0 = 'frame: ' + args.moving_frame + '    \n cmap: ' + args.colormap + ' '
      if args.show_pt_vector:
          str1 = '\nparallel transported vector:' + ' \n'
          str2 = ' (x, y, z) = '
      else:
          str1, str2 = '', ''
      # surface data
      str3 = '\nsurface: ' + obj.surface.shape
      # initial state data
      str4 = '\nunitary: ' + obj.U_label
      str5 = '\ninitial state: \n (θ0, φ0) = (%.1f, %.1f)' % ((obj.θ0 * 180 / np.pi), (obj.φ0 * 180 / np.pi))
      text_string = [str0, str1, str2, str3, str4, str5]
      fig_info = ax.text2D(0.75, 0.15, text_string,
                           bbox={'facecolor': 'w', 'alpha': 0.75, 'pad': 5},
                           transform=ax.transAxes, ha="left")
      ######################
      print(' ')
      print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
      print('$$$ MOVING FRAMES and parallel transport $$$')
      print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
      print(' ')
      return fig, [p1, s2, q0, q1, q2, pt, fig_info, ax], text_string, [f2a, f2b, f2c, f3a, f3b, ax3]


  # update the moving frame, path, and viewpoint
  def update_figure(self, frame_num, args,
                    elev_init=12, view_rotate=35):
      self.update_global_figure(frame_num, args)
      obj = self.obj  # shorthand
      path = obj.R[::-1, :].T
      # color is taken from point xyz coorinates
      col = path[frame_num, :] * 0.5 + 0.5
      # view point
      azim_init = obj.φ[0] * 180 / np.pi
      # update flages on first index
      if args.moving_frame == 'TANGENT':
          cVector, mFrame = obj.pt_tangent[frame_num, :], obj.tangent[:, ::-1, frame_num]
          axis_colours = [np.array([0.0, 0.5, 0.9]), np.array([0.0, 0.5, 0.9]), col]
      elif args.moving_frame == 'DARBOUX':
          cVector, mFrame = obj.pt_darboux[frame_num, :], obj.darboux[:, ::-1, frame_num]
          axis_colours = [np.array([0.0, 0.5, 0.9]), np.array([0.0, 0.5, 0.9]), col]
      elif args.moving_frame == 'FRENET SERRET':
          cVector, mFrame = obj.pt_frenet[frame_num, :], obj.frenet[:, ::-1, frame_num]
          axis_colours = [col, np.array([0.0, 0.5, 0.9]), np.array([0.0, 0.5, 0.9])]
      else: # 'GEOMETRIC PHASE'
          cVector, mFrame = obj.pt_tan_plane_vec[frame_num, :], obj.tangent[:, ::-1, frame_num]
          mFrame[2, :] = cVector[0] * mFrame[0, :] + cVector[1] * mFrame[1, :]
          axis_colours = [np.array([0.0, 0.5, 0.9]), np.array([0.0, 0.5, 0.9]), col]
          args.show_pt_vector = False
          cVector = np.array([cVector[0], cVector[1], 0.0])
      # current view point: rotate current vector by args.view_rotate degrees
      if args.tracking is True:
          view_point = np.matmul(self.rodrigues([0, 0, 1], view_rotate * np.pi / 180), path[frame_num, :].reshape(3, 1))
      else:
          th = elev_init * np.pi / 180
          ph = azim_init * np.pi / 180
          view_point = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)])  # matplotlib convention
          view_point = np.matmul(self.rodrigues([0, 0, 1], view_rotate * np.pi / 180), view_point.reshape(3, 1))
      # alpha gradients for image foreground & background
      alpha_grads = self.frontBack(path, args, current_vector=view_point)
      # update 2-sphere path
      idx = 0
      for pt in self.aXes[1]:
          idx = idx + 1
          pt[0].set_data(path[idx - 1:idx + 1, 0], path[idx - 1:idx + 1, 1])
          pt[0].set_3d_properties(path[idx - 1:idx + 1, 2])
          pt[0].set_alpha(alpha_grads[idx])
          if idx <= frame_num:
              pt[0].set_color('gray')
          else:
              pt[0].set_color('white')
      # update point
      self.aXes[0].set_data(path[frame_num, 0:2])
      self.aXes[0].set_3d_properties(path[frame_num, 2])
      self.aXes[0].set_color(col)
      if args.tracking is not True:
          if np.sum(np.ravel(view_point) * path[frame_num, :]) < 0:
              self.aXes[0].set_alpha(0.35)
      # moving frame
      aXs = self.plotFrame(cVector, mFrame, path[frame_num, :],
                           length=0.2 * np.max(np.sqrt(np.sum(path ** 2,axis=1))))
      for idx in range(3):
          self.aXes[idx + 2].set_data(aXs[idx][:, 0], aXs[idx][:, 1])
          self.aXes[idx + 2].set_3d_properties(aXs[idx][:, 2])
          self.aXes[idx + 2].set_color(axis_colours[idx])
          self.aXes[idx + 2].set_alpha(0.9)
          if args.tracking is not True:
              if np.sum(np.ravel(view_point) * path[frame_num, :]) < 0:
                  self.aXes[idx + 2].set_alpha(0.35)
      # parallel transport 3-vector
      if args.show_pt_vector:
          # update text box info
          pt_vec = aXs[3][-1, :] - path[frame_num, :]
          pt_vec = pt_vec / np.sqrt(np.sum(pt_vec ** 2))
          self.text_string[2] = (' (x, y, z) = ( %.2f, ' % pt_vec[0] +
                                 '%.2f, ' % pt_vec[1] +
                                 '%.2f)' % pt_vec[2])
          # continue
          self.aXes[5].set_data(aXs[-1][:, 0], aXs[-1][:, 1])
          self.aXes[5].set_3d_properties(aXs[-1][:, 2])
          # self.aXes[5].set_color(col)
          if np.sqrt(np.sum(aXs[-1][-1, :] ** 2)) >= 1:
              self.aXes[5].set_alpha(0.85)
          else:
              # self.aXes[idx + 2].set_alpha(0.85 * (np.sum(aXs[idx][-1, :] ** 2)))
              self.aXes[5].set_alpha(0.85)
          if args.tracking is not True:
              if np.sum(np.ravel(view_point) * path[frame_num, :]) < 0:
                  self.aXes[5].set_alpha(0.35)
          self.aXes[5].set_color('black')
      # update text box info
      self.aXes[6].set_text(self.text_string[0] + self.text_string[1] +
                            self.text_string[2] + self.text_string[3] +
                            self.text_string[4] + self.text_string[5])
      # update view
      if args.tracking is True:
          azim = np.arctan2(path[frame_num, 1], path[frame_num, 0]) * 180 / np.pi
      else:
          azim = azim_init
      self.aXes[7].view_init(elev=elev_init, azim=azim + view_rotate)
      plt.pause(1e-3)
      self.drawProgressBar(frame_num / args.display_frames)
      if args.save_frames_as_png:
          plt.savefig('image' + repr(frame_num + 100) + '.png')


  # Update subplots (3,3,6) and (3,3,9): S1 fibre bundle and mobius band
  def update_global_figure(self, frame_num, args):
      # S1 fibre bundle plot
      self.fiGlobal[0][0].set_data(self.obj.t[frame_num] / np.pi, self.obj.ω[frame_num] / np.pi)
      self.fiGlobal[1][0].set_data(self.obj.t[frame_num] / np.pi, self.obj.γ[frame_num] / np.pi)
      self.fiGlobal[2][0].set_data(self.obj.t[frame_num] / np.pi, self.obj.ξ[frame_num] / np.pi)
      # mobius band plot
      arrow = self.obj.mobius[4]
      self.fiGlobal[3][0].set_data(arrow[0, :, frame_num], arrow[1, :, frame_num])
      self.fiGlobal[3][0].set_3d_properties(arrow[2, :, frame_num])
      self.fiGlobal[4][0].set_data(arrow[0, 0, frame_num], arrow[1, 0, frame_num])
      self.fiGlobal[4][0].set_3d_properties(arrow[2, 0, frame_num])
      # t = 2 * np.pi * (frame_num / args.display_frames)
      self.fiGlobal[5].view_init(elev=40, azim=(frame_num / args.display_frames) * 360 + 15)


  # returns alpha gradients of an Nx3 array of coordinates
  # alpha is [small,large] for [background,foreground]
  @staticmethod
  def frontBack(xyz, args, current_vector=None, α_min=0.25, α_max=0.95, elev_init=12, azim_init=25):
      # point and line
      if current_vector is None:
          θ = elev_init * np.pi / 180
          φ = azim_init * np.pi / 180
      else:
          θ = elev_init * np.pi / 180
          φ = np.arctan2(current_vector[1], current_vector[0])
      # reference vector
      ref_vector = np.array([np.cos(θ) * np.cos(φ), np.cos(θ) * np.sin(φ), np.sin(θ)], dtype=object)
      # indices at the front and back of the image
      front = (np.sum(ref_vector * xyz, axis=1) >= 0)
      back = (np.sum(ref_vector * xyz, axis=1) < 0)
      # alpha gradients
      α_gradients = np.ones((args.display_frames,)) * (α_max * front + α_min * back)
      return α_gradients

  # plot coodinates for the moving frame
  @staticmethod
  def plotFrame(currentV, currentF, xyz0, nPoints=2, length=0.2):
      e0 = np.linspace(0, length * currentF[0, :], nPoints) + xyz0
      e1 = np.linspace(0, length * currentF[1, :], nPoints) + xyz0
      e2 = np.linspace(0, length * currentF[2, :], nPoints) + xyz0
      # parallel transported 3-vector
      pt_local = (currentV[0] * currentF[0, :]
                  + currentV[1] * currentF[1, :]
                  + currentV[2] * currentF[2, :])
      pt = np.linspace(0, length * pt_local, nPoints) + xyz0
      return e0, e1, e2, pt

  @staticmethod
  def rodrigues(aXs, angle): #(aXs[3], angle in radians)
      # Lie Algebra Matrices
      LieZ = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
      LieY = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
      LieX = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
      # Axis vector as a matrix
      Kvector = aXs[0] * LieX + aXs[1] * LieY + aXs[2] * LieZ
      # return rodrigues rotation matrix
      return np.identity(3) + Kvector * np.sin(angle) + np.matmul(Kvector, Kvector) * (1 - np.cos(angle))


  @staticmethod
  # progress bar for data loading
  def drawProgressBar(percent, barLen=40):
      sys.stdout.write("\r")
      progress = ""
      for i in range(barLen):
          if i < int(barLen * percent):
              progress += "="
          else:
              progress += " "
      sys.stdout.write("[ %s ] %.2f%% " % (progress, percent * 100))
      sys.stdout.flush()