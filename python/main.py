# MOVING FRAMES AND PARALLEL TRANSPORT #################################################################################
""""
This code animates:
 - the tangent frame
 - the darboux frame, and
 - the frenet serret frame
for the quaternions cited in reference 1).
Also animated are:
 - the parallel transport of a 3-vector, and
 - the geometric phase of a 2-vector.
The geometric phase is generalized to
non-spherical surfaces. Surface options are:
 - the 2-sphere, and
 - bushings surfaces of revolution
Reference:
1) arXiv:1601.02569
The Hopf Fibration and Hidden Variables
in Quantum and Classical Mechanics
 - Brian O'Sullivan
"""
# MEDITATIONS ON GEOMETRY ##############################################################################################
import numpy as np
import cv2
import argparse
from spinor_class import *
from matplotlib_movingframes import *

# ARGUMENTS ############################################################################################################
parser = argparse.ArgumentParser(description='Moving frames and parallel transport')
# surface options & intial state
parser.add_argument('--sphere', default=True)
parser.add_argument('--spinor_initial_state_deg', default=None,
                    choices=[None, (26.1,139.9,0.0)])
parser.add_argument('--projection', default='i', choices=['random', 'i', 'j', 'k'])
# general arguments
parser.add_argument('--moving_frame', default='random',
                    choices=['random', 'TANGENT', 'DARBOUX', 'FRENET SERRET', 'GEOMETRIC PHASE'])
parser.add_argument('--select_unitary', default='random',
                    choices=['random', 'equation (27)', 'equation (28)', 'equation (29)', 'equation (30)',
                             'x_axis', 'y_axis', 'z_axis'])
parser.add_argument('--heuristic_dynamic_phase', default=True) # apply equation (C.13)
parser.add_argument('--display_frames', default=2**7)
parser.add_argument('--global_grid_size', default=(2**7, 2**7))
# display options
parser.add_argument('--tracking', default=True)
parser.add_argument('--show_pt_vector', default=False)
# bushings surface
parser.add_argument('--eta', default=5.0)
parser.add_argument('--Lambda', default=11)
parser.add_argument('--nF', default=23)
# colormap
parser.add_argument('--colormap', default=randomColormap(),
                    choices=[randomColormap(choice=False, colorcode=False, intense=False, pattern=False)])
# integration accuracy [minimum = 2 ** 12]
parser.add_argument('--time_resolution', default=2**16)
# save figures
parser.add_argument('--save_frames_as_png', default=False)
args = parser.parse_args()

# MAIN #################################################################################################################

# 1) surface - unitary - quaternion - rotor - path - moving frames - parallel transport
spinor = Spinor(args)

# 2) plotting
fig = Figure(Plot(spinor, args)  , args)

# update figure
for idx in range(args.display_frames):
    fig.update_figure(idx, args)
plt.show()