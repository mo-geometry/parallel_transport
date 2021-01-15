import numpy as np
from matplotlib.tri import Triangulation
from scipy.special import gammainc, factorial

# OBJECTS ##############################################################################################################

class Bushings:
  def __init__(self, args, vertices=44, cigar=False, pancake=False):
    # initialize
    if args.sphere is False: cigar = True
    # meshgrids θ + φ
    θ, φ = np.meshgrid(np.linspace(0, np.pi, vertices).astype('float64'),
                                 np.linspace(0, 2 * np.pi, vertices).astype('float64'))
    # surface meshgrids
    r, _, _, self.shape = self.surface(θ,
                                       η=args.eta, λ=args.Lambda, nF=args.nF,
                                       cigar=cigar, pancake=pancake)
    self.x, self.y, self.z, self.tri = delunay_triangulation(r, θ, φ)

  def surface(self, θ, cigar=False, pancake=False, η=5.0, λ=11, nF=23, derivatives=None):
      if pancake is True:
          α, β, dαdθ, dβdθ, d2αdθ2, d2βdθ2 = self.bushings_variables(θ, η, shape='pancake')
          shape = 'bushings pancake' + '\n (η, λ, nF) = ( %1.1f, %2d, %2d).' % (η, λ, nF)
      elif cigar is True:
          α, β, dαdθ, dβdθ, d2αdθ2, d2βdθ2 = self.bushings_variables(θ, η, shape='cigar')
          shape = 'bushings cigar' + '\n (η, λ, nF) = ( %1.1f, %2d, %2d).' % (η, λ, nF)
      else:
          r, drdθ, d2rdθ2 = np.ones(θ.shape), np.zeros(θ.shape), np.zeros(θ.shape)
          shape = 'S2 sphere'
          return r, drdθ, d2rdθ2, shape
      # summation term, first and second derivatives
      G, dG, d2G = self.summation_terms(α, β, dαdθ, dβdθ, d2αdθ2, d2βdθ2, λ, nF)
      # bushings function
      r = gammainc(nF + 1, β) + np.exp(-β) * G
      # first and second derivatives
      if derivatives is not None:
          nFac = factorial(nF)
          drdθ = np.exp(-β) * (dG - dβdθ * G + (β ** nF * dβdθ) / nFac)
          d2rdθ2 = (np.exp(-β) * (d2G - d2βdθ2 * G - dβdθ * dG + β ** (nF - 1) * (nF * dβdθ ** 2 + β * d2βdθ2) / nFac)
                    - dβdθ * drdθ)
      else:
          drdθ = np.zeros(θ.shape)
          d2rdθ2 = np.zeros(θ.shape)
      # register and return
      return r, drdθ, d2rdθ2, shape

  @staticmethod
  def bushings_variables(θ, η, shape=None):
      if shape=='cigar':
          α = (η * np.sin(θ)) ** 2
          β = (η * np.cos(θ)) ** 2
          dαdθ = 2 * η ** 2 * np.cos(θ) * np.sin(θ)
          dβdθ = - 2 * η ** 2 * np.cos(θ) * np.sin(θ)
          d2αdθ2 = 2 * η ** 2 * np.cos(2 * θ)
          d2βdθ2 = - 2 * η ** 2 * np.cos(2 * θ)
          return α, β, dαdθ, dβdθ, d2αdθ2, d2βdθ2
      elif shape=='pancake':
          α = (η * np.cos(θ)) ** 2
          β = (η * np.sin(θ)) ** 2
          dαdθ = - 2 * η ** 2 * np.cos(θ) * np.sin(θ)
          dβdθ = 2 * η ** 2 * np.cos(θ) * np.sin(θ)
          d2αdθ2 = - 2 * η ** 2 * np.cos(2 * θ)
          d2βdθ2 = 2 * η ** 2 * np.cos(2 * θ)
          return α, β, dαdθ, dβdθ, d2αdθ2, d2βdθ2

  @staticmethod
  def summation_terms(α, β, dα, dβ, d2α, d2β, λ, nF, G=0.0, dG=0.0, d2G=0.0):
      np.seterr(divide='ignore', invalid='ignore')  # suppress divide by 0 warning
      for k in range(nF + 1):
          idk = np.floor((nF - k) / λ)
          fack = factorial(k)
          fackidk = (fack * factorial(idk))
          # shorthand
          t1 = β ** k * gammainc(idk + 1, α / λ) / fack
          t2 = β ** k * (α / λ) ** idk * (dα / λ) * np.exp(-α / λ) / fackidk
          t3 = k * (dβ / β) * t1
          # derivatives
          dt1 = t2 + t3
          dt2 = (k * (dβ / β) + idk * (dα / α) - (dα / λ) + (d2α / dα)) * t2
          dt3 = k * (d2β / β) * t1 - k * (dβ / β) ** 2 * t1 + k * (dβ / β) * dt1
          G = G + t1
          dG = dG + dt1
          d2G = d2G + dt2 + dt3
      return G, dG, d2G

# delunay triangulation
def delunay_triangulation(r, θ, φ):
    # surface
    x = np.ravel(r * np.sin(θ) * np.cos(φ))
    y = np.ravel(r * np.sin(θ) * np.sin(φ))
    z = np.ravel(r * np.cos(θ))
    # delunay triangulation
    return x, y, z, Triangulation(np.ravel(θ), np.ravel(φ))
