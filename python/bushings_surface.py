import numpy as np
from matplotlib.tri import Triangulation
from scipy.special import gammainc, factorial

# OBJECTS ##############################################################################################################

class Bushings:
  def __init__(self, args, vertices=44, cigar=False, pancake=False):
    # initialize
    if args.sphere is False: cigar = True
    # meshgrids μ + ν
    μ, ν = np.meshgrid(np.linspace(0, np.pi, vertices).astype('float64'),
                                 np.linspace(0, 2 * np.pi, vertices).astype('float64'))
    # surface meshgrids
    r, _, _, self.surface_shape = self.surface(μ, η=args.eta, λ=args.Lambda, nF=args.nF,
                                               cigar=cigar, pancake=pancake)
    self.x, self.y, self.z, self.tri = delunay_triangulation(r, μ, ν)

  def surface(self, μ, cigar=False, pancake=False, η=5.0, λ=11, nF=23, derivatives=None):
      if pancake is True:
          α, β, dαdμ, dβdμ, d2αdμ2, d2βdμ2 = self.bushings_variables(μ, η, shape='pancake')
          surface_shape = 'bushings pancake' + '\n (η, λ, nF) = ( %1.1f, %2d, %2d).' % (η, λ, nF)
      elif cigar is True:
          α, β, dαdμ, dβdμ, d2αdμ2, d2βdμ2 = self.bushings_variables(μ, η, shape='cigar')
          surface_shape = 'bushings cigar' + '\n (η, λ, nF) = ( %1.1f, %2d, %2d).' % (η, λ, nF)
      else:
          r, drdμ, d2rdμ2 = np.ones(μ.shape), np.zeros(μ.shape), np.zeros(μ.shape)
          surface_shape = 'S² sphere'
          return r, drdμ, d2rdμ2, surface_shape
      # summation term, first and second derivatives
      G, dG, d2G = self.summation_terms(α, β, dαdμ, dβdμ, d2αdμ2, d2βdμ2, λ, nF)
      # bushings function
      r = gammainc(nF + 1, β) + np.exp(-β) * G
      # first and second derivatives
      if derivatives is not None:
          nFac = factorial(nF)
          drdμ = np.exp(-β) * (dG - dβdμ * G + (β ** nF * dβdμ) / nFac)
          d2rdμ2 = (np.exp(-β) * (d2G - d2βdμ2 * G - dβdμ * dG + β ** (nF - 1) * (nF * dβdμ ** 2 + β * d2βdμ2) / nFac)
                    - dβdμ * drdμ)
      else:
          drdμ = np.zeros(μ.shape)
          d2rdμ2 = np.zeros(μ.shape)
      # register and return
      return r, drdμ, d2rdμ2, surface_shape

  @staticmethod
  def bushings_variables(μ, η, shape=None):
      if shape=='cigar':
          α = (η * np.sin(μ)) ** 2
          β = (η * np.cos(μ)) ** 2
          dαdμ = 2 * η ** 2 * np.cos(μ) * np.sin(μ)
          dβdμ = - 2 * η ** 2 * np.cos(μ) * np.sin(μ)
          d2αdμ2 = 2 * η ** 2 * np.cos(2 * μ)
          d2βdμ2 = - 2 * η ** 2 * np.cos(2 * μ)
          return α, β, dαdμ, dβdμ, d2αdμ2, d2βdμ2
      elif shape=='pancake':
          α = (η * np.cos(μ)) ** 2
          β = (η * np.sin(μ)) ** 2
          dαdμ = - 2 * η ** 2 * np.cos(μ) * np.sin(μ)
          dβdμ = 2 * η ** 2 * np.cos(μ) * np.sin(μ)
          d2αdμ2 = - 2 * η ** 2 * np.cos(2 * μ)
          d2βdμ2 = 2 * η ** 2 * np.cos(2 * μ)
          return α, β, dαdμ, dβdμ, d2αdμ2, d2βdμ2

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
def delunay_triangulation(r, μ, ν):
    # surface
    x = np.ravel(r * np.sin(μ) * np.cos(ν))
    y = np.ravel(r * np.sin(μ) * np.sin(ν))
    z = np.ravel(r * np.cos(μ))
    # delunay triangulation
    return x, y, z, Triangulation(np.ravel(μ), np.ravel(ν))



