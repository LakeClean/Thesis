#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
#import lightkurve
import os.path
#from astropy.stats import LombScargle
from astropy.timeseries import LombScargle
from bottleneck import nanmedian, nanmean, nanmax, nanmin
from scipy.optimize import minimize_scalar
from scipy.integrate import simps

class powerspectrum(object):
	"""
	Attributes:
		nyquist (float): Nyquist frequency in Hz.
		df (float): Fundamental frequency spacing in Hz.
		standard (tuple): Frequency in microHz and power density spectrum sampled
			from 0 to ``nyquist`` with a spacing of ``df``.
		ls (``astropy.stats.LombScargle`` object):
	"""

	def __init__(self, time, flux, flux_err=None, fit_mean=False, weighted=False):
		"""
		Parameters:
			lightcurve (``lightkurve.LightCurve`` object): Lightcurve to estimate power spectrum for.
			fit_mean (boolean, optional):
		"""

		# Store the input settings:
		self.fit_mean = fit_mean
		self.weighted = weighted

		self.time = time
		self.flux = flux

		# Calculate standard properties of the timeseries:
		indx = np.isfinite(self.flux)
		self.df = 1/(86400*(nanmax(self.time[indx]) - nanmin(self.time[indx]))) # Hz
		self.nyquist = 1/(2*86400*nanmedian(np.diff(self.time[indx]))) # Hz 86400s = 1dayaut
		self.standard = None

		if self.weighted:
			self.flux_err = flux_err
			self.ls = LombScargle(self.time[indx]*86400, self.flux[indx], dy=self.flux_err[indx], center_data=True, fit_mean=self.fit_mean) # , normalization='psd'
		else:
			self.ls = LombScargle(self.time[indx]*86400, self.flux[indx], center_data=True, fit_mean=self.fit_mean) # , normalization='psd'

		# Calculate a better estimate of the fundamental frequency spacing:
		self.df = self.fundamental_spacing_integral()

		# Calculate standard power density spectrum:
		# Start by calculating a complete un-scaled power spectrum:
		self.standard = self.powerspectrum(oversampling=1, nyquist_factor=1, scale=None)

		# Use the un-scaled power spectrum to finding the normalisation factor
		# which will ensure that Parseval's theorem holds:
		N = len(self.ls.t)


		if self.weighted:
			tot_MS = np.sum(((self.ls.y - nanmean(self.ls.y))/self.ls.dy)**2)/np.sum((1/self.ls.dy)**2)
		else:
			tot_MS = np.sum((self.ls.y - nanmean(self.ls.y))**2)/N

		tot_lomb = np.sum(self.standard[1])
		self.normfactor = tot_MS/tot_lomb


		# Re-scale the standard power spectrum to being in power density:
		self.standard = list(self.standard)
		self.standard[1] *= self.normfactor/(self.df*1e6)
		#self.standard.append(self.normfactor/(self.df*1e6))
		self.standard = tuple(self.standard)

	#------------------------------------------------------------------------------
	def fundamental_spacing_minimum(self):
		"""Estimate fundamental spacing using the first minimum spectral window function."""

		# Create "window" time series:
		freq_cen = 0.5*self.nyquist
		x = 0.5*np.sin(2*np.pi*freq_cen*self.ls.t) + 0.5*np.cos(2*np.pi*freq_cen*self.ls.t)

		# Calculate power spectrum for the given frequency range:
		ls = LombScargle(self.ls.t, x, center_data=True, fit_mean=self.fit_mean)

		# Minimize the window function around the first minimum:
		# Normalization is completely irrelevant
		window = lambda freq: ls.power(freq_cen+freq, normalization='psd', method='fast')
		res = minimize_scalar(window, [0.75*self.df, self.df, 1.25*self.df])
		df = res.x
		return df

	#------------------------------------------------------------------------------
	def fundamental_spacing_integral(self):
		"""Estimate fundamental spacing using the integral of the spectral window function."""
		# Integrate the windowfunction
		freq, window = self.windowfunction(width=100*self.df, oversampling=5)		
		#freq, window = self.windowfunction(width=10*self.df, oversampling=5)
		df = simps(window, freq)
		return df*1e-6

	#------------------------------------------------------------------------------
	def powerspectrum(self, freq=None, oversampling=1, nyquist_factor=1, scale='power'):
		"""
		Calculate power spectrum for time series.
		Parameters:
			freq (ndarray, optional): Frequencies to calculate power spectrum for. If set
				to None, the full frequency range from 0 to ``nyquist``*``nyquist_factor`` is calculated.
			oversampling (float, optional): Oversampling factor. Default=1.
			nyquist_factor (float, optional): Nyquist factor. Default=1.
			scale (string, optional): 'power', 'powerdensity' and 'amplitude'. Default='power'.
		Returns:
			tuple: Tuple of two ndarray with frequencies in microHz and corresponding
				power in units depending on the ``scale`` keyword.
		"""

		# The frequency axis in Hertz:
		assume_regular_frequency = False
		if freq is None:
			# If what we are really asking for is the standard power density spectrum, we have already
			# calculated it in the init-function, so just return that:
			if scale == 'powerdensity' and oversampling == 1 and nyquist_factor == 1 and self.standard:
				return self.standard
			# Set the standard frequency axis:
			freq = np.arange(self.df/oversampling, nyquist_factor*self.nyquist, self.df/oversampling, dtype='float64')
			assume_regular_frequency = True

		# Calculate power at frequencies using fast Lomb-Scargle periodiogram:
		power = self.ls.power(freq, normalization='psd', method='fast', assume_regular_frequency=assume_regular_frequency)
#		power = self.ls.power(freq, normalization='psd', method='scipy', assume_regular_frequency=assume_regular_frequency)

		# Due to numerical errors, the "fast implementation" can return power < 0.
		power = np.clip(power, 0, None)

		# Different scales:
		freq *= 1e6 # Rescale frequencies to being in microHz
		if scale is None:
			pass
		elif scale == 'power':
			power *= self.normfactor * 2
		elif scale == 'powerdensity':
			power *= self.normfactor/(self.df*1e6)
		elif scale == 'amplitude':
			power = np.sqrt(power*self.normfactor*2)

		return freq, power

	#------------------------------------------------------------------------------
	def windowfunction(self, width=None, oversampling=10):
		"""Spectral window function.
		Parameters:
			width (float, optional): The width in Hz on either side of zero to calculate spectral window.
			oversampling (float, optional): Oversampling factor. Default=10.
		"""

		if width is None:
			width = 100*self.df

		freq_cen = 0.5*self.nyquist
		Nfreq = int(oversampling*width/self.df)
		freq = freq_cen + (self.df/oversampling) * np.arange(-Nfreq, Nfreq, 1)

		x = 0.5*np.sin(2*np.pi*freq_cen*self.ls.t) + 0.5*np.cos(2*np.pi*freq_cen*self.ls.t)

		# Calculate power spectrum for the given frequency range:
		ls = LombScargle(self.ls.t, x, center_data=True, fit_mean=self.fit_mean)
		power = ls.power(freq, method='fast', normalization='psd', assume_regular_frequency=True)
		power /= power[int(len(power)/2)] # Normalize to have maximum of one

		freq -= freq_cen
		freq *= 1e6
		return freq, power

	#------------------------------------------------------------------------------
	def plot(self, ax=None, xlabel='Frequency (muHz)', ylabel=None, style='powerspectrum'):

		if ylabel is None:
			ylabel = {
				'powerdensity': 'Power density (ppm^2/muHz)',
				'power': 'Power (ppm^2)',
				'amplitude': 'Amplitude (ppm)'
			}['powerdensity'] # TODO: Only one setting for now...

		if style is None or style == 'powerspectrum':
			style = os.path.join(os.path.dirname(__file__), 'powerspectrum.mplstyle')
		elif style == 'lightkurve':
			style = lightkurve.MPLSTYLE

		with plt.style.context(style):
			if ax is None:
				fig, ax = plt.subplots(1)

			ax.loglog(self.standard[0], self.standard[1], 'k-')
			ax.set_xlabel('Frequency (muHz)')
			ax.set_ylabel(ylabel)
			ax.set_xlim(self.standard[0][0], self.standard[0][-1])

