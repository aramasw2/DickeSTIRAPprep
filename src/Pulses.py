from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, animation
from scipy.interpolate import interp1d
from itertools import combinations
from numba import jit
import os
from Qfunctions import*

class Pulse:
    def __init__(self, amplitude, frequency, phase=0, pulse_type="Gaussian", duration=None, center=None, noise=None,noiseamp=0.0,freqmult=100.0):
        self.A = amplitude
        self.w = frequency
        self.phi = phase
        self.tau = duration
        self.noise = noise
        self.na=noiseamp
        self.pulse_type = pulse_type.lower()
        self.tc = center
        self.precompute_noise = False
        self.nfreq=freqmult/duration

    def gaussian_pulse(self, t):
        """Gaussian pulse shape."""
        return self.A*np.exp(-((t-self.tc)**2)/(2*(self.tau)**2))*np.exp(1j*(self.w*t+self.phi))
    
    def square_pulse(self, t):
        """Square pulse shape."""
        if -self.tau+self.tc <= t <= self.tau+self.tc:
            return self.A*np.exp(1j*(self.w*t+self.phi))
        else:
            return 0.0

    def sawtooth_pulse(self, t):
        """Sawtooth pulse shape."""
        period = self.tau if self.duration else 1
        value = self.A * (2 * (t/period - np.floor(t / period + 0.5)))  # Linear ramp between -1 and 1
        return value**np.exp(1j*(self.w*t+self.phi))

    def constant_pulse(self, t):
        """Constant pulse shape."""
        return self.A * np.cos(self.w*t +self.phi)

    def precomputing_noise(self,tlist):
        self.precompute_noise = True
        # Initialize RNG state for Numba's fast noise
        #self.rng_state = numba.random.create_rng(42)  # Seed for reproducibility

        #noise_vals = self.noise(tlist, sigma=self.na)
        #self.noise_func = interp1d(tlist, noise_vals, kind='linear', fill_value="extrapolate")
        self.t_vals=tlist
        self.noise_vals=np.random.normal(0.0, self.na, len(tlist))
        
    def __call__(self, t):
        """Evaluate the pulse at a given time t based on pulse type."""
        if self.pulse_type == 'gaussian':
            pulse_value = self.gaussian_pulse(t)
        elif self.pulse_type == 'square':
            pulse_value = self.square_pulse(t)
        elif self.pulse_type == 'sawtooth':
            pulse_value = self.sawtooth_pulse(t)
        elif self.pulse_type == 'constant':
            pulse_value = self.constant_pulse(t)
        else:
            raise ValueError("Invalid pulse type specified. Choose from 'gaussian', 'square', 'sawtooth', 'constant'.")
        
        # Add noise if the noise function is defined
        if self.noise is not None and self.na !=0:
            if self.precompute_noise == True:
                pulse_value = (1.0+self.na*np.sin(self.nfreq*t))*pulse_value
            else:
                pulse_value = (1.0+fast_piecewise_noise(t, self.t_vals, self.noise_vals))*pulse_value
        if np.imag(pulse_value) == 0.0:
            return np.real(pulse_value)
        else:
            return pulse_value
    def plot_pulse(self,tlist):
        # Generate the pulse using __call__
        pulse = np.array([self(t) for t in tlist])
        
        # Plot the pulse
        plt.figure(figsize=(10, 4))
        plt.plot(tlist, pulse, label=f'{self.pulse_type} Pulse')
        plt.title('Pulse Waveform')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show()
