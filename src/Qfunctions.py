from qutip import *
import numpy as np
from matplotlib import pyplot, animation
from scipy.interpolate import interp1d
from itertools import combinations
from numba import jit
import os

def SysAlgebra(N,Ncav):
    #Qutrit single operators
    si = qeye(3)
    s1m = basis(3,0)*basis(3,1).dag()
    s2m = basis(3,2)*basis(3,1).dag()
    s1 = basis(3,0)*basis(3,0).dag()
    s2 = basis(3,1)*basis(3,1).dag()
    s3 = basis(3,2)*basis(3,2).dag()
    
    #Photon operators
    a = destroy(Ncav)
    Id_cav = qeye(Ncav)
    
    s1m_list = []
    s2m_list = []
    s1_list = []
    s2_list = []
    s3_list = []
    
    cav_list=[]

    for n in range(N):
        op_list = [si] * N + [Id_cav]  # Identity on all sites + cavity mode        
       
        op_list[n] = s1m
        s1m_list.append(tensor(op_list))
        
        op_list[n] = s2m
        s2m_list.append(tensor(op_list))
        
        op_list[n] = s1
        s1_list.append(tensor(op_list))
    
        op_list[n] = s2
        s2_list.append(tensor(op_list))

        op_list[n] = s3
        s3_list.append(tensor(op_list))
        
    cavops=[si]*N
    cavops.append(a)
    cav_list.append(tensor(cavops))
    return s1m_list, s2m_list, s1_list, s2_list, s3_list, cav_list

def dicke_state(N, k):
    """
    Generates the symmetric Dicke state |N, k> in QuTiP.
    
    Parameters:
        N (int): Number of qubits
        k (int): Number of excitations

    Returns:
        Qobj: QuTiP quantum object representing the Dicke state.
    """
    # Generate all computational basis states with k excitations
    basis_states = []
    for positions in combinations(range(N), k):
        state = [basis(2, 1) if i in positions else basis(2, 0) for i in range(N)]
        basis_states.append(tensor(state))
    
    # Create the symmetric superposition
    dicke = sum(basis_states) / np.sqrt(len(basis_states))
    
    return dicke


def generate_pulse_noise(t,sigma=0.1):
    """Generate Gaussian white noise with mean 0 and variance sigma^2"""
    if np.ndim(t) > 0:  # If t is an array (e.g., time list)
        return np.random.normal(0, sigma, size=len(t))
    else:  # If t is a single value
        return np.random.normal(0, sigma)

@jit(nopython=True)
def fast_piecewise_noise(t, t_vals, noise_vals):
    """Fast noise retrieval using binning (no interpolation)."""
    index = np.searchsorted(t_vals, t) - 1
    return noise_vals[max(0, min(index, len(noise_vals) - 1))]
    
def generate_w_state(N_atoms,Ncav):

    # Define basis states for qutrits (3-level atoms)
    g = basis(3, 0)  # Ground state
    e = basis(3, 2)  # Excited state
    
    # Define vacuum state for cavity
    vac_cavity = basis(Ncav, 0)
    
    # Generate the three W-state components
    w_components = []
    for i in range(N_atoms):
        atom_states = [g] * N_atoms  # All atoms start in ground state
        atom_states[i] = (1.0)**(i+1)*e  # Excite one atom
        w_components.append(tensor(atom_states + [vac_cavity]))  # Append tensor product
    
    # Normalize the W-state
    w_state = sum(w_components) / np.sqrt(N_atoms)
    
    return w_state

def generate_dicke_state(N_atoms, N_excitations, Ncav):
    """
    Generate a Dicke state |N, n> where N is the number of atoms (qubits) and n is the number of excitations.

    Parameters:
        N_atoms (int): Number of atoms (qubits).
        N_excitations (int): Number of excitations in the Dicke state.
        Ncav (int): Number of cavity modes.

    Returns:
        Qobj: QuTiP quantum object representing the Dicke state.
    """
    # Define basis states for qubits (3-level atoms)
    g = basis(3, 0)  # Ground state
    e = basis(3, 2)  # Excited state
    
    # Define vacuum state for cavity
    vac_cavity = basis(Ncav, 0)
    
    # Generate all possible combinations of N_excitations excitations across the N_atoms
    basis_states = []
    for positions in combinations(range(N_atoms), N_excitations):
        state = [g if i not in positions else e for i in range(N_atoms)]
        basis_states.append(tensor(state + [vac_cavity]))
    
    # Normalize the state
    dicke_state = sum(basis_states) / np.sqrt(len(basis_states))
    
    return dicke_state



def ACSq(t,Apulse,args):
    return Apulse(t)**2
