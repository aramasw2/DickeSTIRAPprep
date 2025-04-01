from qutip import *
import numpy as np
import numba
from matplotlib import pyplot, animation
from scipy.interpolate import interp1d
from itertools import combinations
from numba import jit
import os
from Qfunctions import*
from Pulses import Pulse
class DickeRaman:
    def __init__(self, Nbits, Ncav, Det, tdet, spem=False,cavdecay=False):
        """Initialize all properties and spin operators, Hamiltonian, etc."""
        self.N = Nbits
        self.Ncav = Ncav
        self.Det = Det
        self.tdet = tdet
        self.spem = spem
        self.cavdecay = cavdecay
        
        #Initialize spin algebra
        self.s1m_list, self.s2m_list, self.s1_list, self.s2_list, \
        self.s3_list, self.cav_list = SysAlgebra(Nbits,Ncav)

        #Collective spin operators
        self.Jx = 1/Nbits*sum(self.s1m_list[n]*self.s2m_list[n].dag()+self.s2m_list[n]*self.s1m_list[n].dag() for n in range(len(self.s1m_list)))
        self.Jy = -1j/Nbits*sum(self.s1m_list[n]*self.s2m_list[n].dag()-self.s2m_list[n]*self.s1m_list[n].dag() for n in  range(len(self.s1m_list)))
        self.Jz = 1/Nbits*sum(self.s1_list[n]-self.s3_list[n] for n in range(len(self.s1_list)))
        self.Jp = self.Jx+1j*self.Jy
        self.Jm = self.Jx-1j*self.Jy
        self.Jsq = self.Jx**2+self.Jy**2+self.Jz**2
        self.J=[self.Jx, self.Jy, self.Jz, self.Jsq]
        
        #Initialize ground state density matrix (Lowest energy, no excitations)
        glist = [basis(3, 0) for _ in range(Nbits)]  # N qutrit ground states
        self.pss0 = tensor(glist+[basis(Ncav,0)])
        self.times= np.array([])
        self.statelist = np.array([])

    def return_operators(self):
        return self.s1m_list, self.s2m_list, self.s1_list, self.s2_list,\
                self.s3_list, self.cav_list

    def initial_state(self,state):
        self.pss0 = state

    def inject_photon(self,npx=1.0,coherent=False):
        current_m = (self.pss0).ptrace(self.N).diag().argmax()  # Find the Fock state with max probability

        if current_m + npx >= self.Ncav:
            raise ValueError(f"Applying a^†^{n} would exceed the max photon number {Ncav-1}.")
            
        if coherent == True:
            D = displace(self.Ncav, npx)
            self.pss0 = tensor(tensor([qeye(3) for _ in range(self.N)]), D)\
                        * self.pss0
        if coherent == False:
            A=(self.cav_list[0].dag())**int(npx)
            self.pss0 = A * self.pss0            

    def construct_ham(self,Pulse1, Pulse2):
        # construct the hamiltonian
        self.Pulse1=Pulse1
        self.Pulse2=Pulse2
        aop=self.cav_list[0]

        #Detunings
        Hd = sum(self.Det * op for op in self.s2_list)+\
             sum(self.tdet * op for op in self.s3_list)


        HintP = sum((op*aop.dag()+op.dag()*aop) for op in self.s1m_list) #Pump cavity interaction
        
        HintS = sum((op+op.dag()) for op in self.s2m_list) #Stokes classical pulse Hamiltonian
        #Initialize Hamiltonian
        self.Ht = [Hd, [HintP,Pulse1], [HintS,Pulse2]]

        return self.Ht

    def freqnoise(self,t,sigma,args):
        return generate_pulse_noise(t,sigma)

    def introduce_freqnoise(self,system="atomic",noise_type="Random",sigma=[0.0,0.0,0.0],noise_freq=100.0):
        aop=self.cav_list[0]
        sigma1, sigma2, sigmac= sigma
        if system == "atomic":
            if noise_type=="Random":
                if sigma1 !=0:
                    self.Ht.extend([[(self.s1_list[n]-self.s2_list[n]),lambda t: self.freqnoise(t, sigma1)]\
                                     for n in range(self.N)])
                if sigma2 !=0:
                    self.Ht.extend([[(self.s2_list[n]-self.s3_list[n]),lambda t: self.freqnoise(t, sigma2)]\
                                     for n in range(self.N)])
            elif noise_type=="Deterministic":
                
                if sigma1 !=0:                    
                    self.Ht.append([sum(self.s1_list[n]-self.s2_list[n] for n in range(self.N)),\
                                     lambda t: sigma1*np.sin(noise_freq*t)])
                if sigma2 !=0:
                    self.Ht.append([sum(self.s2_list[n]-self.s3_list[n] for n in range(self.N)),\
                                     lambda t: sigma2*np.sin(noise_freq*t)])    

                if sigma1 !=0 and sigma2 == 0.0:
                    self.Ht.append([sum(self.s1_list[n]-self.s3_list[n] for n in range(self.N)),\
                                     lambda t: sigma1*np.sin(noise_freq*t)])
        if system == "cavity":
            if noise_type=="Random":
                if sigmac !=0:
                    self.Ht.append([aop.dag()*aop,lambda t: self.freqnoise(t, sigma=sigmac)])
            elif noise_type=="Deterministic":
                if sigmac !=0:
                    self.Ht.append([aop.dag()*aop,lambda t: sigmac*np.sin(noise_freq*t)])

        return self.Ht

    def introduce_drift(self,driftA=0.0):
        if driftA>0.0:
            self.Hdrift=sum(generate_pulse_noise(0.0,sigma=driftA)*\
                            (self.s1_list[n]-self.s3_list[n]) for n in range(len(self.s1_list)))
            self.Ht.append(self.Hdrift)
        return self.Ht

    def global_piswap(self):
        sing_swap = basis(3,0)*basis(3,2).dag() + basis(3,2)*basis(3,0).dag()+ basis(3,1)*basis(3,1).dag()
        self.pss0 = tensor([sing_swap for _ in range(self.N)] + [qeye(self.Ncav)]) * self.pss0
        self.statelist[-1] = self.pss0
        return self.pss0

       
    def introduce_globalpiswap(self,tcx,taux):
        sing_swap = basis(3,0)*basis(3,2).dag() + basis(3,2)*basis(3,0).dag()
        Hswap = sum(tensor([sing_swap if i == n else qeye(3) for i in range(self.N)] + [qeye(self.Ncav)])\
                     for n in range(self.N))
        PiPulse=Pulse(1.0, 0.0, phase=0, pulse_type="pipulse", duration=taux, center=tcx, noise=None,noiseamp=0.0,\
                      freqmult=100.0,sq=False)
        self.Ht.append([Hswap,PiPulse])

    '''For two-photon transitions involving cavity-classical pulse
    coupling, introduce dispersive AC Stark shifts'''


        
    def apply_ACdispersive(self,gx,APulse,Delta0):
        aop=self.cav_list[0]

        
        self.Ht.append(sum(gx**2/Delta0*op*aop.dag()*aop\
                         for op in self.s2_list))

        #self.ACSq= lambda t: APulse(t)**2
        self.ACgrH=[sum(1/Delta0*op for op in self.s1_list),APulse]
        self.Ht.append(self.ACgrH)

        return self.Ht

    def apply_controlfield(self,CPulse):
        self.ctrlH=[self.N*self.Jz,CPulse]
        self.Ht.append(self.ctrlH)

    #Check that states 1 and 3 have the same energy (2P detuning zero)
    def check_ACeffect(self):
        Hevo=QobjEvo(self.Ht)
        return expect(self.Jz, Hevo(0.0))

    def apply_dissipation(self,C1,C2,ka):
        aop=self.cav_list[0]
        self.c_ops=[]
        if self.spem == True:
            self.c_ops.append([np.sqrt(C1)*op for op in self.s1m_list])
            self.c_ops.append([np.sqrt(C2)*op for op in self.s2m_list])
        if self.cavdecay == True:
            self.c_ops.append([np.sqrt(ka)*aop])
    
    def simulate(self,tlist):        
        self.result = mesolve(self.Ht, self.pss0, tlist, self.c_ops,[])
        self.pss0= self.result.states[-1]
        self.times = np.append(self.times, tlist)
        self.statelist = np.append(self.statelist,self.result.states)
        return self.result

    def return_result(self,res="states"):
        if res == "states":
            return self.result.states
        elif res == "operators":
            Jxlst, Jylst, Jzlst, Jsqlst=zip(*[(expect(self.Jx, state), expect(self.Jy, state), expect(self.Jz, state),\
                                   expect(self.Jsq, state)) for state in self.result.states])
            return Jxlst, Jylst, Jzlst, Jsqlst
    
    def simulate_mcwf(self, tlist, num_cpus=20, ntraj=100, options=None):
        # Use MCWF simulation with mesolve or mcsolve
        self.result = mcsolve(self.Ht, self.pss0, tlist, self.c_ops,[], ntraj=ntraj,options=options)
        self.pss0= self.result.states[-1]
        self.times = np.append(self.times, tlist)
        self.statelist = np.append(self.statelist,self.result.states)
        return self.result

    def Bloch_animate(self,filename='bloch_sphere'):
        theta=np.pi/4
        fig = pyplot.figure()
        ax = fig.add_subplot(azim=-40, elev=30, projection="3d")
        sphere = qutip.Bloch(axes=ax)
        Jxlst, Jylst, Jzlst, Jsqlst=zip(*[(expect(self.Jx, state), expect(self.Jy, state), expect(self.Jz, state),\
                               expect(self.Jsq, state)) for state in self.result.states])        
        def animate(i):
           sphere.clear()
           sphere.add_vectors([np.sin(theta), 0, np.cos(theta)], ["r"])
           sphere.add_points([Jxlst[:i+1], Jylst[:i+1], Jzlst[:i+1]])
           sphere.render()
           return ax
        
        ani = animation.FuncAnimation(fig, animate, np.arange(len(Jxlst)), blit=False, repeat=False)
        ani.save(filename + '.mp4', fps=20)
            
