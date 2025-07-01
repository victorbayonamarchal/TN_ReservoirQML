'''
Class to build a spain chain-like Hamiltonian in TeNPy
'''
from tenpy.models.model import CouplingModel, CouplingMPOModel, MPOModel
from tenpy.networks.site import SpinHalfSite

import numpy as np

class SpinChainHamiltonian(CouplingMPOModel):
    def __init__(self, model_params):

        self.model_params = model_params
        self.Js = model_params["Js"]
        self.W = model_params["W"]
        self.h = model_params["h"]
        self.N = model_params["L"]
        self.bc_MPS = model_params["bc_MPS"]
        self.bc_x = model_params["bc_x"]
        self.lattice = model_params["lattice"]
        self.D = np.zeros(self.N)
        self.J = np.zeros((self.N, self.N))
        super().__init__(model_params)

    
    def init_sites(self, model_params):
        """Define local Hilbert spaces: 1/2-spins in this case"""
        return SpinHalfSite(conserve=None) 

    def init_terms(self, model_params):
        """Defin the Hamiltonian parameters"""

        np.random.seed(0)  # For reporudcibility 
        D = np.random.uniform(-self.W, self.W, size=self.N) #onsite disorder
        J = np.random.uniform(-0.5*self.Js, 0.5*self.Js, size=(self.N, self.N)) #spin-spin coupling
        self.D = D
        self.J = J
        for i in range(self.N):
            self.add_onsite_term(0.5*(self.h+D[i]), i, "Sigmaz")
            for j in range(i+1, self.N):
                self.add_coupling_term(J[i,j], i, j, "Sigmax", "Sigmax")