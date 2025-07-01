from tenpy.models.model import CouplingModel, CouplingMPOModel, MPOModel
from tenpy.algorithms.tdvp import TwoSiteTDVPEngine, SingleSiteTDVPEngine
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.linalg import np_conserved as npc
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from tensorflow.keras.datasets import mnist

import numpy as np
from qutip import *

import warnings
warnings.filterwarnings("ignore")
def evolucion_rho(J_, D_, h_, rho0=None, dt=0.1, state=None, energy=False):
    N = len(D_)

    def sigma(op, i):
        return tensor([op if n == i else qeye(2) for n in range(N)])

    # Construcción del Hamiltoniano
    H = 0 * tensor([qeye(2) for _ in range(N)])

    for i in range(N):
        for j in range(i):
            if J_[i, j] != 0:
                H += J_[j, i] * sigma(sigmax(), i) * sigma(sigmax(), j)

    for i in range(N):
        H += 0.5 * (h_ + D_[i]) * sigma(sigmaz(), i)


    # Operador de evolución U = exp(-i H dt)
    U = (-1j * H * dt).expm()
    # Si se proporciona un estado puro
    if state is not None:
        if isinstance(state, np.ndarray):
            state = state.reshape(-1, 1)
            state = Qobj(state, dims=[[2]*N, [1]*N])
        state_dt = U @ state

        if energy:
            e = (state_dt.dag() * H * state_dt).full().real.item()
            return state_dt, e
        return state_dt

    # Si se proporciona una matriz de densidad
    if rho0 is not None:
        if isinstance(rho0, np.ndarray):
            rho0 = Qobj(rho0, dims=[[2]*N, [2]*N])
        rho_dt = U @ rho0 @ U.dag()
        if energy:
            e = (rho_dt * H).tr().real
            return rho_dt, e
        return rho_dt

import numpy as np

def z_i(n, i):
    """
    Construye el operador Z_i ⊗ I ⊗ ... ⊗ I para un sistema de n qubits.
    
    Parámetros:
    n -- número total de qubits
    i -- índice del qubit al que se aplica el operador Z (0-indexado)
    
    Devuelve:
    Una matriz numpy de dimensión (2^n, 2^n)
    """
    # Matrices básicas
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Construimos la lista de operadores
    ops = []
    for j in range(n):
        if j == i:
            ops.append(Z)
        else:
            ops.append(I)

    # Producto tensorial de todos los operadores
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)

    return result

class CustomHamiltonian(CouplingMPOModel):
    def __init__(self, model_params):
        """Constructor del modelo."""
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
        """Define el tipo de sitio: espines 1/2 en este caso."""
        return SpinHalfSite(conserve=None)  # No conservamos la magnetización total

    def init_terms(self, model_params):
        """Define los términos del Hamiltoniano."""

        site = SpinHalfSite(conserve=None)
        np.random.seed(0)  # Para reproducibilidad
        D = np.random.uniform(-self.W, self.W, size=self.N) #onsite disorder
        J = np.random.uniform(-0.5*self.Js, 0.5*self.Js, size=(self.N, self.N)) #spin-spin coupling
        self.D = D
        self.J = J
        for i in range(self.N):
            self.add_onsite_term(0.5*(self.h+D[i]), i, "Sigmaz")
            for j in range(i+1, self.N):
                self.add_coupling_term(J[i,j], i, j, "Sigmax", "Sigmax")


def get_mnis_data(n_train, n_test):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    ## 

 
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]

    x_test = x_test[:n_test]
    y_test = y_test[:n_test]

    ##
    x_train_vectors = x_train.reshape(-1, 28*28, )
    x_test_vectors = x_test.reshape(-1, 28*28, )

    return x_train_vectors, y_train, x_test_vectors, y_test

def get_state(x, n):
    """Encodes the input data into a quantum state."""


    x_ = np.zeros(2**n, dtype=x.dtype)
    x_[:len(x)] = x  # Copiar los valores de arr

    x = x_/np.linalg.norm(x_)  # Normalize the input

    
    return x

def amplitude_encoding(x, n):
    """Encodes the input data into a quantum state."""


    x_ = np.zeros(2**n, dtype=x.dtype)
    x_[:len(x)] = x  # Copiar los valores de arr

    x = x_/np.linalg.norm(x_)  # Normalize the input

    x = npc.Array.from_ndarray_trivial(x.reshape([2]*n), labels=[f'p{i}' for i in range(n)])
    
    mps = MPS.from_full(sites=[SpinHalfSite(conserve=None)]*n,
                        psi=x, 
                        bc='finite',
                        normalize=True)
    
    return mps