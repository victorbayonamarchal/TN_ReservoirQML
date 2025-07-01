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

(x_train, y_train), (x_test, y_test) = mnist.load_data()
## 

train_n = 10000
x_train = x_train[:train_n]
y_train = y_train[:train_n]

x_test = x_test[:5000]
y_test = y_test[:5000]

##
x_train_vectors = x_train.reshape(-1, 28*28, )
x_test_vectors = x_test.reshape(-1, 28*28, )

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

n=10
model_params = {"lattice": "Chain",
                "L": n,
                "bc_MPS": "finite",
                "bc_x": "periodic",
                "Js": 20,
                "W": 1,
                "h": 0.3,
                "N": n
                }
hamiltonian = CustomHamiltonian(model_params)

X_train = []

tdvp_params = {
    'start_time': 0,
    'dt': 0.1,
    'max_dt':0.5,
    'N_steps': 1,
    'trunc_params': {
        'chi_max': 50,
        'svd_min': 1.e-15,
        'trunc_cut': None
    },
    'lanczos_params': {
        'N_steps': 50,     # Más vectores para explorar mejor el subespacio
        'tol': 1e-10
    },
    'max_trunc_error': 1e-10
    
}
def _single_process(vector, hamiltonian, tdvp_params):
    mps = amplitude_encoding(vector, 10)
    z = mps.sites[0].get_op('Sigmaz')
    features_dic = {}
    tdvp_engine = TwoSiteTDVPEngine(mps, hamiltonian, tdvp_params)
    for k in range(1, 11):
        tdvp_engine.run()

        features = np.concatenate([
        mps.expectation_value(z),  # shape (10,)

        np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+1]).item() for i in range(n-1)]),  # shape (9,)
        np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+2]).item() for i in range(n-2)]),
        np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+3]).item() for i in range(n-3)]),
        np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+4]).item() for i in range(n-4)]),
        np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+5]).item() for i in range(n-5)]),
        np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+6]).item() for i in range(n-6)]),
        np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+7]).item() for i in range(n-7)]),
        np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+8]).item() for i in range(n-8)]),
        np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+9]).item() for i in range(n-9)]),

        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2)]).item() for i in range(n-2)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2), ('Sigmaz', i+3)]).item() for i in range(n-3)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2), ('Sigmaz', i+3), ('Sigmaz', i+4)]).item() for i in range(n-4)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2), ('Sigmaz', i+3), ('Sigmaz', i+4), ('Sigmaz', i+5)]).item() for i in range(n-5)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2), ('Sigmaz', i+3), ('Sigmaz', i+4), ('Sigmaz', i+5), ('Sigmaz', i+6)]).item() for i in range(n-6)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2), ('Sigmaz', i+3), ('Sigmaz', i+4), ('Sigmaz', i+5), ('Sigmaz', i+6), ('Sigmaz', i+7)]).item() for i in range(n-7)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2), ('Sigmaz', i+3), ('Sigmaz', i+4), ('Sigmaz', i+5), ('Sigmaz', i+6), ('Sigmaz', i+7), ('Sigmaz', i+8)]).item() for i in range(n-8)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2), ('Sigmaz', i+3), ('Sigmaz', i+4), ('Sigmaz', i+5), ('Sigmaz', i+6), ('Sigmaz', i+7), ('Sigmaz', i+8), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
        
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+4)]).item() for i in range(n-4)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+2), ('Sigmaz', i+4)]).item() for i in range(n-4)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+3), ('Sigmaz', i+4)]).item() for i in range(n-4)]),

        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+2), ('Sigmaz', i+5)]).item() for i in range(n-5)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+3), ('Sigmaz', i+5)]).item() for i in range(n-5)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+4), ('Sigmaz', i+5)]).item() for i in range(n-5)]),

        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+6)]).item() for i in range(n-6)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+2), ('Sigmaz', i+6)]).item() for i in range(n-6)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+3), ('Sigmaz', i+6)]).item() for i in range(n-6)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+4), ('Sigmaz', i+6)]).item() for i in range(n-6)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+5), ('Sigmaz', i+6)]).item() for i in range(n-6)]),

        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+7)]).item() for i in range(n-7)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+2), ('Sigmaz', i+7)]).item() for i in range(n-7)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+3), ('Sigmaz', i+7)]).item() for i in range(n-7)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+4), ('Sigmaz', i+7)]).item() for i in range(n-7)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+5), ('Sigmaz', i+7)]).item() for i in range(n-7)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+6), ('Sigmaz', i+7)]).item() for i in range(n-7)]),

        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+8)]).item() for i in range(n-8)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+2), ('Sigmaz', i+8)]).item() for i in range(n-8)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+3), ('Sigmaz', i+8)]).item() for i in range(n-8)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+4), ('Sigmaz', i+8)]).item() for i in range(n-8)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+5), ('Sigmaz', i+8)]).item() for i in range(n-8)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+6), ('Sigmaz', i+8)]).item() for i in range(n-8)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+7), ('Sigmaz', i+8)]).item() for i in range(n-8)]),


        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+2), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+3), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+4), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+5), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+6), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+7), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
        np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+8), ('Sigmaz', i+9)]).item() for i in range(n-9)])

        ])
        features_dic[2*k] = features

    return features_dic
def _single_process_ed(vector, hamiltonian, tdvp_params):
    state = get_state(vector, 10)
    features_dic = {}
    for j in range(1, 3):
        state = evolucion_rho(hamiltonian.J, hamiltonian.D, hamiltonian.h, dt=0.1, state=state, energy=False)

        features = ([np.array(state.full().conj().T@z_i(n,i)@state.full()).reshape(1,) for i in range(n)] + 
                    
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n,i+1)@state.full()).reshape(1,) for i in range(n-1)] + 
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n,i+2)@state.full()).reshape(1,) for i in range(n-2)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n,i+3)@state.full()).reshape(1,) for i in range(n-3)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n,i+4)@state.full()).reshape(1,) for i in range(n-4)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n,i+5)@state.full()).reshape(1,) for i in range(n-5)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n,i+6)@state.full()).reshape(1,) for i in range(n-6)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n,i+7)@state.full()).reshape(1,) for i in range(n-7)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n,i+8)@state.full()).reshape(1,) for i in range(n-8)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n,i+9)@state.full()).reshape(1,) for i in range(n-9)] +

                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+1)@z_i(n,i+2)@state.full()).reshape(1,) for i in range(n-2)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+1)@z_i(n,i+2)@z_i(n,i+3)@state.full()).reshape(1,) for i in range(n-3)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+1)@z_i(n,i+2)@z_i(n,i+3)@z_i(n,i+4)@state.full()).reshape(1,) for i in range(n-4)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+1)@z_i(n,i+2)@z_i(n,i+3)@z_i(n,i+4)@z_i(n,i+5)@state.full()).reshape(1,) for i in range(n-5)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+1)@z_i(n,i+2)@z_i(n,i+3)@z_i(n,i+4)@z_i(n,i+5)@z_i(n,i+6)@state.full()).reshape(1,) for i in range(n-6)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+1)@z_i(n,i+2)@z_i(n,i+3)@z_i(n,i+4)@z_i(n,i+5)@z_i(n,i+6)@z_i(n,i+7)@state.full()).reshape(1,) for i in range(n-7)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+1)@z_i(n,i+2)@z_i(n,i+3)@z_i(n,i+4)@z_i(n,i+5)@z_i(n,i+6)@z_i(n,i+7)@z_i(n,i+8)@state.full()).reshape(1,) for i in range(n-8)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+1)@z_i(n,i+2)@z_i(n,i+3)@z_i(n,i+4)@z_i(n,i+5)@z_i(n,i+6)@z_i(n,i+7)@z_i(n,i+8)@z_i(n,i+9)@state.full()).reshape(1,) for i in range(n-9)] +

                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+1)@z_i(n,i+4)@state.full()).reshape(1,) for i in range(n-4)] + 
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+2)@z_i(n,i+4)@state.full()).reshape(1,) for i in range(n-4)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+3)@z_i(n,i+4)@state.full()).reshape(1,) for i in range(n-4)] +

                    #[np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+1)@z_i(n,i+5)@state.full()).reshape(1,) for i in range(n-5)] + 
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+2)@z_i(n,i+5)@state.full()).reshape(1,) for i in range(n-5)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+3)@z_i(n,i+5)@state.full()).reshape(1,) for i in range(n-5)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+4)@z_i(n,i+5)@state.full()).reshape(1,) for i in range(n-5)] +  

                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+1)@z_i(n,i+6)@state.full()).reshape(1,) for i in range(n-6)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+2)@z_i(n,i+6)@state.full()).reshape(1,) for i in range(n-6)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+3)@z_i(n,i+6)@state.full()).reshape(1,) for i in range(n-6)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+4)@z_i(n,i+6)@state.full()).reshape(1,) for i in range(n-6)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+5)@z_i(n,i+6)@state.full()).reshape(1,) for i in range(n-6)] +   

                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+1)@z_i(n,i+7)@state.full()).reshape(1,) for i in range(n-7)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+2)@z_i(n,i+7)@state.full()).reshape(1,) for i in range(n-7)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+3)@z_i(n,i+7)@state.full()).reshape(1,) for i in range(n-7)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+4)@z_i(n,i+7)@state.full()).reshape(1,) for i in range(n-7)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+5)@z_i(n,i+7)@state.full()).reshape(1,) for i in range(n-7)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+6)@z_i(n,i+7)@state.full()).reshape(1,) for i in range(n-7)] +  

                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+1)@z_i(n,i+8)@state.full()).reshape(1,) for i in range(n-8)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+2)@z_i(n,i+8)@state.full()).reshape(1,) for i in range(n-8)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+3)@z_i(n,i+8)@state.full()).reshape(1,) for i in range(n-8)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+4)@z_i(n,i+8)@state.full()).reshape(1,) for i in range(n-8)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+5)@z_i(n,i+8)@state.full()).reshape(1,) for i in range(n-8)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+6)@z_i(n,i+8)@state.full()).reshape(1,) for i in range(n-8)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+7)@z_i(n,i+8)@state.full()).reshape(1,) for i in range(n-8)] +   

                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+1)@z_i(n,i+9)@state.full()).reshape(1,) for i in range(n-9)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+2)@z_i(n,i+9)@state.full()).reshape(1,) for i in range(n-9)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+3)@z_i(n,i+9)@state.full()).reshape(1,) for i in range(n-9)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+4)@z_i(n,i+9)@state.full()).reshape(1,) for i in range(n-9)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+5)@z_i(n,i+9)@state.full()).reshape(1,) for i in range(n-9)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+6)@z_i(n,i+9)@state.full()).reshape(1,) for i in range(n-9)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+7)@z_i(n,i+9)@state.full()).reshape(1,) for i in range(n-9)] +
                    [np.array(state.full().conj().T@z_i(n,i)@z_i(n, i+8)@z_i(n,i+9)@state.full()).reshape(1,) for i in range(n-9)] 

                    )
        features_dic[2*j] = features
    
    return features_dic


def _single_process_ne(vector, hamiltonian, tdvp_params):
    mps = amplitude_encoding(vector, 10)
    z = mps.sites[0].get_op('Sigmaz')
    features_dic = {}


    features = np.concatenate([
    mps.expectation_value(z),  # shape (10,)

    np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+1]).item() for i in range(n-1)]),  # shape (9,)
    np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+2]).item() for i in range(n-2)]),
    np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+3]).item() for i in range(n-3)]),
    np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+4]).item() for i in range(n-4)]),
    np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+5]).item() for i in range(n-5)]),
    np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+6]).item() for i in range(n-6)]),
    np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+7]).item() for i in range(n-7)]),
    np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+8]).item() for i in range(n-8)]),
    np.array([mps.correlation_function(ops1=z, ops2=z, sites1=[i], sites2=[i+9]).item() for i in range(n-9)]),

    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2)]).item() for i in range(n-2)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2), ('Sigmaz', i+3)]).item() for i in range(n-3)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2), ('Sigmaz', i+3), ('Sigmaz', i+4)]).item() for i in range(n-4)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2), ('Sigmaz', i+3), ('Sigmaz', i+4), ('Sigmaz', i+5)]).item() for i in range(n-5)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2), ('Sigmaz', i+3), ('Sigmaz', i+4), ('Sigmaz', i+5), ('Sigmaz', i+6)]).item() for i in range(n-6)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2), ('Sigmaz', i+3), ('Sigmaz', i+4), ('Sigmaz', i+5), ('Sigmaz', i+6), ('Sigmaz', i+7)]).item() for i in range(n-7)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2), ('Sigmaz', i+3), ('Sigmaz', i+4), ('Sigmaz', i+5), ('Sigmaz', i+6), ('Sigmaz', i+7), ('Sigmaz', i+8)]).item() for i in range(n-8)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+2), ('Sigmaz', i+3), ('Sigmaz', i+4), ('Sigmaz', i+5), ('Sigmaz', i+6), ('Sigmaz', i+7), ('Sigmaz', i+8), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
    
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+4)]).item() for i in range(n-4)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+2), ('Sigmaz', i+4)]).item() for i in range(n-4)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+3), ('Sigmaz', i+4)]).item() for i in range(n-4)]),

    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+2), ('Sigmaz', i+5)]).item() for i in range(n-5)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+3), ('Sigmaz', i+5)]).item() for i in range(n-5)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+4), ('Sigmaz', i+5)]).item() for i in range(n-5)]),

    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+6)]).item() for i in range(n-6)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+2), ('Sigmaz', i+6)]).item() for i in range(n-6)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+3), ('Sigmaz', i+6)]).item() for i in range(n-6)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+4), ('Sigmaz', i+6)]).item() for i in range(n-6)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+5), ('Sigmaz', i+6)]).item() for i in range(n-6)]),

    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+7)]).item() for i in range(n-7)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+2), ('Sigmaz', i+7)]).item() for i in range(n-7)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+3), ('Sigmaz', i+7)]).item() for i in range(n-7)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+4), ('Sigmaz', i+7)]).item() for i in range(n-7)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+5), ('Sigmaz', i+7)]).item() for i in range(n-7)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+6), ('Sigmaz', i+7)]).item() for i in range(n-7)]),

    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+8)]).item() for i in range(n-8)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+2), ('Sigmaz', i+8)]).item() for i in range(n-8)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+3), ('Sigmaz', i+8)]).item() for i in range(n-8)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+4), ('Sigmaz', i+8)]).item() for i in range(n-8)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+5), ('Sigmaz', i+8)]).item() for i in range(n-8)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+6), ('Sigmaz', i+8)]).item() for i in range(n-8)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+7), ('Sigmaz', i+8)]).item() for i in range(n-8)]),


    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+2), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+3), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+4), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+5), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+6), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+7), ('Sigmaz', i+9)]).item() for i in range(n-9)]),
    np.array([mps.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+8), ('Sigmaz', i+9)]).item() for i in range(n-9)])

    ])
    #features_dic[2*k] = features

    return features

def multiprocess(ncores, x_vectors, hamiltonian, tdvp_params):
    X_list = []

    procs = []
    with ProcessPoolExecutor(ncores) as executor:
        for i in range(len(x_vectors)):
            proc = executor.submit(_single_process, x_vectors[i], hamiltonian, tdvp_params)
            procs.append(proc)
        for i in range(len(x_vectors)):
            X_list.append(procs[i].result() )
            print(f'Procesada imagen {i+1}/{len(x_vectors)}')
            
    return X_list    

ncores = 8 # Number of cores

## First localized area
h_I = 20
W_I = 2000
## Third localized area
h_III = 0.2
W_III = 0.2
## Second ergodic area
h_II = 2
W_II = 100
## Fourth ergodic area
h_IV = 2000
W_IV = 0.2

h_IV_alt = 20
W_IV_alt = 0.2

## Transition I -> II
h_I_to_II = 2
W_I_to_II = 200
## Transition II -> III
h_II_to_III = 2
W_II_to_III = 10
# Transition III -> IV
h_III_to_IV = 10
W_III_to_IV = 2
# Transition IV -> I
h_IV_to_I = 200
W_IV_to_I = 200
# Transition II -> IV
h_II_to_IV = 20
W_II_to_IV = 20


zones = {
    'IV_alt': (h_IV_alt, W_IV_alt),
#    'III_to_IV': (h_III_to_IV, W_III_to_IV),
#    'III': (h_III, W_III),
#    'IV': (h_IV, W_IV),
#    'I': (h_I, W_I),
#    'II': (h_II, W_II),
#    'II_to_III': (h_II_to_III, W_II_to_III),
#    'I_to_II': (h_I_to_II, W_I_to_II),
#    'IV_to_I': (h_IV_to_I, W_IV_to_I)
#    'II_to_IV': (h_II_to_IV, W_II_to_IV)
}

#for zone_name, zone_values in zip(zones.keys(), zones.values()):
#    h, W = zone_values
#    model_params['h'] = h
#    model_params['W'] = W
#    hamiltonian = CustomHamiltonian(model_params)
#
#    X_train = multiprocess(ncores, x_train_vectors, hamiltonian, tdvp_params)
#    X_test = multiprocess(ncores, x_test_vectors, hamiltonian, tdvp_params)
#
#    np.savez(f'X_train_TN_10000_time_0.npz', x=X_train, y=y_train)
#    np.savez(f'X_test_TN_5000_time_0.npz', x=X_test, y=y_test)

for zone_name, zone_values in zip(zones.keys(), zones.values()):
    h, W = zone_values
    model_params['h'] = h
    model_params['W'] = W
    hamiltonian = CustomHamiltonian(model_params)

    X_train = multiprocess(ncores, x_train_vectors, hamiltonian, tdvp_params)
    X_test = multiprocess(ncores, x_test_vectors, hamiltonian, tdvp_params)


    X_train_TN_10000_time_2 = [x_tr[2] for x_tr in X_train]
    X_train_TN_10000_time_4 = [x_tr[4] for x_tr in X_train]
    X_train_TN_10000_time_6 = [x_tr[6] for x_tr in X_train]
    X_train_TN_10000_time_8 = [x_tr[8] for x_tr in X_train]
    X_train_TN_10000_time_10 = [x_tr[10] for x_tr in X_train]
    X_train_TN_10000_time_12 = [x_tr[12] for x_tr in X_train]
    X_train_TN_10000_time_14 = [x_tr[14] for x_tr in X_train]
    X_train_TN_10000_time_16 = [x_tr[16] for x_tr in X_train]
    X_train_TN_10000_time_18 = [x_tr[18] for x_tr in X_train]
    X_train_TN_10000_time_20 = [x_tr[20] for x_tr in X_train]

    X_test_TN_5000_time_2 = [x_tr[2] for x_tr in X_test]
    X_test_TN_5000_time_4 = [x_tr[4] for x_tr in X_test]
    X_test_TN_5000_time_6 = [x_tr[6] for x_tr in X_test]
    X_test_TN_5000_time_8 = [x_tr[8] for x_tr in X_test]
    X_test_TN_5000_time_10 = [x_tr[10] for x_tr in X_test]
    X_test_TN_5000_time_12 = [x_tr[12] for x_tr in X_test]
    X_test_TN_5000_time_14 = [x_tr[14] for x_tr in X_test]
    X_test_TN_5000_time_16 = [x_tr[16] for x_tr in X_test]
    X_test_TN_5000_time_18 = [x_tr[18] for x_tr in X_test]
    X_test_TN_5000_time_20 = [x_tr[20] for x_tr in X_test]
        
    # Save all the data

    np.savez(f'X_train_ED_10000_time_2_zone={zone_name}.npz', x=X_train_TN_10000_time_2, y=y_train)
    np.savez(f'X_test_ED_5000_time_2_zone={zone_name}.npz', x=X_test_TN_5000_time_2, y=y_test)

    np.savez(f'X_train_ED_10000_time_4_zone={zone_name}.npz', x=X_train_TN_10000_time_4, y=y_train)
    np.savez(f'X_test_ED_5000_time_4_zone={zone_name}.npz', x=X_test_TN_5000_time_4, y=y_test)

    np.savez(f'X_train_ED_10000_time_6_zone={zone_name}.npz', x=X_train_TN_10000_time_6, y=y_train)
    np.savez(f'X_test_ED_5000_time_6_zone={zone_name}.npz', x=X_test_TN_5000_time_6, y=y_test)

    np.savez(f'X_train_ED_10000_time_8_zone={zone_name}.npz', x=X_train_TN_10000_time_8, y=y_train)
    np.savez(f'X_test_ED_5000_time_8_zone={zone_name}.npz', x=X_test_TN_5000_time_8, y=y_test)  

    np.savez(f'X_train_ED_10000_time_10_zone={zone_name}.npz', x=X_train_TN_10000_time_10, y=y_train)
    np.savez(f'X_test_ED_5000_time_10_zone={zone_name}.npz', x=X_test_TN_5000_time_10, y=y_test)

    np.savez(f'X_train_ED_10000_time_12_zone={zone_name}.npz', x=X_train_TN_10000_time_12, y=y_train)
    np.savez(f'X_test_ED_5000_time_12_zone={zone_name}.npz', x=X_test_TN_5000_time_12, y=y_test)    

    np.savez(f'X_train_ED_10000_time_14_zone={zone_name}.npz', x=X_train_TN_10000_time_14, y=y_train)
    np.savez(f'X_test_ED_5000_time_14_zone={zone_name}.npz', x=X_test_TN_5000_time_14, y=y_test)

    np.savez(f'X_train_ED_10000_time_16_zone={zone_name}.npz', x=X_train_TN_10000_time_16, y=y_train)
    np.savez(f'X_test_ED_5000_time_16_zone={zone_name}.npz', x=X_test_TN_5000_time_16, y=y_test)

    np.savez(f'X_train_ED_10000_time_18_zone={zone_name}.npz', x=X_train_TN_10000_time_18, y=y_train)
    np.savez(f'X_test_ED_5000_time_18_zone={zone_name}.npz', x=X_test_TN_5000_time_18, y=y_test)

    np.savez(f'X_train_ED_10000_time_20_zone={zone_name}.npz', x=X_train_TN_10000_time_20, y=y_train)
    np.savez(f'X_test_ED_5000_time_20_zone={zone_name}.npz', x=X_test_TN_5000_time_20, y=y_test)