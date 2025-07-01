'''
Functions to perform encoding, evolution and measurements
'''

from tenpy.algorithms.tdvp import TwoSiteTDVPEngine, SingleSiteTDVPEngine
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.linalg import np_conserved as npc
import numpy as np
from qutip import *

def build_hamiltonian(J_, D_, h_):
    """Builds the spin chain Hamiltonian matrix
    -- Input:
    J_, D_, h_ : Hamiltonian parameters

    -- Output:
    H: Hamiltonian matrix
    """
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

    return H

def evolve_state(J_, D_, h_, rho0=None, dt=0.1, state=None, energy=False):
    """Evolves a quantum state (pure or mixed) for a certain dt time
    -- Input: 
    J_, D_, h_ : Hamiltonian parameters
    rho0: density matrix
    dt: evolution time
    state: quantum state
    energy: whether to return the energy of the evolved quantum state

    -- Output:
    state: evolved state by dt
    e: if energy is True, the energy of state
    """

    N = len(D_)
    H = build_hamiltonian(J_, D_, h_)
    U = (-1j * H * dt).expm()


    # If a pure state is given
    if state is not None:
        if isinstance(state, np.ndarray):
            state = state.reshape(-1, 1)
            state = Qobj(state, dims=[[2]*N, [1]*N])
        state_dt = U @ state

        if energy:
            e = (state_dt.dag() * H * state_dt).full().real.item()
            return state_dt, e
        return state_dt

    # If a density matrix is given
    if rho0 is not None:
        if isinstance(rho0, np.ndarray):
            rho0 = Qobj(rho0, dims=[[2]*N, [2]*N])
        rho_dt = U @ rho0 @ U.dag()
        if energy:
            e = (rho_dt * H).tr().real
            return rho_dt, e
        return rho_dt


def z_i(n, i):
    """
    Builds the operator I ⊗ Z_i ⊗ I ⊗ ... ⊗ I for an n-qubits system.
    
    -- Input:
    n: number of qubits
    i: index to which the operator z is applied (0-indexed)
    
    -- Output:
    Numpy matrix (2^n, 2^n)
    """
    # Matrices básicas
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    
    ops = []
    for j in range(n):
        if j == i:
            ops.append(Z)
        else:
            ops.append(I)

    
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)

    return result

def get_quantum_state_from_amplitude_encoding(x, n):
    """Encodes the input data into a quantum state.
    
    -- Input:
    n: number of qubits
    x: vector to be encoded
    
    -- Output:
    quantum state (2^n, )
    """


    x_ = np.zeros(2**n, dtype=x.dtype)
    x_[:len(x)] = x  

    x = x_/np.linalg.norm(x_)  

    
    return x

def get_MPS_from_amplitude_encoding(x, n):
    """Encodes the input data into a quantum state and then its corresponding MPS.

    -- Input:
    n: number of qubits
    x: vector to be encoded
    
    -- Output:
    MPS of the encoded data into the quantum state
    """

    x = get_quantum_state_from_amplitude_encoding(x, n)

    x = npc.Array.from_ndarray_trivial(x.reshape([2]*n), labels=[f'p{i}' for i in range(n)])
    
    mps = MPS.from_full(sites=[SpinHalfSite(conserve=None)]*n,
                        psi=x, 
                        bc='finite',
                        normalize=True)
    
    return mps

def compute_correlators_in_state(state):
    """Computes a list of measurements of correlators over the MPS, according to the theoretical description.

    -- Input:
    state: state of the system

    -- Output:
    features: list of measurements
    """

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
    return features

def compute_correlators_in_mps(mps):
    """Computes a list of measurements of correlators over the MPS, according to the theoretical description.

    -- Input:
    mps: MPS of the state

    -- Output:
    features: list of measurements
    """
    z = mps.sites[0].get_op('Sigmaz')

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
    
    return features

