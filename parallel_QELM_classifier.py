from concurrent.futures import ProcessPoolExecutor

from datatools import *
from spinchain import SpinChainHamiltonian
from qtools import *



x_train_vectors, y_train, x_test_vectors, y_test = get_mnist_data(n_train=10000, n_test=5000)

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
hamiltonian = SpinChainHamiltonian(model_params)

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
        'N_steps': 50,     
        'tol': 1e-10
    },
    'max_trunc_error': 1e-10
    
}
def _single_process(vector, hamiltonian, tdvp_params):
    """Computes the correlators mentioned in the theoretical description during a time evolution, in steps,
    using TDVP.
    
    -- Input:
    vector: image vector to be processed
    hamiltonian: TeNPy hamiltonian
    tdvp_params: TDVP parameters for the time evolution

    -- Output:
    features_dic: dictionary with the correlations computed for each time evaluated during the evolution.
    
    """
    mps = get_MPS_from_amplitude_encoding(vector, n)
    features_dic = {}
    tdvp_engine = TwoSiteTDVPEngine(mps, hamiltonian, tdvp_params)
    for k in range(1, 11):
        tdvp_engine.run()

        features = compute_correlators_in_mps(mps)

        features_dic[hamiltonian.Js*tdvp_params['dt']*k] = features

    return features_dic
def _single_process_ed(vector, hamiltonian):
    """Computes the correlators mentioned in the theoretical description during a time evolution, in steps,
    using Exact Diagonalization.
    
    -- Input:
    vector: image vector to be processed
    hamiltonian: TeNPy hamiltonian

    -- Output:
    features_dic: dictionary with the correlations computed for each time evaluated during the evolution.
    
    """
    state = get_quantum_state_from_amplitude_encoding(vector, n)
    features_dic = {}
    for j in range(1, 3):
        state = evolve_state(hamiltonian.J, hamiltonian.D, hamiltonian.h, dt=0.1, state=state, energy=False)

        features = compute_correlators_in_state(state)

        features_dic[2*j] = features
    
    return features_dic


def _single_process_ne(vector):
    """Computes the correlators for an image. Just a wrapper over compute_correlators_in_mps.
    
    -- Input:
    vector: image vector to be processed
 
    -- Output:
    features: correlations computed 
    
    """

    return compute_correlators_in_mps(get_MPS_from_amplitude_encoding(vector, n))

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


#### Points definition

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
## Fourth ergodic area (other point)
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
    'III_to_IV': (h_III_to_IV, W_III_to_IV),
    'III': (h_III, W_III),
    'IV': (h_IV, W_IV),
    'I': (h_I, W_I),
    'II': (h_II, W_II),
    'II_to_III': (h_II_to_III, W_II_to_III),
    'I_to_II': (h_I_to_II, W_I_to_II),
    'IV_to_I': (h_IV_to_I, W_IV_to_I),
    'II_to_IV': (h_II_to_IV, W_II_to_IV)
}


for zone_name, zone_values in zip(zones.keys(), zones.values()):
    h, W = zone_values
    model_params['h'] = h
    model_params['W'] = W
    hamiltonian = SpinChainHamiltonian(model_params)

    X_train = multiprocess(ncores, x_train_vectors, hamiltonian, tdvp_params)
    X_test = multiprocess(ncores, x_test_vectors, hamiltonian, tdvp_params)


        
    time_steps = range(2, 21, 2)  
    for t in time_steps:
        X_train_time = [x_tr[t] for x_tr in X_train]
        X_test_time = [x_tr[t] for x_tr in X_test]

        # Guardar los archivos
        np.savez(f'X_train_TN_10000_time_{t}_zone={zone_name}.npz', x=X_train_time, y=y_train)
        np.savez(f'X_test_TN_5000_time_{t}_zone={zone_name}.npz', x=X_test_time, y=y_test)