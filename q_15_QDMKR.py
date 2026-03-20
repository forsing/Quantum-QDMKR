"""
QDMKR - Quantum Dual/Multi-Kernel Regression
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/Users/4c/Desktop/GHQ/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
LAMBDA_REG = 0.01
 

def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def compute_kernel(fmap, X_feats):
    n = len(X_feats)
    svs = []
    for feat in X_feats:
        circ = fmap.assign_parameters(feat)
        sv = Statevector.from_instruction(circ)
        svs.append(sv)

    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            fid = abs(svs[i].inner(svs[j])) ** 2
            K[i, j] = fid
            K[j, i] = fid
    return K


def ridge_predict(K, y, lam=LAMBDA_REG):
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(n), y)
    return K @ alpha


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    n_states = 1 << NUM_QUBITS
    X_feats = np.array([value_to_features(v) for v in range(n_states)])

    fmap_zz1 = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)
    fmap_zz2 = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=2)
    fmap_pauli = PauliFeatureMap(feature_dimension=NUM_QUBITS,
                                 reps=1, paulis=['Z', 'ZZ', 'ZZZ'])

    kernels = {
        'ZZ-r1': fmap_zz1,
        'ZZ-r2': fmap_zz2,
        'Pauli-ZZZ': fmap_pauli,
    }

    print(f"\n--- Racunanje 3 kvantna kernela ({NUM_QUBITS}q) ---")
    K_dict = {}
    for name, fmap in kernels.items():
        print(f"  {name}...", end=" ", flush=True)
        K = compute_kernel(fmap, X_feats)
        K_dict[name] = K
        print(f"rang={np.linalg.matrix_rank(K)}")

    K_combined = sum(K_dict.values()) / len(K_dict)
    print(f"  Kombinovani kernel: rang={np.linalg.matrix_rank(K_combined)}")

    print(f"\n--- QDMKR po pozicijama (3 kernela usrednjeno) ---")
    dists = []
    for pos in range(7):
        y = build_empirical(draws, pos)
        pred = ridge_predict(K_combined, y)
        pred = pred - pred.min()
        if pred.sum() > 0:
            pred /= pred.sum()
        dists.append(pred)

        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"  Poz {pos+1} [{MIN_VAL[pos]}-{MAX_VAL[pos]}]: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QDMKR, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Racunanje 3 kvantna kernela (5q) ---
  ZZ-r1... rang=32
  ZZ-r2... rang=32
  Pauli-ZZZ... rang=32
  Kombinovani kernel: rang=32

--- QDMKR po pozicijama (3 kernela usrednjeno) ---
  Poz 1 [1-33]: 1:0.168 | 2:0.147 | 3:0.130
  Poz 2 [2-34]: 8:0.086 | 5:0.076 | 9:0.076
  Poz 3 [3-35]: 13:0.064 | 12:0.063 | 14:0.062
  Poz 4 [4-36]: 23:0.064 | 21:0.063 | 18:0.063
  Poz 5 [5-37]: 29:0.065 | 26:0.064 | 27:0.063
  Poz 6 [6-38]: 33:0.084 | 32:0.081 | 35:0.080
  Poz 7 [7-39]: 7:0.183 | 38:0.153 | 37:0.133

==================================================
Predikcija (QDMKR, deterministicki, seed=39):
[1, 8, 13, 23, 29, 33, 38]
==================================================
"""



"""
QDMKR - Quantum Dual/Multi-Kernel Regression

3 razlicita kvantna kernela istovremeno:
ZZ reps=1 - bazicni ZZ feature map
ZZ reps=2 - dublji ZZ sa vise entanglementa
Pauli Z+ZZ+ZZZ - troqubitne interakcije, hvata trostruke korelacije
Kombinovani kernel = prosek sva tri (kernel averaging/ensemble)
Ridge regresija nad usrednjenim kernelom
Ideja: razliciti feature mapovi "vide" razlicite aspekte strukture, kombinacija daje robusniji model
Deterministicki, bez iterativnog treniranja
"""
