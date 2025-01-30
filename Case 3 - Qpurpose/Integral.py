from networkx import maximum_spanning_edges
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np


def Integral(final_circ: QuantumCircuit, shots: int = 10e6):
    """
    Inputs:
    State: Time Evolved State, it is just after QVST, with the 3 quantum registers

    Outputs:
    sum: The integral of the state, in the expected boundaries.
    """

    n = qreg.size[1]
    N = 2**n

    creg = final_circ.cregs
    qreg = final_circ.qregs

    final_circ.measure(qreg[0], creg[0])
    final_circ.measure(qreg[1], creg[1])
    final_circ.measure(qreg[2], creg[2])

    sim = AerSimulator()
    qc_comp = transpile(final_circ, sim)
    res = sim.run(qc_comp, shots=shots).result()
    counts = res.get_counts(0)

    select = (n + 1) * "0"
    total = 0
    z = np.zeros(N)
    for key in counts:
        L = key.split()
        if L[1] + L[2] == select:
            z[int(L[0], 2)] = np.sqrt(counts[key] / shots)
            total += counts[key]
    succesRate = total / shots
    print("Succes rate: " + str(succesRate))

    imin = N / 4
    imax = 3 * N / 4
    sum = 0

    for i in range(imin, imax):
        sum += z[i] * 4 / N * 1 / succesRate

    return sum


def find_max(final_circ: QuantumCircuit, shots: int = 10e6):
    """Find the maximum coefficient of the state.

    Inputs:
    State: Time Evolved State, it is just after QVST, with the 3 quantum registers

    Outputs:
    sum: The integral of the state, in the expected boundaries.
    """

    n = qreg.size[1]
    N = 2**n

    creg = final_circ.cregs
    qreg = final_circ.qregs

    final_circ.measure(qreg[0], creg[0])
    final_circ.measure(qreg[1], creg[1])
    final_circ.measure(qreg[2], creg[2])

    sim = AerSimulator()
    qc_comp = transpile(final_circ, sim)
    res = sim.run(qc_comp, shots=shots).result()
    counts = res.get_counts(0)

    select = (n + 1) * "0"
    total = 0
    z = np.zeros(N)

    maximum = 0.0
    for key in counts:
        L = key.split()
        if L[1] + L[2] == select:
            z[int(L[0], 2)] = np.sqrt(counts[key] / shots)
            if counts[key] > maximum:
                maximum = key
            else:
                continue

    return maximum
