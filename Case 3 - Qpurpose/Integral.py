def Integral (state):
    """
    Inputs:
    State: Time Evolved State, it is just after QVST, with the 3 quantum registers

    Outputs: 
    sum: The integral of the state, in the expected boundaries
    """

    n = qreg.size[1]
    N = 2**n

    creg = state.cregs
    qreg = state.qregs

    state.measure(qreg[0],creg[0])
    state.measure(qreg[1],creg[1])
    state.measure(qreg[2],creg[2])

    sim = AerSimulator()
    qc_comp = transpile(qc,sim)
    res = sim.run(qc_comp,shots = shots).result()
    counts = res.get_counts(0)

    select = (n+1)*'0'
    total = 0
    z = np.zeros(N)
    for key in counts:
        L = key.split()
        if L[1]+L[2] == select:
            z[int(L[0],2)] = np.sqrt(counts[key]/shots)
            total += counts[key]
    succesRate = total/shots
    print ('Succes rate: '+ str(succesRate))

    imin = N/4
    imax = 3*N/4
    sum = 0

    for i in range (imin, imax):
        sum += z[i] * 4/N*1/succesRate

    return sum