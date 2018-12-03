import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as ssl
import scipy.sparse.linalg
import timeit
import sys
from Parser import *

class Simulator(object):
    # TODO: there's a bug that the pwl current must start with 0
    """otherwise, the results will have large error with Hspice"""
    def __init__(self, netlist):
        self._Ii = np.empty(())
        self._Result = np.empty(())
        self._parser = None
        self._filename = netlist
        self._parser = SpiceParser(self._filename)

    def buildAg(self):
        # size of adjacent matrix is decided by m = num of comps, n = num of nodes
        m = len(self._parser._Resistors)
        n = len(self._parser._NodeMap)
        Ag = np.zeros((m, n))
        diag = []
        index = 0
        for res in self._parser._Resistors:
            name = res._name
            Np = res._Np
            Nn = res._Nn
            val = 1 / res._val
            diag.append(val)
            if Np not in ['GND', 'gnd', '0']:
                Ag[index, self._parser._NodeMap[Np]] = 1
            if Nn not in ['GND', 'gnd', '0']:
                Ag[index, self._parser._NodeMap[Nn]] = -1
            index += 1
        G = np.diag(diag)
        return Ag, G

    def buildAc(self):
        m = len(self._parser._Capacitors)
        n = len(self._parser._NodeMap)
        Ac = np.zeros((m, n))
        diag = []
        index = 0
        for cap in self._parser._Capacitors:
            name = cap._name
            Np = cap._Np
            Nn = cap._Nn
            val = cap._val
            diag.append(val)
            if Np not in ['GND', 'gnd', '0']:
                Ac[index, self._parser._NodeMap[Np]] = 1
            if Nn not in ['GND', 'gnd', '0']:
                Ac[index, self._parser._NodeMap[Nn]] = -1
            index += 1
        C = np.diag(diag)
        return Ac, C

    def buildAl(self):
        m = len(self._parser._Inductors)
        n = len(self._parser._NodeMap)
        Al = np.zeros((m, n))
        diag = []
        index = 0
        for ind in self._parser._Inductors:
            name = ind._name
            Np = ind._Np
            Nn = ind._Nn
            val = 1 / ind._val # h / L
            diag.append(val)
            if Np not in ['GND', 'gnd', '0']:
                Al[index, self._parser._NodeMap[Np]] = 1
            if Nn not in ['GND', 'gnd', '0']:
                Al[index, self._parser._NodeMap[Nn]] = -1
            index += 1
        L = np.diag(diag)
        return Al, L

    def buildAi(self):
        m = len(self._parser._CurrentSource)
        n = len(self._parser._NodeMap)
        length = len(self._parser._steps)
        Ai = np.zeros((m, n))
        Ii = np.zeros((m, length))
        index = 0
        for csrc in self._parser._CurrentSource:
            name = csrc._name
            Np = csrc._Np
            Nn = csrc._Nn
            val = csrc._val
            Ii[index] = val
            if Np not in ['GND', 'gnd', '0']:
                Ai[index, self._parser._NodeMap[Np]] = 1
            if Nn not in ['GND', 'gnd', '0']:
                Ai[index, self._parser._NodeMap[Nn]] = -1
            index += 1
        self._Ii = Ii
        return Ai, Ii

    def createMatrix(self, adj, diag):
        mat = np.dot(adj.T, diag)
        return np.dot(mat, adj)

    def createVector(self, adj, vector):
        mat = np.dot(adj.T, vector)
        return mat

    def buildDcCircuit(self):
        Ag, diagG = self.buildAg()
        Ac, diagC = self.buildAc()
        Al, diagL = self.buildAl()
        # m, n = Al.shape
        # diagL = np.diag([1 for i in range(m)])
        Ai, vecI = self.buildAi()
        vecI = vecI[:, 0] # take the initial value
        G = self.createMatrix(Ag, diagG)
        # print(G)
        # sys.exit()
        L = self.createMatrix(Al, diagL)
        Ii = self.createVector(Ai, vecI)
        return G + L, Ii

    def buildTransientCircuit(self):
        Ag, diagG = self.buildAg()
        Ac, diagC = self.buildAc()
        Al, diagL = self.buildAl()
        Ai, vecI = self.buildAi()
        G = self.createMatrix(Ag, diagG)

        C = self.createMatrix(Ac, diagC)
        L = self.createMatrix(Al, diagL)
        Ii = self.createVector(Ai, vecI)
        h = self._parser._delta
        # solving Av(t+delta) = Bv(t) + I
        A = G + (2 * C / h) + (h * L / 2)
        B = -G + (2 * C / h) - (h * L / 2)
        S = h * L
        return A, B, S, Ii

    def simulate(self):
        # self.solveDC()
        start = timeit.default_timer()
        # self.solveDC()
        self.solveTransient()
        stop = timeit.default_timer()
        print('total simulation time: %.5fs' %(stop - start))

    def solveDC(self):
        n = len(self._parser._Nodes) # number of node
        vn = np.ones((n, ))
        A, Ii = self.buildDcCircuit()
        x = np.linalg.solve(A, Ii)
        # x = scipy.sparse.linalg.cg(A, Ii)[0]
        return x

    def solveTransient(self):
        n = len(self._parser._Nodes) # number of node
        t = len(self._parser._steps) # number of steps
        vn = np.zeros((n, t))
        vn[:, 0] = self.solveDC()
        A, B, S, Ii = self.buildTransientCircuit()
        lu, piv = ssl.lu_factor(A)
        il = np.zeros((n, ))
        for step in range(1, t):
            iv = np.dot(B, vn[:, step - 1]) # first part of the right hand side
            ii = Ii[:, step] + Ii[:, step - 1] # second part of the right hand side
            rhs = iv + ii - il
            # TODO: somehow the lu solve is slower with small circuit
            vn[:, step] = ssl.lu_solve((lu, piv), rhs)
            # vn[:, step] = np.linalg.solve(A, rhs)
            # vn[:, step] = scipy.sparse.linalg.cg(A, rhs)[0] # solving by conjugate gradient
            il = il + np.dot(S, vn[:, step] + vn[:, step - 1]) # update current of inductors
            # sys.exit()
        self._Result = vn

    def plot(self, node):
        waveform = self._Result[self._parser._NodeMap[node], :]
        maximal, minimal = np.max(waveform), np.min(waveform)
        maxNoise = max(maximal - 1, 1 - minimal)
        p2pNoise = maximal - minimal
        print('max noise: %.3fv\npeak-peak noise: %.3fv' %(maxNoise, maximal - minimal))
        plt.subplot(211)
        plt.plot(self._parser._steps, waveform)

    def plotCurrentSource(self, k):
        i = self._Ii[:-1,:]
        waveform = i.sum(axis=0)
        # waveform = self._Ii.sum(axis=0)
        plt.subplot(212)
        plt.plot(self._parser._steps, waveform)

    def poltNoiseMap(self):
        # TODO:
        pass

if __name__ == "__main__":
    simulator = Simulator('./pgNet.sp')
    simulator.simulate()
    simulator.plot('n14')
    simulator.plotCurrentSource(1)
    plt.show()
