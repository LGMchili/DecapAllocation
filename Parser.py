import numpy as np
import matplotlib.pyplot as plt
class Component(object):
    def __init__(self, name, Np, Nn, val):
        self._name = name
        self._Np = Np
        self._Nn = Nn
        self._val = val

class SpiceParser:
    def __init__(self, filename=None):
        self._Resistors = []
        self._Capacitors = []
        self._Inductors = []
        self._CurrentSource = []
        self._VoltageSource = []
        self._Nodes = []
        self._NodeMap = {}
        self._steps = np.linspace(0,1,1)
        self._delta = 0.1
        self.Parse(filename)
        self.initCurrentSource()

    def Parse(self, filename):
        # lines: list[string] which are the lines of the file to be parsed
        # returns the Circuit object
        #
        # NOTE: Spice traditianally ignored the first line,
        #       but I hate this and simply *will not* do this.
        with open(filename, 'r') as content:
            for line in content:
                line = line.rstrip()
                line  = line.lower()
                words = line.split()
                if not words or len(words) < 4 or words[0].startswith(('*', '#')):
                    # the option probe is ignore here
                    continue
                else:
                    self.ParseLine(words)


    def ParseLine(self, words):
        if words[0].startswith('.'):
            # .tran startTime endTime delta
            if(words[0].startswith('.tran')):
                self.addDirective(words)
            else:
                print(words)
        elif words[0].startswith('.vdd'):
            self._Vdd = float(words[1])
        else:
            self.addComponent(words)

    def addDirective(self, words):
        start = float(words[1])
        end = float(words[2])
        delta = float(words[3])
        steps = int((end - start) / delta) + 1
        self._steps = np.linspace(start, end, steps)
        self._delta = delta

    def addComponent(self, words):
        name = words[0]
        Np = words[1]
        Nn = words[2]
        if Np not in ['GND', 'gnd', '0'] and Np not in self._Nodes:
            n = len(self._NodeMap)
            self._NodeMap[Np] = n
            self._Nodes.append(Np)
            # print("add node %s as %d" %(Np,n))
        if Nn not in ['GND', 'gnd', '0'] and Nn not in self._Nodes:
            n = len(self._NodeMap)
            self._NodeMap[Nn] = n
            self._Nodes.append(Nn)
            # print("add node %s as %d" %(Np,n))

        if name.startswith('i'):
            self.addCurrentSource(name, Np, Nn, words[3:])
        else:
            val = float(words[3])
            if name.startswith('r'):
                self.addResistor(name, Np, Nn, val)
            elif name.startswith('c'):
                self.addCapacitor(name, Np, Nn, val)
            elif name.startswith('l'):
                self.addInductor(name, Np, Nn, val)
            elif name.startswith('v'):
                self.addVoltageSource(name, Np, Nn, val)

    def addResistor(self, name, Np, Nn, val):
        assert val > 0
        res = Component(name, Np, Nn, val)
        self._Resistors.append(res)

    def addCapacitor(self, name, Np, Nn, val):
        assert val > 0
        cap = Component(name, Np, Nn, val)
        self._Capacitors.append(cap)

    def addInductor(self, name, Np, Nn, val):
        assert val > 0
        ind = Component(name, Np, Nn, val)
        self._Inductors.append(ind)

    def addCurrentSource(self, name, Np, Nn, params):
        # waveform = [-val for i in range(len(self._steps))]
        csrc = Component(name, Np, Nn, params)
        self._CurrentSource.append(csrc)

    def addVoltageSource(self, name, Np, Nn, val):
        val = float(val)
        vsrc = Component(name, Np, Nn, val)
        self._VoltageSource.append(vsrc)

    def initCurrentSource(self):
        if(len(self._steps) > 1):
            # transient analysis
            for i in range(len(self._CurrentSource)):
                if (self._CurrentSource[i]._val[0] == 'pwl'):
                    # pwl source
                    # time, value, time, value, ..., ...
                    params = [float(x) for x in self._CurrentSource[i]._val[1:]]
                    times = params[0::2] # indexing by [start:end:step]
                    values = params[1::2]
                    values = [-x for x in values]
                    waveform = np.interp(self._steps, times, values)
                    # uncomment to debug
                    # plt.plot(self._steps, waveform)
                    # plt.show()
                else:
                    # dc source
                    val = -float(self._CurrentSource[i]._val[0])
                    waveform = np.interp(self._steps, [0], [val])
                self._CurrentSource[i]._val = waveform
        else:
            # dc analysis
            # current source initialize as single point value
            for i in range(len(self._CurrentSource)):
                val = -float(self._CurrentSource[i]._val[0])
                self._CurrentSource[i]._val = val

if __name__ == "__main__":
    parser = SpiceParser('./testCir.sp')
