import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import linprog

def layoutDictToList(design, load):
    module = []
    whiteSpace = []
    for key in design.keys():
        if(key == 'ws'):
            for ws in design[key]:
                width, height, offset_x, offset_y = ws
                whiteSpace.append([(offset_x, offset_y), (offset_x + width, offset_y + height), 0, key])
        else:
            # original format in [width, height, offset_x, offset_y]
            width, height, offset_x, offset_y = design[key]
            # format to [bottom left corner, top right corner, load current, module name]
            module.append([(offset_x, offset_y), (offset_x + width, offset_y + height), load[key], key])
    return module, whiteSpace

def buildGraph(module, whiteSpace):
    m = len(module)
    n = len(whiteSpace)
    graph = np.zeros((n, m))
    for i in range(m):
        mdl = module[i]
        for j in range(n):
            ws = whiteSpace[j]
            if(ws[0][0] == mdl[1][0] or ws[1][0] == mdl[0][0]):
                if((ws[0][1] >= mdl[0][1] and ws[0][1] <= mdl[1][1]) or (ws[1][1] >= mdl[0][1] and ws[1][1] <= mdl[1][1])):
                    graph[j][i] = 1
            elif(ws[0][1] == mdl[1][1] or ws[1][1] == mdl[0][1]):
                if((ws[0][0] >= mdl[0][0] and ws[0][0] <= mdl[1][0]) or (ws[1][0] >= mdl[0][0] and ws[1][0] <= mdl[1][0])):
                    graph[j][i] = 1
    return graph

def levelization(module, direction = 'v'):
    level = [[]]
    result = [[]]
    index = 0
    if(direction == 'v'):
        mdl = sorted(module, key = lambda x: (x[0][1], x[0][0]))
        for i in range(len(mdl)):
            curr = mdl[i]
            for m in level[index]:
                if(curr[0][0] >= m[0][0] and curr[0][0] <= m[1][0] and curr[0][1] >= m[1][1]):
                    level.append([])
                    result.append([])
                    index += 1
                    break
            level[index].append(curr)
            # result[index].append(curr[3])
            result[index].append(module.index(curr))
    else:
        mdl = sorted(module, key = lambda x: (x[0][0], x[0][1]))
        for i in range(len(mdl)):
            curr = mdl[i]
            for m in level[index]:
                if(curr[0][1] >= m[0][1] and curr[0][1] <= m[1][1] and curr[0][0] >= m[1][0]):
                    level.append([])
                    result.append([])
                    index += 1
                    break
            level[index].append(curr)
            # result[index].append(curr[3])
            result[index].append(module.index(curr))
    return result

def getArea(whiteSpace):
    a = []
    for ws in whiteSpace:
        width = ws[1][0] - ws[0][0]
        height = ws[1][1] - ws[0][1]
        area = width * height
        a.append(area)
    return np.array(a)

def getCapReq(module, cox):
    r = []
    # TODO: calculate the decap budget
    for mdl in module:
        r.append(mdl[2] / cox)
    return np.array(r)

def buildConstrain(connection, area, req, m, n):
    # each line of constrain represents the allocation of each whiteSpace
    A = np.empty((m, m*n))
    for i in range(m):
        left = np.zeros((i*n))
        right = np.zeros(((m-i-1)*n))
        if(left.shape): a = np.hstack((left, connection[i]))
        if(right.shape): a = np.hstack((a, right))
        A[i] = a
    for i in range(n):
        temp = np.zeros((m, n))
        temp[:, i] = 1
        temp = np.reshape(temp, (1, m*n))
        A = np.vstack((A, temp))
    b1 = area
    b2 = req
    B = np.hstack((b1, b2))
    return A, B

def plotLayout(ax, module, whiteSpace = []):
    for rect in module:
        width = rect[1][0] - rect[0][0]
        height = rect[1][1] - rect[0][1]
        r = patches.Rectangle((rect[0][0],rect[0][1]),width,height,linewidth=1,edgecolor='b',facecolor='r')
        ax.add_patch(r)
    for ws in whiteSpace:
        width = ws[1][0] - ws[0][0]
        height = ws[1][1] - ws[0][1]
        w = patches.Rectangle((ws[0][0],ws[0][1]),width,height,linewidth=1,edgecolor='b',facecolor='y')
        ax.add_patch(w)

def updateLayout(module, demand, alpha, layoutX, layoutY, dmd):
    # levelY = [[0,1,2], [3,4]]
    # levelX = [[0,3], [1,4], [2]]
    levelY  = levelization(module, 'v')
    levelX = levelization(module, 'h')
    totalDmd = np.sum(demand)
    extY = alpha * totalDmd / layoutX
    extX = (1 - alpha)*totalDmd / (layoutY + extY)
    newLayoutX = layoutX + extX
    newLayoutY = layoutY + extY
    for i in range(len(levelY)):
        Bws = sum([dmd[x] for x in levelY[i]]) * alpha / layoutX
        print('insertion demand:', sum([dmd[x] for x in levelY[i]]), 'at level:', i, 'by moving:', Bws, 'in Y-direction')
        for j in range(i+1, len(levelY)):
            l = levelY[j]
            for n in l:
                module[n][0] = (module[n][0][0], module[n][0][1] + Bws)
                module[n][1] = (module[n][1][0], module[n][1][1] + Bws)
    for i in range(len(levelX)):
        Bws = sum([dmd[x] for x in levelX[i]]) * (1-alpha) / newLayoutY
        print('insertion demand:', sum([dmd[x] for x in levelX[i]]), 'at level:', i, 'by moving:', Bws, 'in X-direction')
        for j in range(i+1, len(levelX)):
            l = levelX[j]
            for n in l:
                module[n][0] = (module[n][0][0] + Bws, module[n][0][1])
                module[n][1] = (module[n][1][0] + Bws, module[n][1][1])
    return newLayoutX, newLayoutY

def getPgDimension(pwrPin):
    """
    get the rectangle size of pg net
    """
    pins = []
    for ps in pwrPin.values():
        for p in ps:
            pins.append(p)
    minHeight, minWidth = float('inf'), float('inf')
    pgX, pgY = pins[0][0], pins[0][1]
    for i in range(1, len(pins)):
        if(pins[i-1][1] != pins[i][1]):
            minHeight = min(minHeight, abs(pins[i-1][1] - pins[i][1]))
            pgY = max(pgY, pins[i][1])
        if(pins[i-1][0] != pins[i][0]):
            minWidth = min(minWidth, abs(pins[i-1][0] - pins[i][0]))
            pgX = max(pgX, pins[i][0])
    return minWidth, minHeight, pgX, pgY

def mergePin(pwrPin):
    pins = {}
    result = []
    for ps in pwrPin.values():
        for p in ps:
            if(p[0] in pins):
                if(p[1] in pins[p[0]]):
                    pins[p[0]][p[1]] += p[2]
                else:
                    pins[p[0]][p[1]] = p[2]
            else:
                pins[p[0]] = {p[1]:p[2]}
    for k1 in pins.keys():
        for k2 in pins[k1].keys():
            result.append([k1, k2, pins[k1][k2]])
    return result

def addCurrentSource(file, pwrPin, nodeNumX, nodeNumY, dimension):
    layoutX, layoutY = dimension[0], dimension[1]
    pitchX, pitchY = layoutX / nodeNumX, layoutY / nodeNumY
    num = 0
    # pins = mergePin(pwrPin)
    # print(layoutX, layoutY)
    file.write('*load\n')
    decapBudget = getCapBudget(pwrPin)
    for key in pwrPin.keys():
        ps = pwrPin[key]
        caps = decapBudget[key]
        # TODO: get decap demand for each module here
        for i in range(len(ps)):
            x, y  = int(ps[i][0] // pitchX), int((ps[i][1] // pitchY))
            c = caps[i]
            # print(x, y, ',', p[0], p[1])
            node = 'n' + str(x * nodeNumX + y)
            curr = 'i' + str(num) + ' ' + node + ' ' + 'gnd' + ' ' + 'pwl ' + \
            '.35e-6 0 .4e-6 ' + str(ps[i][2]) + ' .45e-6 0'
            cap = 'c' + str(num) + ' ' + node + ' ' + 'gnd' + ' '  + str(c)
            # x = 'n' + str(p[0] // minWidth)
            # y = 'n' + str(p[1] // minHeight)
            file.write(curr + '\n')
            file.write(cap + '\n')
            num += 1

def addVoltageSource(file):
    file.write('*pdn\n')
    pdn = 'iin nn0 gnd -1e3\nRp0 nn0 gnd 1e-3\nRp1 nn0 nn01 10e-3\nLp1 nn01 nn02 1e-9\nCp1 nn02 0 1000e-6\nRp2 nn0 nn1 1e-3\nLp2 nn1 nn2 10e-9\n' + \
            'Rp3 nn2 nn21 12e-3\nLp3 nn21 nn22 500e-12\nCp3 nn22 0 25e-6\nRp4 nn2 nn3 1e-3\nLp4 nn3 nn4 200e-12\n' + \
            'Rp5 nn4 nn41 15e-3\nLp5 nn41 nn42 100e-12\nCp5 nn42 0 500e-9\nRp6 nn4 nn5 1e-3\nLp6 nn5 nn6 30e-9\n' + \
            'Rp7 nn6 n1 5e-3\nCp7 n1 0 1e-9\n'
    file.write(pdn)
    file.write('*voltage source\n')
    # formation in {pad:[voltage, resistance, inductance]}
    # vsrc = {'n1':[1, 1.25, 1e-9]}
    vsrc = {}
    i = 0
    for k in vsrc.keys():
        v, r, l = vsrc[k]
        index = str(i)
        lines = 'iv_' + index + ' x_' + index + ' gnd ' + str(-v/r) + '\n' \
                + 'r_pad_' + index + ' x_' + index + ' gnd ' + str(r) + '\n' \
                + 'l_pad_' + index + ' x_' + index + ' ' + k + ' ' + str(l) + '\n'
        file.write(lines)

def netlistGenerator(pwrPin, nodeNumX, nodeNumY, dimension):
    # minWidth, minHeight, pgX, pgY = getPgDimension(pwrPin)
    # nodeNumX, nodeNumY = int(pgX // minWidth), int(pgY // minHeight)
    with open('pgNet.sp', 'w') as file:
        # horizontal
        for i in range(nodeNumY):
            y = i * nodeNumX + 1
            for j in range(nodeNumX - 1):
                x = i * (nodeNumX - 1) + j + 1
                comp = ''
                comp += 'r' + str(x) + ' ' # type
                comp += 'n' + str(y) + ' ' + 'n' + str(y + 1) + ' '# num
                comp += '0.25' # value
                y += 1
                file.write(comp + '\n')
        # vertical
        for i in range(nodeNumY - 1):
            y = i * nodeNumX + 1
            for j in range(nodeNumX):
                x = i * nodeNumX + j + 1 + nodeNumY * (nodeNumX - 1)
                comp = ''
                comp += 'r' + str(x) + ' ' # type
                comp += 'n' + str(y) + ' ' + 'n' + str(y + nodeNumX) + ' '# num
                comp += '0.25'# value
                y += 1
                file.write(comp + '\n')
        addCurrentSource(file, pwrPin, nodeNumX, nodeNumY, dimension)
        addVoltageSource(file)
        # simulation options
        file.write('*simulation options\n')
        file.write('.tran 0 2e-6 0.001e-6\n')
        file.write('.probe n1\n')

def getCapBudget(pwrPin):
    cycleTime = 10e-6
    tox = 200e-6 # thickness of dielectric
    dielectric = 3.9
    cox = dielectric / tox
    vlim = 10 # maximal supply noise
    # theta = max(1, v[k] / vlim)
    decapBudget = {}
    for k in pwrPin.keys():
        pins = pwrPin[k]
        # the assumed cycle time is 20us, so the width of triangle waveform is 10us
        decapBudget[k] = []
        for p in pins:
            current = p[2]
            charge = cycleTime * current / 2
            budget = charge / vlim
            decapBudget[k].append(budget)
    return decapBudget

def addDecap(design, load, dimension):
    np.set_printoptions(threshold=np.nan)
    layoutX, layoutY = dimension
    Cox = 0.0000002 # thickness of oxide
    alpha = 0.5 # alpha portion of the additional whiteSpace is obtained by extending floorplan in y-direction
    module, whiteSpace = layoutDictToList(design, load)
    graph = buildGraph(module, whiteSpace)
    area = getArea(whiteSpace)
    req = getCapReq(module, Cox)
    m = len(whiteSpace)
    n = len(module)
    A, B = buildConstrain(graph, area, req, m, n)
    C = np.reshape(-graph, (1, m*n))[0]
    res = linprog(C, A_ub=A, b_ub=B, options={"disp": True})
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,2,1)
    ax.set_xlim([0,layoutX])
    ax.set_ylim([0,layoutY])
    plotLayout(ax, module)
    # print(res.x.reshape((m,n)))
    # print(req)
    dmd = res.slack[m:n+m+1]
    print(dmd)
    newLayoutX, newLayoutY = updateLayout(module, dmd, alpha, layoutX, layoutY, dmd)
    axx = fig.add_subplot(1,2,2)
    axx.set_ylim([0,newLayoutY])
    axx.set_xlim([0,newLayoutX])
    plotLayout(axx, module)
    plt.show()
