import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import linprog

def createRect(name, cords):
    pass

class module:
    def __init__(self, name, width, height):
        self._width = width
        self._height = height
        self._area = width * height

# with open('./ami49.txt') as file:
#     fig = plt.figure(figsize=(12,9))
#     ax = fig.add_subplot(1,1,1)
#     ax.set_ylim([0,100])
#     ax.set_xlim([0,100])
#     modules = []
#     for line in file:
#         line = line.strip()
#         line = re.sub(';', '', line)
#         if(line.startswith('MODULE')):
#             name = line.split()[1]
#         if(line.startswith('DIMENSIONS')):
#             cords = line.split()[1:]
#             width = float(cords[0])
#             height = float(cords[3])
#             m = module(name, width, height)
#             modules.append(m)

def findWhiteSpaceVertex(chip, modules):
    whiteSpace = []
    for v in chip:
        whiteSpace.append(v)
    for mdl in modules:
        vertex = [mdl[0], mdl[1], (mdl[0][0], mdl[1][1]), (mdl[1][0], mdl[0][1])]
        for v in vertex:
            if(v not in whiteSpace):
                whiteSpace.append(v)
            else:
                whiteSpace.remove(v)

    whiteSpace = sorted(whiteSpace, key = lambda x: (x[1], -x[0]), reverse = True) # sort by cordinates
    return whiteSpace

def findNearest(curr, vertexes, d):
    pass

def buildGraph(module, whiteSpace):
    m = len(module)
    n = len(whiteSpace)
    g = np.zeros((n, m))
    for i in range(m):
        mdl = module[i]
        for j in range(n):
            ws = whiteSpace[j]
            if(ws[0][0] == mdl[1][0] or ws[1][0] == mdl[0][0]):
                if((ws[0][1] >= mdl[1][1] and ws[0][1] <= mdl[0][1]) or (ws[1][1] >= mdl[1][1] and ws[1][1] <= mdl[0][1])):
                    g[j][i] = 1
            elif(ws[0][1] == mdl[1][1] or ws[1][1] == mdl[0][1]):
                if((ws[0][0] >= mdl[0][0] and ws[0][0] <= mdl[1][0]) or (ws[1][0] >= mdl[0][0] and ws[1][0] <= mdl[1][0])):
                    g[j][i] = 1
    return g

def getArea(whiteSpace):
    a = []
    for ws in whiteSpace:
        width = ws[1][0] - ws[0][0]
        height = ws[0][1] - ws[1][1]
        area = width * height
        a.append(area)
    return np.array(a)

def getCapReq(module):
    r = []
    for mdl in module:
        r.append(mdl[2] / 0.01)
    return np.array(r)

def buildConstrain(connection, area, req, m, n):
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

def updateLayout(module, demand):
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
        print('insertion demand =', sum([dmd[x] for x in levelY[i]]), 'vertucally moving by:',Bws)
        for j in range(i+1, len(levelY)):
            l = levelY[j]
            for n in l:
                module[n][0] = (module[n][0][0], module[n][0][1] + Bws)
                module[n][1] = (module[n][1][0], module[n][1][1] + Bws)
    for i in range(len(levelX)):
        Bws = sum([dmd[x] for x in levelX[i]]) * (1-alpha) / newLayoutY
        print('insertion demand =', sum([dmd[x] for x in levelX[i]]), 'horizaontally moving by:',Bws)
        for j in range(i+1, len(levelX)):
            l = levelX[j]
            for n in l:
                module[n][0] = (module[n][0][0] + Bws, module[n][0][1])
                module[n][1] = (module[n][1][0] + Bws, module[n][1][1])
    return newLayoutX, newLayoutY

def plotLayout(ax, rects, whiteSpace = []):
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

def levelization(module, direction = 'v'):
    level = [[]]
    result = [[]]
    index = 0
    if(direction == 'v'):
        mdl = sorted(module, key = lambda x: (x[0][1], x[0][0]))
        for i in range(len(mdl)):
            curr = mdl[i]
            for m in level[index]:
                if(curr[0][0] >= m[0][0] and curr[0][0] <= m[1][0] and curr[1][1] >= m[0][1]):
                    level.append([])
                    result.append([])
                    index += 1
                    break
            level[index].append(curr)
            result[index].append(module.index(curr))
    else:
        mdl = sorted(module, key = lambda x: (x[0][0], x[0][1]))
        for i in range(len(mdl)):
            curr = mdl[i]
            for m in level[index]:
                if(curr[0][1] >= m[1][1] and curr[0][1] <= m[0][1] and curr[0][0] >= m[1][0]):
                    level.append([])
                    result.append([])
                    index += 1
                    break
            level[index].append(curr)
            result[index].append(module.index(curr))
    return result

cox = 0.01
alpha = 0.5 # alpha portion of the additional whiteSpace is obtained by extending floorplan in y-direction
chip = [(0,0), (0,100), (100,0), (100,100)]
layoutX, layoutY = chip[2][0] - chip[0][0], chip[1][1] - chip[0][1]
# module as [cordinates, charge, module name]
module = [[(0,25),(50,0),10,'1'], [(50,60),(70,0),5,'2'], [(70,60),(100,40),15,'3'], [(0,100),(40,25),1,'4'], [(40,100),(100,70),2,'5']]
whiteSpace = [[(40,70),(50,25)], [(50,70),(100,60)], [(70,40),(100,0)]]

connection = buildGraph(module, whiteSpace) # adjacent matrix
area = getArea(whiteSpace)
req = getCapReq(module)
m = len(whiteSpace)
n = len(module)
A, B = buildConstrain(connection, area, req, m, n)
C = np.reshape(-connection, (1, m*n))[0]
res = linprog(C, A_ub=A, b_ub=B, options={"disp": True})
# print(res.x.reshape((m,n)))
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,2,1)
ax.set_ylim([0,layoutY])
ax.set_xlim([0,layoutX])
plotLayout(ax, module)

dmd = res.slack[m:n+m+1]
newLayoutX, newLayoutY = updateLayout(module, dmd)

axx = fig.add_subplot(1,2,2)
axx.set_ylim([0,newLayoutY])
axx.set_xlim([0,newLayoutX])
plotLayout(axx, module)

plt.show()
