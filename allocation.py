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
