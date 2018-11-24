import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math

def rotateRects(module):
    # add rotated rects to list
    rects = module.copy()
    for k in rects.keys():
        width, height = rects[k][0], rects[k][1]
        rects[k] = [rects[k]]
        if(width != height):
            rects[k].append([height, width]);
    return rects

def removeRedundant(l):
    result = []
    for i in range(len(l)):
        result.append(l[i])
        for j in range(len(l)):
            if(i != j and l[i][0] >= l[j][0] and l[i][1] >= l[j][1]):
                result.pop()
                break
    return result
#
# def packing(r1, r2, d):
#     if(d == 'H'):
#
#     else:

def rotateToMin(seq, module):
    rects = module.copy()
    for k in rects.keys():
        width, height = rects[k][0], rects[k][1]
        rects[k].append([k])
        rects[k] = [rects[k]]
        if(width != height):
            rects[k].append([height, width, [k+'_']]);
    stack = []
    for m in seq:
        if(m == 'H' or m == 'V'):
            operator = m
            temp = []
            m1 = stack.pop()
            m2 = stack.pop()
            # print(m1, m2)
            for r1 in rects[m1]:
                for r2 in rects[m2]:
                    w1, h1, w2, h2 = r1[0], r1[1], r2[0], r2[1]
                    if(m == 'H'):
                        temp.append([max(w1, w2), h1 + h2, r1[2] + r2[2]])
                    elif(m == 'V'):
                        temp.append([w1 + w2, max(h1, h2), r1[2] + r2[2]])
            temp = removeRedundant(temp)
            stack.append(m1+m2)
            rects[m1+m2] = temp
        else:
            stack.append(m)
    result = min(temp, key=lambda x: x[0] * x[1])[2]
    # print(result)
    for r in result:
        if(r.endswith('_')):
            name = r[:-1]
            module[name][0], module[name][1] = module[name][1], module[name][0]

def plotLayout(seq, module):
    rects = module.copy()
    rotateToMin(seq, rects)
    stack = []
    pack = {}
    whiteSpace = []
    for k in rects.keys():
        # format in [width, height, x, y]
        pack[k] = [[rects[k][0], rects[k][1], 0, 0, k]]
    for m in seq:
        if(m == 'H' or m == 'V'):
            operator = m
            m1 = stack.pop()
            m2 = stack.pop()
            w1, h1, w2, h2 = rects[m1][0], rects[m1][1], rects[m2][0], rects[m2][1]
            if(m1+m2 not in pack): pack[m1+m2] = []
            offset_x = rects[m2][0]
            offset_y = rects[m2][1]
            if(m == 'H'):
                rects[m1+m2] = [max(w1, w2), h1 + h2]
                for mm in pack[m1]: # right part
                    mm[3] += offset_y
                    pack[m1+m2].append(mm)
                for mm in pack[m2]: # left part
                    pack[m1+m2].append(mm)
                # whiteSpace
                if(w1 < w2): pack[m1+m2].append([w2-w1, h1, w1, offset_y, 'ws'])
                elif(w2 < w1): pack[m1+m2].append([w1-w2, h2, w2, 0, 'ws']) # whiteSpace at the bottom part, thus don't need to add offset_y
            elif(m == 'V'):
                rects[m1+m2] = (w1 + w2, max(h1, h2))
                for mm in pack[m1]: # right part
                    mm[2] += offset_x
                    pack[m1+m2].append(mm)
                for mm in pack[m2]: # left part
                    pack[m1+m2].append(mm)
                if(h1 < h2): pack[m1+m2].append([w1, h2-h1, offset_x, h1, 'ws'])
                elif(h2 < h1): pack[m1+m2].append([w2, h1-h2, 0, h2, 'ws']) # whiteSpace at the left part, thus don't need to add offset_x
            stack.append(m1+m2)
            # print(rects[m2], rects[m1], operator)
        else:
            stack.append(m)
    result = pack[m1+m2]
    chip_x, chip_y = rects[m1+m2]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_ylim([0,chip_y])
    ax.set_xlim([0,chip_x])
    for rr in result:
        if(rr[4] == 'ws'):
            # whiteSpace
            w = patches.Rectangle((rr[2],rr[3]),rr[0],rr[1],linewidth=1,edgecolor='black',facecolor='grey',label='ws')
        else:
            w = patches.Rectangle((rr[2],rr[3]),rr[0],rr[1],linewidth=1,edgecolor='black',facecolor='r')
        # ax.text(rr[2]+0.5*rr[0], rr[3]+0.5*rr[1], rr[4], horizontalalignment="center", verticalalignment='center')
        ax.add_patch(w)

def calculateCost(seq, module):
    rects = rotateRects(module)
    stack = []
    for m in seq:
        if(m == 'H' or m == 'V'):
            operator = m
            temp = []
            m1 = stack.pop()
            m2 = stack.pop()
            # print(m1, m2)
            for r1 in rects[m1]:
                for r2 in rects[m2]:
                    w1, h1, w2, h2 = r1[0], r1[1], r2[0], r2[1]
                    if(m == 'H'):
                        temp.append([max(w1, w2), h1 + h2])
                    elif(m == 'V'):
                        temp.append([w1 + w2, max(h1, h2)])
            temp = removeRedundant(temp)
            stack.append(m1+m2)
            rects[m1+m2] = temp
        else:
            stack.append(m)
    rect = min(temp, key=lambda x: x[0] * x[1])
    return rect[0] * rect[1]

def getPossibleMove(E):
    pm_op1, pm_op3 = [], []
    for i in range(len(E) - 1):
        if(E[i].isdigit() and E[i+1].isdigit()):
            pm_op1.append(i)
        if(E[i].isdigit() and not E[i+1].isdigit() and 2*getBallot(E, i+1) < i and E[i+1] != E[i-1]):
            pm_op3.append(i)
        elif(not E[i].isdigit() and E[i+1].isdigit()):
            if(i == len(E) - 2 or E[i] != E[i+2]):
                pm_op3.append(i)
    return pm_op1, pm_op3

def op1(E, pm):
    e = E.copy()
    k = random.randint(0, len(pm) - 1)
    i = pm[k]
    e[i], e[i+1] = e[i+1], e[i]
    # for i in range(len(e)):
    #     if(e[i].isdigit() and e[i+1].isdigit()):
    #         e[i], e[i+1] = e[i+1], e[i]
    return e

def op2(E):
    e = E.copy()
    chain = getChain(e)
    dice = random.randint(0, len(chain) - 1)
    index = chain[dice]
    for k in index:
        if(e[k] == 'H'):
            e[k] = 'V'
        elif(e[k] == 'V'):
            e[k] = 'H'
    return e

def op3(E, pm):
    if(not pm):
        return E
    done = False
    e = E.copy()
    k = random.randint(0, len(pm) - 1)
    i = pm[k]
    e[i], e[i+1] = e[i+1], e[i]
    # for i in range(len(e) - 1):
    #     if(e[i].isdigit() and not e[i+1].isdigit() and 2*getBallot(E, i+1) < i and e[i+1] != e[i-1]):
    #         done = True
    #         e[i], e[i+1] = e[i+1], e[i]
    #         break
    #     elif(not e[i].isdigit() and e[i+1].isdigit()):
    #         if(i == len(e) - 2 or e[i] != e[i+2]):
    #             done = True
    #             e[i], e[i+1] = e[i+1], e[i]
    #             break
    # print(''.join(E))
    # print(''.join(e))
    return e

def getBallot(E, i):
    count = 0
    for j in range(i+1):
        if(not E[j].isdigit()):
            count += 1
    return count

def getChain(E):
    e = E.copy()
    chain, temp = [], []
    for i in range(len(e)):
        if(not e[i].isdigit()):
            temp.append(i)
            if(i == len(e) - 1 or e[i+1].isdigit()):
                chain.append(temp)
                temp = []
    return chain

def getNeighborhoodStructure(E, dice):
    pm_op1, pm_op3 = getPossibleMove(E)
    if(dice == 1):
        # op1
        newE = op1(E, pm_op1)
        # print(1, E, newE)
    elif(dice == 2):
        # op2
        newE = op2(E)
        # print(2, E, newE)
    else:
        # op3
        newE = op3(E, pm_op3)
        # print(3, E, newE)

    # print(pm_op1, pm_op3, 'dice:', dice)
    # print(''.join(newE))

    # E1, E2, E3 = op1(E), op2(E, chain), op3(E)
    # cost = [calculateCost(E1, modules), calculateCost(E2, modules), calculateCost(E3, modules)]
    # # print(cost)
    # result = [E1, E2, E3]
    # index = cost.index(min(cost))
    # newE = result[index]
    return newE

def genInitialSolution(module):
    E = []
    i = 0
    # initial solution
    keys = [*module.keys()]
    random.shuffle(keys)
    for k in keys:
        E.append(k)
        if(i >= 1):
            E.append('V')
            # if(random.uniform(0, 1) > 0.5):
            #     E.append('V')
            # else:
            #     E.append('H')
        i += 1
    return E

def sa(module):
    E = genInitialSolution(module)
    E = decode('603213H4053V30H584218VHV5733H55HV395150HV2937V5447VHVHV5634HV3112HV1943V38V3620V2252VH59V2146V2815VHVH49172326HV4124VHV27V48V11HVH142516H4535V44HVHVH')
    result = []
    l = len(module)
    Ebest = E
    best_cost = calculateCost(E, module)
    iter = 0
    T = 1000000000 # initial temperature
    r = 0.85 # reduce ratio
    epsilon = 1 # minimal temperature
    max_iteration = 5*l # max iterator at each time
    reject = 0
    result.append(best_cost)
    # solving
    while(reject / max_iteration < 0.95 and T > epsilon):
        reject = 0
        for iter in range(max_iteration):
            dice = random.randint(1, 3)
            newE = getNeighborhoodStructure(E, dice)
            new_cost  = calculateCost(newE, module)
            delta_cost = new_cost - calculateCost(E, module)
            if(delta_cost <= 0 or random.uniform(0, 1) < math.exp(-delta_cost / T)):
                result.append(new_cost)
                E = newE
                if(new_cost < best_cost):
                    Ebest = newE
                    best_cost = new_cost
            else:
                reject += 1
        print('best cost:', best_cost, 'current cost:', new_cost, 'reject times:', reject, 'current temperature:', T)
        #     print('best cost:', best_cost, 'best solution:', Ebest,
        #      'current cost:', new_cost, 'current solution:', newE, 'reject times:', reject)
        T = r*T
    # end solving
    return Ebest, result

def mergeWhiteSpace():
    pass

def decode(seq):
    l = len(seq)
    result = []
    temp = ''
    i = 0
    for j in range(l):
        if(seq[j].isdigit()):
            temp += seq[j]
            i += 1
            if(i == 2):
                result.append(temp)
                i = 0
                temp = ''
        else:
            result.append(seq[j])
    return result
# fp = ['1','2','H','3','4','H','5','H','V']
if __name__ == '__main__':
    with open('./ami49.txt') as file:
        # fig = plt.figure(figsize=(12,9))
        # ax = fig.add_subplot(1,1,1)
        # ax.set_ylim([0,100])
        # ax.set_xlim([0,100])
        nameMap = {}
        modules = {}
        for line in file:
            line = line.strip()
            line = re.sub(';', '', line)
            if(line.startswith('MODULE')):
                name = line.split()[1]
            if(line.startswith('DIMENSIONS')):
                cords = line.split()[1:]
                width = float(cords[0])
                height = float(cords[3])
                # +11 in case of name of two combined module may have same name with a original module
                mappedName = str(len(nameMap) + 11)
                nameMap[mappedName] = name
                modules[mappedName] = [width, height]
    # print(nameMap)
    # print(modules)
    # modules = {'1':[4,6], '2':[4,4], '3':[4,3], '4':[4,4], '5':[4,3]} # rect in (width, height), for test only
    E = []
    i = 0
    # initial solution
    fp, y = sa(modules)
    x = range(len(y))
    plt.plot(x, y)
    plt.xlabel("iteration")
    plt.ylabel("cost")
    print('result:',''.join(fp))
    plotLayout(fp, modules)
    plt.show()
    with open('./out.txt', 'w') as f:
        for item in fp:
            if(item.isdigit()):
                f.write("%s " % nameMap[item])
            else:
                f.write("%s " % item)
# ami49
# 603213H4053V30H584218VHV5733H55HV395150HV2937V5447VHVHV5634HV3112HV1943V38V3620V2252VH59V2146V2815VHVH49172326HV4124VHV27V48V11HVH142516H4535V44HVHVH
# best cost: 104491520.0
# ami33
# 172830V24H3426VH2933H32VH13H38H4236H12HVH1639V1435VH1519HV1841H27H20HVH114022HV3743V31HV25V2123HV44HV
# best cost: 4548376.0
# xerox
# 2119181711H12HVH2015HV1413H16HVH
# best cost: 61446196.0
# apte
# 201518H1916H12H17HV1314H11VHV
# best cost: 149090000.0
# hp
# 121814HV19V17V2221V16V1513V1120VHVH
# best cost: 90724872.0
