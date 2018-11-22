import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def rotateRects(module):
    # add rotated rects to list
    rects = module.copy()
    for k in rects.keys():
        width, height = rects[k][0], rects[k][1]
        rects[k] = [rects[k]]
        if(width != height):
            rects[k].append((height, width));
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

def plotLayout(seq, module):
    rects = module.copy()
    stack = []
    pack = {}
    for k in rects.keys():
        # format in [width, height, x, y]
        pack[k] = [[rects[k][0], rects[k][1], 0, 0]]
    for m in fp:
        if(m == 'H' or m == 'V'):
            operator = m
            m1 = stack.pop()
            m2 = stack.pop()
            w1, h1, w2, h2 = rects[m1][0], rects[m1][1], rects[m2][0], rects[m2][1]
            if(m1+m2 not in pack): pack[m1+m2] = []
            offset_x = rects[m2][0]
            offset_y = rects[m2][1]
            if(m == 'H'):
                temp = (max(w1, w2), h1 + h2)

                for mm in pack[m1]: # right part
                    mm[3] += offset_y
                    pack[m1+m2].append(mm)
                for mm in pack[m2]: # left part
                    pack[m1+m2].append(mm)
            elif(m == 'V'):
                temp = (w1 + w2, max(h1, h2))
                for mm in pack[m1]: # right part
                    mm[2] += offset_x
                    pack[m1+m2].append(mm)
                for mm in pack[m2]: # left part
                    pack[m1+m2].append(mm)
            rects[m1+m2] = temp
            stack.append(m1+m2)
            # print(rects[m2], rects[m1], operator)
        else:
            stack.append(m)
    result = pack[m1+m2]
    x, y = rects[m1+m2]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_ylim([0,y])
    ax.set_xlim([0,x])
    for rr in result:
        w = patches.Rectangle((rr[2],rr[3]),rr[0],rr[1],linewidth=1,edgecolor='b',facecolor='y')
        ax.add_patch(w)

def calculateCost(fp, module):
    rects = rotateRects(module)
    stack = []
    for m in fp:
        if(m == 'H' or m == 'V'):
            operator = m
            temp = []
            m1 = stack.pop()
            m2 = stack.pop()
            for r1 in rects[m1]:
                for r2 in rects[m2]:
                    w1, h1, w2, h2 = r1[0], r1[1], r2[0], r2[1]
                    if(m == 'H'):
                        temp.append((max(w1, w2), h1 + h2))
                    elif(m == 'V'):
                        temp.append((w1 + w2, max(h1, h2)))
            temp = removeRedundant(temp)
            stack.append(m1+m2)
            rects[m1+m2] = temp
        else:
            stack.append(m)
    rect = max(temp)
    return rect[0] * rect[1]
fp = ['1','2','H','3','4','H','5','H','V']
module = {'1':(4,6), '2':(4,4), '3':(4,3), '4':(4,4), '5':(4,3)} # rect in (width, height)
cost = calculateCost(fp, module)
print('curr area:', cost)
plotLayout(fp, module)
plt.show()
