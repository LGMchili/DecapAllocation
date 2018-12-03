from floorplanning import *
from allocation import *
modules, nameMap, loadMap, pwrPin = readDesign('./xerox.txt')
seq = decode('21161811H12HV13H1514HV192017VHVH')
rects, dimension, rotationMap = getDesign(seq, modules)
design = rectsToDict(rects, nameMap) # resulting design from floorplanning
print('number of whiteSpace:', len(design['ws']))
print('number of modules:', len(design) - 1)
# addDecap(design, loadMap, dimension)
placePin(design, pwrPin, nameMap, rotationMap)
# plotDesign(design, dimension, 'detailed', pwrPin)
nodeNumX, nodeNumY = 20, 20
# TODO: draw the noise map
netlistGenerator(pwrPin, nodeNumX, nodeNumY, dimension)
