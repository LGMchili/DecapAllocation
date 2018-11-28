from floorplanning import *
from allocation import *
modules, nameMap, loadMap = readDesign('./xerox.txt')
seq = decode('21161811H12HV13H1514HV192017VHVH')
rects, dimension = getDesign(seq, modules)
design = rectsToDict(rects, nameMap) # resulting design from floorplanning
print('number of whiteSpace:', len(design['ws']))
print('number of modules:', len(design) - 1)
addDecap(design, loadMap, dimension)
plotDesign(design, dimension, 'detailed')
