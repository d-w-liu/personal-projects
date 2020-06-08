import numpy as np
import random
import matplotlib.pyplot as plt

random.seed()

xcoordinates = []
ycoordinates = []
i = 0

for i in range(0,1000):
    xcoordinates.append(random.gauss(50,12))
    ycoordinates.append(random.gauss(50,12))
    while xcoordinates[i]<=0:
        xcoordinates[i] = random.gauss(30,5)
    while ycoordinates[i]<=0:
        ycoordinates[i] = random.gauss(30,5)

plt.scatter(xcoordinates, ycoordinates, marker='o', s=5)