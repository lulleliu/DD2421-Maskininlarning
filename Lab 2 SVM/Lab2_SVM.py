import random, math, random
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#The location of the decision boundary is given by the weights (w) and the bias (b) so the problem
#is to find the values for w and b which maximizes the margin, i.e. the distance to any datapoint.

def generate_data(seed):
    np.random.seed(seed)
    classA = np.concatenate(
        (np.random.randn(10,2) * 0.2 + [1.5, 0.5],
        np.random.randn(10,2) * 0.2 + [-1.5, 0.5]))
    
    classB = np.random.randn(20,2) * 0.2 + [0.0,-0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate(
        (np.ones(classA.shape[0]),
        -np.ones(classB.shape[0])))
    
    N = inputs.shape[0] # Number of rows (samples)

    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    return N, permute, inputs, targets, classA, classB


def kernel(data1: np.array, data2: np.array, kerneltype = "linear", d = 3, sigma = 1):
    match kerneltype:
        case "linear": 
            return np.dot(data1, data2)
        case "polynominal": 
            return (np.dot(data1, data2)+1)^d
        case "RBF":
            return np.exp(-np.dot(data2 - data1, data2 - data1) / 2 / sigma / sigma)

def objective(alpha: np.array, x: np.array, t: np.array, kerneltype = "linear"):
    target = 0
    N = np.size(x, 0)
    for i in np.arange(N):
        for j in np.arange(N):
           target += alpha[i] * alpha[j] * t[i] * t[j] * kernel(x[i], x[j], kerneltype)
    target -= alpha.sum()
    return target

def zerofun(alpha: np.array):
    return np.dot(alpha, targets)

def shorten(alpha: np.array, x: np.array, y: np.array):
    ls = []
    ls_x = []
    ls_y = []
    ls_ind = []
    for ind, val in enumerate(alpha):
        if val > 0.0001:
            ls.append(val)
            ls_x.append(x[ind])
            ls_y.append(y[ind])
            ls_ind.append(ind)
    return ls, ls_x, ls_y, ls_ind

def indicator(s, x, y, b, alpha_list, kerneltype = 'linear'):
    ind_sum = 0
    for i in range(len(alpha_list)):
        ind_sum += alpha_list[i]*y[i]*kernel(s, x[i], kerneltype)
    
    return ind_sum - b

def calc_b(a: np.array, x0, y0, x, y, kerneltype = 'linear'):
    b = 0
    for i in range(len(alpha_list)):
        b += a[i]*y[i]*kernel(x0, x[i], kerneltype)
    b -= y0
    return b



N, permute, inputs, targets, classA, classB = generate_data(114514)

start0 = np.zeros(N)
kern_type = 'linear'

# C used for boundsx
C = 0.995

ret = minimize(objective, args = (inputs, targets, kern_type), x0 = start0, bounds = [(0, C) for b in range(N)], constraints = {'type': 'eq', 'fun': zerofun})
alpha = ret["x"]
print(shorten(alpha, inputs, targets))  ######################
alpha_list, x_list, y_list, ind_list = shorten(alpha, inputs, targets)
x0 = x_list[0]
y0 = y_list[0]
b = calc_b(alpha_list, x0, y0, inputs, targets, kern_type)

# PLOTTING
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
grid = np.array([[indicator(np.array([x, y]), x_list, y_list, b, alpha_list, kern_type) for x in xgrid] for y in ygrid])
print(grid)
plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
colors = ('red', 'black', 'blue'), linewidths = (1, 3, 1))

plt.axis('equal') # Force same scale on both axes
plt.savefig('svmplot.pdf') # Save a copy in a file
plt.show() # Show th eplot on the screen
