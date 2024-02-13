import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def generate_data():
    classA = numpy.concatenate(
        (numpy.random.randn(10,2) * 0.2 + [1.5, 0.5],
        numpy.random.randn(10,2) * 0.2 + [-1.5, 0.5]))
    
    classB = numpy.random.randn(20,2) * 0.2 + [0.0,-0.5]

    inputs = numpy.concatenate((classA, classB))
    targets = numpy.concatenate(
        (numpy.ones(classA.shape[0]),
        -numpy.ones(classB.shape[0])))
    
    N = inputs.shape[0] # Number of rows (samples)

    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets [permute]

    return N, permute, inputs, targets

print(generate_data())
    
