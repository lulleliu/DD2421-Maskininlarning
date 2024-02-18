import numpy, random,  math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

### Generating Test Data
numpy.random.seed(100)
classA = numpy.concatenate(
    (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
     numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
classB = numpy.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate(
    (numpy.ones(classA.shape[0]),
     -numpy.ones(classB.shape[0])))

N = inputs.shape[0] # Number of rows (samples)

permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

###

# Kernel function
def K(vector1, vector2):
    p = 2 # Exponent for polynomial kernel
    sigma = 0.5 # For RBF kernel
    #return numpy.dot(vector1, vector2) # Linear kernel
    return (numpy.dot(vector1, vector2) + 1)**p # Polynomial kernel
    ##return numpy.exp(-(numpy.abs(vector1-vector2)**2)/(2*sigma**2)) # RBF Kernel

def p_matrix():
    P = numpy.empty([N, N])
    for i in range(N):
        for j in range(N):
            P[i][j] = targets[i]*targets[j]*K(inputs[i], inputs[j])
    return P
P = p_matrix()
        


# Objective function, eq. 4
def objective(alpha):
##    for i in len(alpha):
##        for j in len(alpha):
    dual_sum = numpy.dot(alpha, P)
    dual_sum = numpy.dot(dual_sum, alpha)
            
    return 0.5*dual_sum - numpy.sum(alpha)

# Constraint, eq. 5
def zerofun(vector): 
    return numpy.dot(vector, targets)

# Plockar ut alla element över en treshold
def extract_non_zero(alpha):
    treshold = 10e-5
    non_zero = []
    for index, element in numpy.ndenumerate(alpha):
        if abs(element) > treshold:
            non_zero.append((element, inputs[index], targets[index]))     
    return non_zero

# Equation 7
# x_i corresponding to non zero alpha = support vectors
def treshold_b():
    b = 0
    s = non_zero[0][1] # Valfri support vector
    t_s = non_zero[0][2] # Target för den support vectorn
##    for i in range(N):
##        b += alpha[i]*targets[i]*K(s, inputs[i])
    for alphas in non_zero:
        b += alphas[0]*alphas[2]*K(s, alphas[1])
    b = b - t_s 
    return b

def indicator(x, y):
    ind_sum = 0
##    for i in range(N):
##        ind_sum = alpha[i]*targets[i]*K([x, y], s)
    for alphas in non_zero:
        ind_sum += alphas[0]*alphas[2]*K([x, y], alphas[1])
    
    return ind_sum - b
        
    
    
if __name__ == '__main__':

    # From 4 implementation
    C = 1
    start = numpy.zeros(N)  # Initial guess of the alpha-vector
    B = [(0, C) for b in range(N)]
    ##B = [(0, None) for b in range(N)] # Only lower bounds
    XC = {'type':'eq', 'fun':zerofun}

    ret = minimize(objective, start, bounds = B, constraints = XC)
    alpha = ret['x']
    non_zero = extract_non_zero(alpha)
    b = treshold_b()



    ### 6 Plotting
    plt.plot([p[0] for p in classA],
             [p[1] for p in classA],
             'b.')
    plt.plot([p[0] for p in classB],
             [p[1] for p in classB],
             'r.')
    
    xgrid = numpy.linspace(-5, 5)
    ygrid = numpy.linspace(-4, 4)
    grid = numpy.array([[indicator(x, y)
                         for x in xgrid]
                        for y in ygrid])
    print(grid)
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
    colors = ('red', 'black', 'blue'), linewidths = (1, 3, 1))

    plt.axis('equal') # Force same scale on both axes
    plt.savefig('svmplot.pdf') # Save a copy in a file
    plt.show() # Show th eplot on the screen

    # 6.1 Plotting the Decision Boundary

   
