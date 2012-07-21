#### 5/9/2012   **************** Regularization Exercises *******************
import regularization
from numpy import *
from pylab import plot,figure,label,title,legend
# load x5,y5 data from text file
x5 = loadtxt("/Users/bchoi/Desktop/ex5Data/ex5Linx.dat");
y5 = loadtxt("/Users/bchoi/Desktop/ex5Data/ex5Liny.dat");

# try to fit a 5th-order polynomial to create weights for prediction
x5p = ones((size(x5,0),5)); # initialize x5p, will store all powers of x5 
for k in range(5):
 x5p[:,k] = x5**(k+1);

# Gradient Descent via L2-regularizatino to determine parameter weights (theta) 
# --> could also use L1, but could run into sparsity issues with L1, which allows
# larger weight value disparities since it has a more relaxed penalty on weight norm!
# theta = nan, learningRate = 10 (can use 0,1,10,etc.), alpha = 1, numIters = 100
alpha = 0.5; # learning rate for 
numIters = 10000;
theta = zeros((size(x5p,1)+1,1))
theta10,Jhist10 = regularization.gradient_descent_L2(x5p,y5,theta,10, alpha, numIters)
theta1,Jhist1 = regularization.gradient_descent_L2(x5p,y5,theta,1, alpha, numIters)
theta0,Jhist0 = regularization.gradient_descent_L2(x5p,y5,theta,0, alpha, numIters)

# create x-data for plotting "smooth" lines for plotting fit-lines
xvals = arange(-1,1,0.01)
numweights = 6; 
a = zeros((len(xvals),numweights)); # initialize a to store x-data being created
for k in range(numweights): # loop thru 6 parameter weights (including intercept value)
 a[:,k] = xvals**k

# plot actual x5 vs. y5 data (actual values being plotted for reference)
figure();
plot(x5,y5,'ro')
# plot data of "fitted, regularized parameter weights" vs. actual values (y5)
plot(a[:,1],a.dot(theta0),'g:',linewidth=2)
#plot(a[:,1],polyval(theta0[::-1],a[:,1]),'m:',linewidth=2)
plot(a[:,1],a.dot(theta1),'b:',linewidth=2)
plot(a[:,1],a.dot(theta10),'k:',linewidth=2)
legend(('Raw Data','unregularized','regularized, l=1','regularized, l=10'))
title('Fit Optimization via L2-Regularization')

# NOTE: The above plot obviates the need for regularization, particularly
# when the number of obs are on par (or roughly on par) with the number of 
# features in the data set, which is very susceptible to overfit!
# --> Incidentally, specifying a "penalty" parameter (lambda) which is
# too small or too large can lead to overfits or limitations on weight
# values which restricts proper weighting, respectively.
# --> proper visualization and integrity checks of the regularized
# fits should be performed to make sure weights have "learned" properly
