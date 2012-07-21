# PCA_comparisons
# This script compares the different types of PCA available in sklearn.decomposition

# use PCA to find most predictive PCs and their loadings to determine best features
from sklearn.decomposition import PCA
pca = PCA(n_components=4);
pca.fit(iris.data);
print pca.explained_variance_ratio_
# whitened version of PCA
pca = PCA(n_components=4,whiten=True);
pca.fit(iris.data);
print pca.explained_variance_ratio_

# look into ProbabilisticPCA, RandomizedPCA, KernelPCA, amd SparsePCA methods
# RandomizedPCA
from sklearn import decomposition
pca = decomposition.RandomizedPCA(n_components=4,whiten=True);
pca.fit(iris.data);
print pca.explained_variance_ratio_,pca.components_
# ProbabilisticPCA
pca = decomposition.ProbabilisticPCA(n_components=4,whiten=True);
pca.fit(iris.data);
print pca.explained_variance_ratio_,pca.components_
# KernelPCA linear
pca = decomposition.KernelPCA(n_components=4,kernel='linear');
pca.fit(iris.data);
print pca.lambdas_
# KernelPCA quadratic
pca = decomposition.KernelPCA(n_components=4,kernel='poly',degree=2);
pca.fit(iris.data);
print pca.lambdas_
# KernelPCA cubic
pca = decomposition.KernelPCA(n_components=4,kernel='poly',degree=3);
pca.fit(iris.data);
print pca.lambdas_
# KernelPCA RBF (radial basis function)
pca = decomposition.KernelPCA(n_components=4,kernel='rbf');
pca.fit(iris.data);
print pca.lambdas_

#############################################################################
########## Determine how to use fit_transform and inverse_transform #########
#############################################################################
pca = PCA(n_components=1);
pca.fit(iris.data);
X_pca = pca.fit_transform(iris.data);
X_back = pca.inverse_transform(X_pca);
pca_diff = abs(iris.data-X_back);
# NOTE1: pca.components_ ARE the LOADINGS or WEIGHTS! not the eigvectors!
# seems that X_pca or the fit_transform output ARE the EIGVECTORS?!?!?!
# NOTE2: fit_transform() method simply TAKES mean-centered version of xdata, and then multiplies it by the loadings of the number of components specified
# NOTE3: inverse_transform() method then takes the fit_transform() method output and in turn 
a,b = bMath.meancenter(iris.data)
X_pca2 = dot(a,pca.components_.T) # SAME as pca.fit_transform(iris,data)
# remember pca.components_ only takes on number of components specified by user
# Here a is mean-centered FULL rank matrix of original X-data for iris.data
X_pca - X_pca2 # yields a vector of ZEROS, showing equivalence of two methods

# now we need to determine HOW pca.inverse_transform() returns data back to the original data space of iris.data
# Basically we are taking the eigvectors (array) and transforming them BACK to the orignal data space using the pca.components_ weights, matrix dividing??
X_back2 = X_pca*pca.components_; # (150x1)*(1x4) --> 150x4 original dims!!
for k in range(size(X_back2,1)): # loop thru all columns of X_back2
 X_back2[:,k] = X_back2[:,k] + b[k]; # add back "subtracted mean"

 
#############################################################################
######### Comparison of LDA vs. PCA on Iris Data (LDA is supervised) ########
#############################################################################
import pylab as pl

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LDA(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print 'explained variance ratio (first two components):', \
    pca.explained_variance_ratio_

pl.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    pl.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
pl.legend()
pl.title('PCA of IRIS dataset')

pl.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    pl.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
pl.legend()
pl.title('LDA of IRIS dataset')

pl.show()


#############################################################################
############# Using KernelPCA for Capturing Nonlinear Variance ##############
#############################################################################
import numpy as np
import pylab as pl

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

np.random.seed(0) # this is used to produce a KNOWN random output (for example)

X, y = make_circles(n_samples=400, factor=.3, noise=.05)

kpca = decomposition.KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)
pca = PCA()
X_pca = pca.fit_transform(X)

reds = y == 0
blues = y == 1

figure()
subplot(2,2,1); # Original Data Space
title("Original Space")
plot(X[reds,0],X[reds,1],'ro');
plot(X[blues,0],X[blues,1],'bo');
xlabel("$x_1$"),ylabel("$x_2$")

subplot(2,2,2); # PCA Data Space (PC1 vs. PC2)
title("PC1 vs. PC2 Projection")
plot(X_pca[reds,0],X_pca[reds,1],'ro');
plot(X_pca[blues,0],X_pca[blues,1],'bo');
xlabel("$PC_1$"),ylabel("$PC_2$")

subplot(2,2,3); # KernelPCA Data Space (first 2 phi components)
title("Projection by KernelPCA"); 
plot(X_kpca[reds,0],X_kpca[reds,1],'ro');
plot(X_kpca[blues,0],X_kpca[blues,1],'bo');
xlabel("$component_1$"),ylabel("$component_2$")

subplot(2,2,4); # KernelPCA (converting projection BACK to Original Space)
title("Return to Original Space from KernelPCA");
plot(X_back[reds,0],X_back[reds,1],'ro');
plot(X_back[blues,0],X_back[blues,1],'bo');
xlabel("$x_1$ (back)"),ylabel("$x_2$ (back)")

# RESULTS are VERY clear, KernelPCA has BEST discriminating power between the red and blue classes here, easily picking up the nonlinear projections which can correctly classify ALL the points with ease. For instance, simply using the FIRST component from KernelPCA, that alone could be used to separate out the red and blue classes simply using the decision boundry between [-0.3,0.1], which is a considerable margin as a margin for classification



#############################################################################
############### SparePCA and L1-Regularization of Components## ##############
#############################################################################







