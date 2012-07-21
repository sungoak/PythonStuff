# Classification Models Performance Comparisons
# Specifically: NaiveBayes, PNN, LDA/QDA, KNN, Logistic classifiers compared!

######## 6/27/2012
# - Test Holdout validation of PNN vs. other classifiers (LDA, KNN)....
# - The following code load linearSVM, KNN, and logistic classifiers from sklearn:

###############################################################
################### IRIS CLASSIFICATION TEST ##################
###############################################################

from sklearn import datasets;
iris = datasets.load_iris();

# load classification methods()
from sklearn import neighbors, datasets, linear_model, svm
classifiers = dict(
 knn=neighbors.KNeighborsClassifier(),
 logistic=linear_model.LogisticRegression(C=1e5),
 svm=svm.LinearSVC(C=1e5, loss='l1'),)

# Create Training and Test Datasets for Cross-validation
testIDX = hstack((arange(10,20),arange(60,70),arange(110,120)));
trainIDX = ones((size(iris.data,0)));
trainIDX[testIDX] = 0;
trainIDX = flatnonzero(trainIDX);
testdata = iris.data[testIDX,:];
traindata = iris.data[trainIDX,:];
y_predPNN = zeros((size(testIDX,0)))

# PNN Classifer Predictions
for k in range(size(testIDX,0)): 
 y_predPNN[k] = naivebayes.Rearden_pnn(traindata,iris.target[trainIDX],testdata[k,:],0.01,scaleInfo)
print "The PNN classification accuracy = " + str(len(flatnonzero(y_predPNN==iris['target'][testIDX]))/float(len(y_predPNN))) 

# KNN Nearest Neighbors Classifier Predictions
classifiers['knn'].fit(traindata,iris.target[trainIDX]);
y_predKNN = classifiers['knn'].predict(testdata);
print "The KNN classification accuracy = " + str(len(flatnonzero(y_predKNN==iris['target'][testIDX]))/float(len(y_predKNN))) 

# linearSVM Classifier Predictions
classifiers['svm'].fit(traindata,iris.target[trainIDX]);
y_predSVM = classifiers['svm'].predict(testdata);
print "The linearSVM classification accuracy = " + str(len(flatnonzero(y_predSVM==iris['target'][testIDX]))/float(len(y_predSVM))) 

# Logistic Classifier Predictions
classifiers['logistic'].fit(traindata,iris.target[trainIDX]);
y_predLog = classifiers['logistic'].predict(testdata);
print "The logistic classification accuracy = " + str(len(flatnonzero(y_predLog==iris['target'][testIDX]))/float(len(y_predLog))) 

# NaiveBayes Classifer Predictions
nbModIris = naivebayes.Rearden_nbtrain(traindata,iris.target[trainIDX]);
predNB = naivebayes.Rearden_nbclassify(testdata,nbModIris)
print "The Naive-Bayes classification accuracy = " + str(len(flatnonzero(predNB['classes'][:,0]==iris['target'][testIDX]))/float(len(predNB['classes'][:,0])))

## In order for NB model to function properly with "semi-continuous" data as with the IRIS data, we need to quantize or bin the values!!!
# --->>> We might have to come up with a categorical grouping method as seen in Bouille's paper with a merging criterion (Gini-like diversity index?)


#**** Let's try to use the NB-pkg IN sklearn pkg and see if it results in roughly the same accuracy or if there is something VERY differet

# MultinomialNB Classifier Predictions
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB();
y_predMNB = mnb.fit(traindata, iris.target[trainIDX]).predict(testdata)
print "MultinomialNB classification accuracy = " + str(len(flatnonzero(y_predMNB==iris['target'][testIDX]))/float(len(y_predMNB)))

# GaussianNB Classifier Predictions
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB();
y_predGNB = gnb.fit(traindata, iris.target[trainIDX]).predict(testdata)
print "MultinomialNB classification accuracy = " + str(len(flatnonzero(y_predGNB==iris['target'][testIDX]))/float(len(y_predGNB)))



 
###############################################################################
################ DIGITIZATION of IRIS DATA FOR MULTINOMIAL_NB #################
###############################################################################

# Here we are "digitizing" Iris data to see how the effects of creating discrete feature values from continous ones affects different NB modeling techniques including: PNN, GaussianNB, MultinomiaNB, and NB (Rearden_nbclassify)

# Digitize iris.data and then re-compute model performance metrics
# Digitization here is trivially and empirically done by simply binning by integer values of the iris data feature values (better bin values likely exist)
bins = [1]*5; # initialize list of arrays containing bins for digitization
bins[0] = array([4,5,6,7,8]);
bins[1] = array([2,3,4,5]);
bins[2] = array([1,2,3,4,5,6,7]);
bins[3] = array([0,1,2,3]);

irisBinnedData = zeros_like(iris.data);
for k in range(size(iris.data,1)):
 irisBinnedData[:,k] = digitize(iris.data[:,k],bins[k]);

testIDX = hstack((arange(10,20),arange(60,70),arange(110,120)));
trainIDX = ones((size(irisBinnedData,0)));
trainIDX[testIDX] = 0;
trainIDX = flatnonzero(trainIDX);
testdata = zeros_like(irisBinnedData[testIDX,:]); # testdata initialization
testdata = irisBinnedData[testIDX,:];
traindata = zeros_like(irisBinnedData[trainIDX,:]); # traindata initialization
traindata = irisBinnedData[trainIDX,:];
y_predPNN = zeros((size(testIDX,0)))

# NaiveBayes Classifer Predictions
nbModIris = naivebayes.Rearden_nbtrain(traindata,iris.target[trainIDX]);
predNB = naivebayes.Rearden_nbclassify(testdata,nbModIris)
print "The Naive-Bayes classification accuracy = " + str(len(flatnonzero(predNB['classes'][:,0]==iris['target'][testIDX]))/float(len(predNB['classes'][:,0])))
# acc = 0.8

# NaiveBayes Classifer Predictions (using only first/last feature))
nbModIris2 = naivebayes.Rearden_nbtrain(traindata[:,array([0,3])],iris.target[trainIDX]);
predNB2 = naivebayes.Rearden_nbclassify(testdata[:,array([0,3])],nbModIris2)
print "The Naive-Bayes classification accuracy = " + str(len(flatnonzero(predNB2['classes'][:,0]==iris['target'][testIDX]))/float(len(predNB2['classes'][:,0])))
# acc = 0.8

naivebayes.Rearden_nbholdoutCVsets(irisBinnedData[trainIDX,:],iris.target[trainIDX],irisBinnedData[testIDX,:],iris.target[testIDX])

# Basically what the MULTINOMIAL_NB model in sklearn package is doing is using
# some sort of "grouping" algorithm (possibly using Gini diversity index) to 
# determine how to "digitize" the feature values to be more meaningful in a 
# multinomial NB model (now my NAIVEBAYES implementation has the SAME accuracy 
# as with the multinomialNB() method in sklearn!!)


# MultinomialNB Classifications w/ digitized Iris Data
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB();
y_predMNB = mnb.fit(traindata, iris.target[trainIDX]).predict(testdata)
print "MultinomialNB classification accuracy = " + str(len(flatnonzero(y_predMNB==iris['target'][testIDX]))/float(len(y_predMNB)))
# acc = 0.7

# GaussianNB Classifications w/ digitized Iris Data
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB();
y_predGNB = gnb.fit(traindata, iris.target[trainIDX]).predict(testdata)
print "MultinomialNB classification accuracy = " + str(len(flatnonzero(y_predGNB==iris['target'][testIDX]))/float(len(y_predGNB)))
# acc = 0.1

### ---->>>> This very effectively demonstrates the dependence of NB model 
# performance on feature digitization and the need for an algorithm which
# digitizes things properly, optimally, and systematically for the BEST
# predictions overall!!


# RESULTS of Classifier Comparisons
# PNN and KNN classifiers seemed to have generalized the BEST for IRIS dataset
# linearSVM and Logistic classifiers are very close in performance
#Next we look at data that is a bit noisier and harder to classify and gauge the different classifiers ability to generalize


#################################################################
################### DIGITS CLASSIFICATION TEST ##################
#################################################################

# DIGITS dataset seems PERFECT for testing BOTH classification performance of different models as well as testing FEATURE SELECTION of different schemes such as NMIFS(), GAMIFS(), and other built-in feature selections methods in sklearn
from sklearn import datasets;
digits = datasets.load_digits();

# load classification methods()
from sklearn import neighbors, datasets, linear_model, svm
classifiers = dict(
 knn=neighbors.KNeighborsClassifier(),
 logistic=linear_model.LogisticRegression(C=1e5),
 svm=svm.LinearSVC(C=1e5, loss='l1'),)

# create cross-validation train and test sets
testIDX = array([],int);
for k in unique(digits['target']):
 testIDX = hstack((testIDX,flatnonzero(digits['target']==k)[:20]));
trainIDX = ones((len(digits['target'])));
trainIDX[testIDX] = 0;
trainIDX = flatnonzero(trainIDX);

testdata = digits.data[testIDX,:];
traindata = digits.data[trainIDX,:];

# PNN Probabilistic Neural Network Classifier Predictions
y_predPNN = zeros((size(testIDX,0)))
for k in range(size(testIDX,0)): 
 y_predPNN[k] = naivebayes.Rearden_pnn(traindata,digits.target[trainIDX],testdata[k,:],0.1)
print "The PNN classification accuracy = " + str(len(flatnonzero(y_predPNN==digits['target'][testIDX]))/float(len(y_predPNN))) 

# KNN Nearest Neighbors Classifier Predictions
classifiers['knn'].fit(traindata,digits.target[trainIDX]);
y_predKNN = classifiers['knn'].predict(testdata);
print "The KNN classification accuracy = " + str(len(flatnonzero(y_predKNN==digits['target'][testIDX]))/float(len(y_predKNN))) 

# linearSVM Classifier Predictions
classifiers['svm'].fit(traindata,digits.target[trainIDX]);
y_predSVM = classifiers['svm'].predict(testdata);
print "The linearSVM classification accuracy = " + str(len(flatnonzero(y_predSVM==digits['target'][testIDX]))/float(len(y_predSVM))) 

# Logistic Classifier Predictions
classifiers['logistic'].fit(traindata,digits.target[trainIDX]);
y_predLog = classifiers['logistic'].predict(testdata);
print "The logistic classification accuracy = " + str(len(flatnonzero(y_predLog==digits['target'][testIDX]))/float(len(y_predLog))) 

# NaiveBayes Classifier Predictions
nbModDigits = naivebayes.Rearden_nbtrain(traindata,digits.target[trainIDX]);
predNB = naivebayes.Rearden_nbclassify(testdata,nbModDigits)
print "The Naive-Bayes classification accuracy = " + str(len(flatnonzero(predNB['classes'][:,0]==digits['target'][testIDX]))/float(len(y_predLog)))

# NMIFS() feature selection 
import informationTheory
traindata2 = array(traindata,int)
featrankDigits,subsetDigits = informationTheory.nmifs3(traindata2,digits.target[trainIDX],38)

selfeatsDigits = flatnonzero(subsetDigits);

# NaiveBayes Classifier Predictions (NIMFS3() features only)
nbModDigits = naivebayes.Rearden_nbtrain(traindata[:,selfeatsDigits],digits.target[trainIDX]);
predNB = naivebayes.Rearden_nbclassify(testdata[:,selfeatsDigits],nbModDigits)
print "The Naive-Bayes classification accuracy = " + str(len(flatnonzero(predNB['classes'][:,0]==digits['target'][testIDX]))/float(len(y_predLog)))

### NOW, rerun holdout validation with ONLY nmifs3 selected features
# PNN Probabilistic Neural Network Classifier Predictions
y_predPNN2 = zeros((size(testIDX,0)))
for k in range(size(testIDX,0)): 
 y_predPNN2[k] = naivebayes.Rearden_pnn(traindata[:,flatnonzero(subsetDigits)],digits.target[trainIDX],testdata[k,flatnonzero(subsetDigits)],0.4)
print "The PNN classification accuracy = " + str(len(flatnonzero(y_predPNN2==digits['target'][testIDX]))/float(len(y_predPNN2))) 


# RESULTS:
# PNN has BEST classification accuracy (93.5%!)
# KNN has next best at 93%
# linearSVM and Logistic classifiers have 90% and 88% accuracies, respectively
# Naive-Bayes classification accuracy was dismal at ~ 66% 
# NB-model with 40 features --> (71% with NMIFS())
# NB-model with 20 features --> (47% with NMIFS())
#---> NB-model might have been affected by the fact that there are 64 features, many of which have many unique values (16 or more)
# Seems PNN and KNN have some inherent overlap in how they perform overall!

# RESULTS of NMIFS3 feature selected PNN predictions
# --> the prediction quality goes down significantly even when only 20 features 



#############################################################################
################## HR+Acxiom Data Classification Testing ####################
#############################################################################

#NEXT things to do is to test PNN predictions w/ and w/o NMIFS3()!!!
allUserIDX = arange(size(hrAxUnn2,0)); # all indices from 0 to 2318
testIDX = permutation(allUserIDX)[:250];
trainIDX = ones((len(allUserIDX)));
trainIDX[testIDX] = 0;
trainIDX = flatnonzero(trainIDX);

traindata = hrAxUnn2[trainIDX,:];
testdata = hrAxUnn2[testIDX,:];

winnersNum,garbage,garbage = Rearden_grp2idx(winners[:,0]);
selHrAxfeats = flatnonzero(subset);

# PNN classificatio with ONLY subset of features from NMIFS3() feat ranking
y_predPNN3 = zeros((size(testIDX,0)))
for k in range(size(testIDX,0)): 
 print k
 y_predPNN3[k] = naivebayes.Rearden_pnn(traindata[:,selHrAxfeats],winnersNum[trainIDX],testdata[k,selHrAxfeats],0.15)
print "The PNN classification accuracy = " + str(len(flatnonzero(y_predPNN3==winnersNum[testIDX]))/float(len(y_predPNN3))) 
# acc ~ 0.15

# PNN classification with ALL features
y_predPNN3 = zeros((size(testIDX,0)))
for k in range(size(testIDX,0)): 
 print k
 y_predPNN3[k] = naivebayes.Rearden_pnn(traindata,winnersNum[trainIDX],testdata[k,:],0.15)
print "The PNN classification accuracy = " + str(len(flatnonzero(y_predPNN3==winnersNum[testIDX]))/float(len(y_predPNN3))) 
# acc ~ 0.104

# **** NEED to check HOW PNN() method outputs predictions for test dataset
# --->> seems there is numerical conversion for outputs where class labels are not getting designated properly, fix this then assess performance of PNN!
# **** Biggest issue with PNN predictions on HR_acxiom data set is that there is an inherent dependence on correlation which may be STRONG for a given observation but since we're taking the "average" correlation, only a single obs from a small represented category (or class) can look like a dominant sum in PNN
# Therefore, in this case, as we dont have sample balance and the data are very noisy, it seems that the multinomial versions of NB are better suited and it is evidenced in the classification accuracy between PNN vs NB (multinomial)

# NaiveBayes Classifer Predictions (selected features only)
nbModIris = naivebayes.Rearden_nbtrain(traindata[:,selHrAxfeats],winnersNum[trainIDX]);
predNB = naivebayes.Rearden_nbclassify(testdata[:,selHrAxfeats],nbModIris)
print "The Naive-Bayes classification accuracy = " + str(len(flatnonzero(predNB['classes'][:,0]==winnersNum[testIDX]))/float(len(predNB['classes'][:,0])))
# acc = 0.356
nbModIris = naivebayes.Rearden_nbtrain(traindata,winnersNum[trainIDX]);
predNB = naivebayes.Rearden_nbclassify(testdata,nbModIris)
print "The Naive-Bayes classification accuracy = " + str(len(flatnonzero(predNB['classes'][:,0]==winnersNum[testIDX]))/float(len(predNB['classes'][:,0])))
# acc = 0.348

# MultinomialNB Classifications w/ digitized Iris Data
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB();
y_predMNB = mnb.fit(traindata[:,selHrAxfeats], winnersNum[trainIDX]).predict(testdata[:,selHrAxfeats])
print "MultinomialNB classification accuracy = " + str(len(flatnonzero(y_predMNB==winnersNum[testIDX]))/float(len(y_predMNB)))
# acc = 0.272
y_predMNB = mnb.fit(traindata, winnersNum[trainIDX]).predict(testdata)
print "MultinomialNB classification accuracy = " + str(len(flatnonzero(y_predMNB==winnersNum[testIDX]))/float(len(y_predMNB)))
# acc = 0.26

# GaussianNB Classifications w/ digitized Iris Data
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB();
y_predGNB = gnb.fit(traindata[:,selHrAxfeats], winnersNum[trainIDX]).predict(testdata[:,selHrAxfeats])
print "GaussianNB classification accuracy = " + str(len(flatnonzero(y_predGNB==winnersNum[testIDX]))/float(len(y_predGNB)))
# acc = 0.064
y_predGNB = gnb.fit(traindata, winnersNum[trainIDX]).predict(testdata)
print "GaussianNB classification accuracy = " + str(len(flatnonzero(y_predGNB==winnersNum[testIDX]))/float(len(y_predGNB)))
# acc = 0.184


######################################################################
#################### Bootstrap UPSAMPLING Test #######################
######################################################################

################# 6/28/2012
#- seems the biggest problem in dealing with PNN and HR+acxiom data classification is the SAMPLE IMBALANCE and trying to deal with that
#- we can try to down-sample or undersample the over represented classes, but we might be able to weight the "summs" of the PNN properly to reflect integrity of EACH class, therefore auto accounting for this and improving classification
#- We should actually try UPSAMPLING under represented classes and see if we can achieve some sort of sample balance and determine how well the PNN performs
# - use the following code for upsampling:

# BOOTSTRAP UPSAMPLING towards MAXIMUM class label size
import random # module which allows for more random sampling methods
winCat,catLabels,catCounts = Rearden_strmode(winners[:,0]);
maxcatCount = max(catCounts);
uniqCats = unique(winners);
for k in range(len(catLabels)): # loop thru all classes, determine IDXs
 currCatIDX = flatnonzero(winners[:,0]==catLabels[k]);
 upsampleAmount = maxcatCount - len(currCatIDX);
 upsampleRatio = maxcatCount/len(currCatIDX);
 tiledUpsample = tile(currCatIDX,upsampleRatio); # repeats w/ len ~ maxcatCount
 upsampledIDX = random.sample(tiledUpsample,len(tiledUpsample));


# BOOTSTRAP UPSAMPLING towards USER-SPECIFIED class label size
import random # module which allows for more random sampling methods
winCat,catLabels,catCounts = Rearden_strmode(winners[:,0]);
maxcatCount = 200; # 200 samples chosen s.t. total population size ~ 2400
uniqCats = unique(winners);
hrAxUnn2Boot = array([],int); # initialize empty array
winnersBoot = array([],int); # initialize empty array
for k in range(len(catLabels)): # loop thru all classes, determine IDXs
 currCatIDX = flatnonzero(winners[:,0]==catLabels[k]);
 upsampleAmount = maxcatCount - len(currCatIDX);
 upsampleRatio = maxcatCount/len(currCatIDX);
 if upsampleRatio >= 2: # if upsampling is needed
  tiledUpsample = tile(currCatIDX,upsampleRatio); # repeats w/ len ~ maxcatCount
  upsampledIDX = random.sample(tiledUpsample,len(tiledUpsample));
 elif upsampleRatio == 1: # case ratio is less than integer multiply of sample
  upsampledIDX = hstack((currCatIDX,permutation(currCatIDX)[:upsampleAmount]));
 else: # if down sampling is needed
  upsampledIDX = random.sample(currCatIDX,maxcatCount); # randsamp maxcatCount
 if k == 0: # no concatenation for first iteration
  hrAxUnn2Boot = hrAxUnn2[upsampledIDX,:];
  winnersBoot = winners[upsampledIDX,0];
 else: 
  hrAxUnn2Boot = vstack((hrAxUnn2Boot,hrAxUnn2[upsampledIDX,:]));
  winnersBoot = hstack((winnersBoot,winners[upsampledIDX,0]));

# verify bootstrapping of sample
for k in range(len(unique(winnersBoot))):
 print len(flatnonzero(winnersBoot==uniqCats[k]))

# PNN Classification

allUserIDX = arange(size(hrAxUnn2Boot,0));

for k in range(len(uniqCats)):
 currCatIDX = flatnonzero(winners[:,0]==catLabels[k]);
 
 if len(currCatIDX)<20: # case there ARENT even enough class examples
  amountNeeded = 20-len(currCatIDX);
  currCatIDX = hstack((currCatIDX,permutation(currCatIDX)[:amountNeeded]));
  
 if k == 0:
  testIDX = permutation(currCatIDX)[:20];
 else:
  testIDX = hstack((testIDX,permutation(currCatIDX)[:20]))
  
trainIDX = ones((len(allUserIDX)));
trainIDX[testIDX] = 0;
trainIDX = flatnonzero(trainIDX);

traindata = hrAxUnn2Boot[trainIDX,:];
testdata = hrAxUnn2Boot[testIDX,:];

winnersNum,garbage,garbage = Rearden_grp2idx(winnersBoot);

y_predPNN3 = zeros((size(testIDX,0)))
for k in range(size(testIDX,0)): 
 print k
 y_predPNN3[k] = naivebayes.Rearden_pnn(traindata[:,flatnonzero(subset)],winnersNum[trainIDX],testdata[k,flatnonzero(subset)],0.15)
print "The PNN classification accuracy = " + str(len(flatnonzero(y_predPNN3==winnersNum[testIDX]))/float(len(y_predPNN3))) 

y_predPNN3 = zeros((size(testIDX,0)))
for k in range(size(testIDX,0)): 
 print k
 y_predPNN3[k] = naivebayes.Rearden_pnn(traindata,winnersNum[trainIDX],testdata[k,:],0.35)
print "The PNN classification accuracy = " + str(len(flatnonzero(y_predPNN3==winnersNum[testIDX]))/float(len(y_predPNN3))) 


hrfts = [];
for k in range(len(hrfeats)):
 hrfts.append(flatnonzero(hrAxfeats==hrfeats[k]));
hrfts = squeeze(hrfts)

y_predPNN3 = zeros((size(testIDX,0)))
rbfmeans = -1*ones((size(testdata,0),len(uniqCats)));
rbfsds = -1*ones((size(testdata,0),len(uniqCats)));

for k in range(size(testIDX,0)): 
 print k
 y_predPNN3[k],rbfmeans[k,:],rbfsds[k,:] = naivebayes.Rearden_pnn(traindata[:,hrfts],winnersNum[trainIDX],testdata[k,hrfts],0.35)
print "The PNN classification accuracy = " + str(len(flatnonzero(y_predPNN3==winnersNum[testIDX]))/float(len(y_predPNN3))) 

# perhaps we need to partition test vs. train by class, stratify the 
# classes within the test and train data sets
# -->>>> This however, DID NOT work and looks like PNN is NOT converging onethe solutions as good as ordinary NB-model (as seen in other studies)
# --> PNN would likely be better with BETTER quality data!!!



# ----->>>>> Now look at possibly using PNN approach for deal-recommendations in our HR and HR+Acxiom datasets (gauge its performance with and w/o FS)

#NEXT things to do is to test PNN predictions w/ and w/o NMIFS3()!!!
allUserIDX = arange(size(hrAxUnn,0)); # all indices from 0 to 2318
testIDX = permutation(allUserIDX)[:250];
trainIDX = ones((len(allUserIDX)));
trainIDX[testIDX] = 0;
trainIDX = flatnonzero(trainIDX);

traindata = hrAxUnn2[trainIDX,:];
testdata = hrAxUnn2[testIDX,:];

winnersNum,garbage,garbage = Rearden_grp2idx(winners[:,0]);

y_predPNN3 = zeros((size(testIDX,0)))
for k in range(size(testIDX,0)): 
 print k
 y_predPNN3[k] = naivebayes.Rearden_pnn(traindata[:,flatnonzero(subset)],winnersNum[trainIDX],testdata[k,flatnonzero(subset)],0.15)
print "The PNN classification accuracy = " + str(len(flatnonzero(y_predPNN3==winnersNum[testIDX]))/float(len(y_predPNN3))) 

y_predPNN3 = zeros((size(testIDX,0)))
for k in range(size(testIDX,0)): 
 print k
 y_predPNN3[k] = naivebayes.Rearden_pnn(traindata,winnersNum[trainIDX],testdata[k,:],0.15)
print "The PNN classification accuracy = " + str(len(flatnonzero(y_predPNN3==winnersNum[testIDX]))/float(len(y_predPNN3))) 

# **** NEED to check HOW PNN() method outputs predictions for test dataset
# -->> seems there is numerical conversion for outputs where class labels are not getting designated properly, fix this then assess performance of PNN!


