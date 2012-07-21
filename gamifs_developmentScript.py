# GAMIFS practice script

# One thing we need to do is think of how to implement a "stopping condition"

# fitness method for GAMIFS
def gamifs_fitness2(mod,xdata,ydata,r=0.2,numSel,numTot,a=0.5):
 acc = mod(xdata,ydata,r);
 J = acc - a*(numSel/numTot); 
 return J
 
 
# ----> Excerpt from GAMIFS() method
# Assume upstream computation will have:
pop,featrank,subset = informationTheory.initializeGA_nmifs(xdata,ydata,popsize,theta,rho,delta);

# Determine Fitness values for each individual in GA population (current generation)
if Jmax == 0: # case NO Jmax was specified (first generation of GA)
 J = zeros((popsize));
 for k in range(popsize): # loop thru EACH individual in population & obtain fitness
  acc = naivebayes.Rearden_nbholdoutCV(xtrain,ytrain,0.2);
  J[k] = gamifs_fitness(acc,len(flatnonzero(subset)),size(xtrain,1));
 Jmax = max(J); # extract maximum Jmax value to be used
 
 
 # >>>>>> Need to create a new method in naivebayes module which can do a one time test assessment on classificaiton accuracy using a specified holdout sample. The holdout sample size should be able to be specified by a fraction OR actual number

# --> Finished Rearden_nbholdout() method, to test it out, make sure it produces the SAME accuracy numbers as doing it with command line executions
# Look at the classifierComparisonsNB.py script and make sure it computes and outputs the same accuracy numbers for 20% holdout samples!!

# - Then think about how to integrate CMP rather than JUST ACCURACY into assessment

# Test naivebayes.Rearden_nbholdout() method on iris data
# compare feature selection (ranking) of PCA and NMIFS
naivebayes.Rearden_nbholdoutCV(iris.data,iris.target,0.2)
naivebayes.Rearden_nbholdoutCV(irisBinnedData,iris.target,0.2)
# user NMIFS3 to rank the 4 features of iris dataset
a,b = informationTheory.nmifs3(iris.data,iris.target,4)
a,b = informationTheory.nmifs3(irisBinnedData,iris.target,4)

# gamifs_fitness() method practice run
J = informationTheory.gamifs_fitness(naivebayes.Rearden_nbholdoutCV(irisBinnedData,iris.target,0.2),size(irisBinnedData,1),size(irisBinnedData,1))
# gamifs_fitness2() method practice run (user-specified function/model)
J = informationTheory.gamifs_fitness2(naivebayes.Rearden_nbholdoutCV, irisBinnedData,iris.target,array([1,1,1,1]))


#############################################################################
## CONSIDERATIONS for GAMIFS() for 2012Jul03
#############################################################################
# Need to consider HOW to incorporate "subset" information of population individual into GA algorithm
# Possibly reproduce NONLINEAR data set for testing GAMIFS integrity
# Weighted Random Number Selection Method
# -->>>> Need to consider a way to weight selection of a random integer for purposes of selecting individuals from our population in GA algo, which will be proportional to the fitness of that individual
## --->> Need to modify nbholdoutCV as well so that we can SPECIFY the train and test sets, so we can obtain fitness value on equal footing for ALL individuals

#############################################################################
# CREATED weighted_rand() method to produce weighted random selections!
#############################################################################
# Integrity Check for weighted_rand() method:
g=bMath.weighted_rand(array([0.25,0.6,0.15]),10000)
len(flatnonzero(g==0))/10000.0; # should be ~ w[0] = 0.25
len(flatnonzero(g==1))/10000.0; # should be ~ w[1] = 0.60
len(flatnonzero(g==2))/10000.0; # should be ~ w[2] = 0.15

# NEXT step is to integrate bMath.weighted_rand() method into GAMIFS() algorithm for weighted individual selection by fitness value prior to crossover execution

## **** Need to modify nbholdoutCV as well so that we can SPECIFY the train and test sets, so we can obtain fitness value on equal footing for ALL individuals


####
# Test functionality of naivebayes.Rearden_nbholdoutCVsets() method and informationTheory.gamifs_fitness3() methods to make sure they work as expected
# Use Iris data as test dataset to see if methods are working properly
testIDX = hstack((arange(10,20),arange(60,70),arange(110,120)));
trainIDX = ones((size(iris.data,0)));
trainIDX[testIDX] = 0;
trainIDX = flatnonzero(trainIDX);
testdata = irisBinnedData[testIDX,:];
traindata = irisBinnedData[trainIDX,:];

J = informationTheory.gamifs_fitness3(naivebayes.Rearden_nbholdoutCVsets,traindata,iris.target[trainIDX],testdata,iris.target[testIDX],array([1,1,1,1]))
# verified that informationTheory.gamifs_fitness3() method is working properly!!

## NEXT UP: determine how to implement crossover properly with weighted random selection that is driven by proportionality towards fitness value of individuals in GA population (for a given generation)


# FOR 2012Jul05 Work:
# Look at line 1278 in method GAMIFS() and maybe output ACTUAL fitness values and see what kind of proportional weighted random selection we expect
# Then continue on and implement the "weighted" random selection in gamifs being weight by the fitness value of EACH individual (model variation) in population
def gamifs_fitness3(func,xtrain,ytrain,xtest,ytest,subset,a=0.5):


### 2012Jul06
# From analysis of gamifs_fitness3() and using informationTheory.gamifs() execution on irisBinnedData and iris.targets, specifically the following code:
# informationTheory.gamifs(irisBinnedData,iris.target,naivebayes.Rearden_nbholdoutCVsets,0.2,100,200,0,1)
# Results seem to point out that for IRIS data that the first two components are the MOST important and lead to BETTER overall classification accuracy when they are used alone rather than the full 4 features

# Next task, integrate ROULETTE WHEEL or weighted random selection into gamifs()
# will be approx line 1291 now (where currently gamifs is using RANDINT, it needs to be replaced with bMath.weighted_rand() function, which is used in the following format:
# idx = weighted_rand(weights,2); # here two weighted random selections are made

# Crossover function was verified to be working properly on both 10 element and 4 element individuals (iris and digits modified data sets)

# ISSUE #1: crossover function allows for a [0,0,0,0] to be possible, for instance if [1,1,0,0] and [0,0,0,1] are crossed over, then we would get [0,0,0,0] very likely, so we need to somehow account for this in the gamifs() methods
# -----> CORRECTED [by using a boolean statement that checks sum(c1),sum(c2)!=0]

# Need to focus now on the mutation and tournament portions of gamifs()
# specifically, the mutop_nmifs() needs to take ONLY xtrain,ytrain arguments, since it is assessing whether or not (via nmifs) there should be a mutation of a feature, addition of a feature, or a removal of a feature, in a probabilistic fashion using p1, pi specified before (based on IEEE paper)

# ISSUE #2: now with the introduction of the mutation operator, and addition and removal operations as well, there are now MORE ways to possibly create an all "zero" array, which will lead to runtime errors, we need another boolean type check statement like before with the crossover method.
# -----> CORRECTED using usual BOOLEAN CHECK statement + WHILE-loop

# --> HOWEVER, now there seems to be a pass-by-reference issue with c1 changing as c1m is being computed by mutop_nmifs() method, need to FIX THIS!!!!!


### 2012Jul07
# Starting to look into "divide by zero" error originating from "zero array"
# Seems while-loop-boolean statement is NOT working as it should to make sure we get zero-only arrays for c1m and c2m (as well as c1 and c2).
# We have to remember that was is LIKELY going on is a "pass-by-reference" issue where it seems c1 is actually changing INTO a "zero-only" array after being called by and manipulated WITHIN mutop_nmifs() method

# Above issue seems to be solved (seemingly?). However, a new issue that arises is the TOURNAMENT portion is erroring out with c1m and c2m being NOT issued array values within the MUTATION segment, if the Jc1 > (1-delta)*Jmax conditions are not being met (which is what determines if there is a mutation or not in gamifs())
# RESOLUTION: simply add an ELSE statement portion to to the Jc1 > (1-delta)*Jmax statement to allow for c1m=c1 and c2m=c2 initializations for hamming distance comparisons in tournaments between pairing up parent/offsprings

# NEXT STEP and considerations, what exactly is the OUTPUT of gamifs():
# Is it a constantly REFINED population that was initialized by NMIFS3 algo, and then what is returned is genetically enhanced population of individuals, and this will yield (hopefully) some sort of converged list of features that seemingly OPTIMIZE the "performance" of the data and model (in this case NB)?
# We need to answer what gamifs() is supposed to yield before we can really complete and use this method for true advanced feature selection


### 2012Jul08
# Something to consider is the WHILE and FOR loops inside GAMIFS(), particularly near line 1292,1293 where we START the iterations thru choosing randomly selected individuals from the population (for current generation)

# Now that the WHILE and FOR loops have been correctly adjusted, we need to consider HOW J[]-arrays or fitness values array is going to store and be called as well as be returned at the END of the gamifs algorithm, since what will be returned is the pop,J objects:
# pop = contains SUBSETS for each of the initial individuals (after GAMIFS)
# J = contains fitness values for EACH subset of initial individuals
# TOGETHER, pop and J will yield the BEST fitness value for a given subset corresponding to the BEST feature subset for the model-data combination given

# NEXT STEP:
# Need to consider HOW to recompute and integrate updating J or fitness values will be done INSIDE WHILE loop, which is one of the last segments needed for gamifs() algorithm to work (see line 1292 of gamifs method)


### 2012Jul09

#############################################################################
# NONLINEAR AND Dataset: NMIFS3 (shortcomings) vs. GAMIFS (state space search)
#############################################################################

# create NonlinearANDdata for testing NMIFS against GAMIFS
import random;
lambd = 0.1; # create exponential distribution with mean = 1/lambd --> 10
# create RELEVANT features (from uniform distribution [-1,1]): 
relevFeats = ones((100,6));
for j in range(size(relevFeats,1)): # number of samples in each features
 for k in range(size(relevFeats,0)):
  relevFeats[k,j] = random.uniform(-1,1);
# create IRRELEVANT features (from exponential dist w/ mean = 10)
irrelFeats = ones((100,5));
for j in range(size(irrelFeats,1)): # number of samples in each features
 for k in range(size(irrelFeats,0)):
  irrelFeats[k,j] = random.expovariate(0.1);
# create REDUNDANT features (from last 3 features of RELEVANT set)
redunFeats = ones((100,3));
redunFeats = relevFeats[:,3:];

# concatenate to create full NonlinearANDdata set
nlin = hstack((irrelFeats,relevFeats,redunFeats))

# create target vector containing class labels for EACH sample (row):
nlinCls = zeros((size(nlin,0)));
for k in range(size(nlin,0)): # loop thru all rows, samples in nlin data array
 if nlin[k,6]*nlin[k,7]*nlin[k,8] > 0 and nlin[k,9]+nlin[k,10]+nlin[k,11] > 0:
  nlinCls[k] = 1; 
 if nlin[k,6]*nlin[k,7]*nlin[k,8] < 0 and nlin[k,9]+nlin[k,10]+nlin[k,11] < 0:
  nlinCls[k] = 2;

classifiableIDX = flatnonzero(nlinCls); # not all nlin data fit into class 1,2

# only keep nonzero elements of nlinCls (and corresponding in nlin)
temp = nlin[classifiableIDX,:];
tempCls = nlinCls[classifiableIDX];

# Determine feature ranking via NMIFS3() method:
featrankNLIN,subsetNLIN = informationTheory.nmifs3(temp,tempCls,size(nlin,1));
# results of above NMIFS3() execution are non-sensical b/c the features are of continuous type and the algorithm works best with discrete feature values

#############################################################################
# digitize nlin data to work within NMIFS3() feature ranking algo:
#############################################################################
temp2 = zeros_like(temp);
for k in range(size(temp,1)): # loop thru all features in temp (nlin)
 garbage,binEdges = histogram(temp[:,k],4); # use 5 bins to start within
 idx = digitize(temp[:,k],binEdges);
 temp2[:,k] = binEdges[idx-1];

# Re-run digitized version for feature ranking of nonlinearAND dataset
featrankNLIN,subsetNLIN = informationTheory.nmifs3(temp2,tempCls,size(temp2,1));
# RESULTS: definitely shows the shortcomings of NMIFS3() which CAN NOT find the true relevant features b/c their relevance is nonlinear, and incremental search and feature ranking algorithms like NMIFS3 can not find nonlinear relevance well
# As expected, the features f6,f7,f8 (the nonlinear elements) are NOT included in the top ranked features, since their importance is nonlinear, but it DID find the linear important features f9,f10,f11 easily and even ranked the duplicate f12,f13,f14 features highly

# Compare results and feature ranking of NMIFS3 vs. GAMIFS:
pop,J = informationTheory.gamifs(temp,tempCls,naivebayes.Rearden_nbholdoutCVsets)
pop[flatnonzero(J==max(J)),:]
mean(pop[flatnonzero(J==max(J)),:],axis=0)

# RESULTS point to still weighting heavily the f9,f10,f11 features, as well as capturing most if not ALL the f6,f7,f8 features (nonlinearly important features)

# Try optimizing the feature selection of GAMIFS
# optimize parameter values of a,theta,rho and numgen
reload(naivebayes),reload(informationTheory)
pop,J = informationTheory.gamifs(temp,tempCls,naivebayes.Rearden_nbholdoutCVsets,0.2,300,300,0.1,0.3)
pop[flatnonzero(J==max(J)),:]
mean(pop[flatnonzero(J==max(J)),:],axis=0)



### RESULTS are disappointing, NOT finding the right features!!!
# first thing to do is to make sure that the mutation-related methods are working as they should (compare against IEEE nmifs paper):
#1. mutop_nmifs
#2. add_nmifs
#3. remI_nmifs
#4. remR_nmifs


##### 2012Jul11
# ONE of the reasons why we might be getting shitty results is BECAUSE we are NOT using the right values for temp2!! since there are different values now for temp2 and temp w.r.t. HOW the CLASS labels were CREATED!!! Remember for the nonlinearAND dataset the features thems directly create the class labels!!
# PROPOSED SOLUTION: change random.uniform --> random.choice instead!!

# create RELEVANT features (from uniform distribution [-1,1]): 
relevFeats = ones((1000,6));
for j in range(size(relevFeats,1)): # number of samples in each features
 for k in range(size(relevFeats,0)):
  relevFeats[k,j] = random.choice([-1,-0.5,0,0.5,1]);
# create IRRELEVANT features (from exponential dist w/ mean = 10)
irrelFeats = ones((1000,5));
for j in range(size(irrelFeats,1)): # number of samples in each features
 for k in range(size(irrelFeats,0)):
  irrelFeats[k,j] = random.expovariate(0.1);
binEdges = array([0,10,20,30,40,100])
tempIrr = zeros_like(irrelFeats);
for k in range(size(irrelFeats,1)):
 idx = digitize(irrelFeats[:,k],binEdges);
 binValues = binEdges[idx];
 tempIrr[:,k] = binValues;
 
 
# create REDUNDANT features (from last 3 features of RELEVANT set)
redunFeats = ones((1000,3));
redunFeats = relevFeats[:,3:];

# concatenate to create full NonlinearANDdata set
nlin = hstack((tempIrr,relevFeats,redunFeats))

nlinCls = zeros((size(nlin,0)));
for k in range(size(nlin,0)): # loop thru all rows, samples in nlin data array
 if nlin[k,6]*nlin[k,7]*nlin[k,8] > 0 and nlin[k,9]+nlin[k,10]+nlin[k,11] > 0:
  nlinCls[k] = 1; 
 if nlin[k,6]*nlin[k,7]*nlin[k,8] < 0 and nlin[k,9]+nlin[k,10]+nlin[k,11] < 0:
  nlinCls[k] = 2;

classifiableIDX = flatnonzero(nlinCls); # not all nlin data fit into class 1,2

# only keep nonzero elements of nlinCls (and corresponding in nlin)
temp = nlin[classifiableIDX,:];
tempCls = nlinCls[classifiableIDX];

# NMIFS feature ranking
featrankNLIN,subsetNLIN = informationTheory.nmifs3(temp,tempCls,size(nlin,1));

# GAMIFS (using naivebayes package)
pop,J = informationTheory.gamifs(temp,tempCls,naivebayes.Rearden_nbholdoutCVsets)
pop[flatnonzero(J==max(J)),:]
mean(pop[flatnonzero(J==max(J)),:],axis=0)

# GAMIFS (using knn frpop,J = informationTheory.gamifs(temp,tempCls,bMath.Rearden_holdoutCVsets,'mnb')
array(pop[flatnonzero(J==max(J)),:],int)
mean(pop[flatnonzero(J==max(J)),:],axis=0)om sklearn)
pop,J = informationTheory.gamifs(temp,tempCls,bMath.Rearden_holdoutCVsets)
pop[flatnonzero(J==max(J)),:]
mean(pop[flatnonzero(J==max(J)),:],axis=0)

# TRY different classification algorithms and see which one works best with the GAMIFS for this nonlinearAND dataset. Specifically, we also need to work and optimize the bMath.Rearden_holdoutCVsets() method to accept and work with several model types, NOT just KNN
pop,J = informationTheory.gamifs(temp,tempCls,bMath.Rearden_holdoutCVsets,'knn')
array(pop[flatnonzero(J==max(J)),:],int)
mean(pop[flatnonzero(J==max(J)),:],axis=0)
# knn classifier has VERY good accuracy & outcome for feature selection!)

pop,J = informationTheory.gamifs(temp,tempCls,bMath.Rearden_holdoutCVsets,'log')
array(pop[flatnonzero(J==max(J)),:],int)
mean(pop[flatnonzero(J==max(J)),:],axis=0)

pop,J = informationTheory.gamifs(temp,tempCls,bMath.Rearden_holdoutCVsets,'svm')
array(pop[flatnonzero(J==max(J)),:],int)
mean(pop[flatnonzero(J==max(J)),:],axis=0)
# linearSVM has decent ACCURACY values, but feature selection is poor!

pop,J = informationTheory.gamifs(temp,tempCls,bMath.Rearden_holdoutCVsets,'gnb')
array(pop[flatnonzero(J==max(J)),:],int)
mean(pop[flatnonzero(J==max(J)),:],axis=0)

pop,J = informationTheory.gamifs(temp,tempCls,bMath.Rearden_holdoutCVsets,'mnb')
array(pop[flatnonzero(J==max(J)),:],int)
mean(pop[flatnonzero(J==max(J)),:],axis=0)

pop,J = informationTheory.gamifs(temp,tempCls,bMath.Rearden_holdoutCVsets,'nb')
array(pop[flatnonzero(J==max(J)),:],int)
mean(pop[flatnonzero(J==max(J)),:],axis=0)

pop,J = informationTheory.gamifs(temp,tempCls,bMath.Rearden_holdoutCVsets,'lda')
array(pop[flatnonzero(J==max(J)),:],int)
mean(pop[flatnonzero(J==max(J)),:],axis=0)

pop,J = informationTheory.gamifs(temp,tempCls,bMath.Rearden_holdoutCVsets,'qda')
array(pop[flatnonzero(J==max(J)),:],int)
mean(pop[flatnonzero(J==max(J)),:],axis=0)

pop,J = informationTheory.gamifs(temp,tempCls,bMath.Rearden_holdoutCVsets,'svmrbf',0.2,300,300)
array(pop[flatnonzero(J==max(J)),:],int)
mean(pop[flatnonzero(J==max(J)),:],axis=0)
# svmrbf was NEARLY PERFECT for feature selection AND accuracy!



### 2012Jul13

#############################################################################
# GAMIFS and NMIFS3 feature ranking and performance on HR+Acxiom Datasets
#############################################################################

# Here we will try various classification techniques, of which, the custom NaiveBayes model should prove to work best (or roughly the best overall)
# Consider looking at CPM not just ACC as overall accuracy/performance metric

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

# GAMIFS using svm-rbf classifier
pop,J = informationTheory.gamifs(hrAxUnn2,winnersNum,bMath.Rearden_holdoutCVsets,'svmrbf',0.2,300,300)
array(pop[flatnonzero(J==max(J)),:],int)
mean(pop[flatnonzero(J==max(J)),:],axis=0)
# look at a cutoff value for J --> J_cutoff = 0.41 (from histogram)
mean(pop[flatnonzero(J>0.41),:],axis=0)
idxSVC = flatnonzero(mean(pop[flatnonzero(J>0.41),:],axis=0) > 0.55)
hrAxfeats[idxSVC]

# GAMIFS using KNN classifier
pop,J = informationTheory.gamifs(hrAxUnn2,winnersNum,bMath.Rearden_holdoutCVsets,'knn',0.2,300,300)
array(pop[flatnonzero(J==max(J)),:],int)
mean(pop[flatnonzero(J==max(J)),:],axis=0)
# look at a cutoff value for J --> J_cutoff = 0.41 (from histogram)
mean(pop[flatnonzero(J>0.35),:],axis=0)
idxKNN = flatnonzero(mean(pop[flatnonzero(J>0.35),:],axis=0) > 0.60)
hrAxfeats[idxKNN]


# GAMIFS using Rearden_NB classifier
pop,J = informationTheory.gamifs(hrAxUnn2,winnersNum,bMath.Rearden_holdoutCVsets,'nb',0.2,300,300)
array(pop[flatnonzero(J==max(J)),:],int)
mean(pop[flatnonzero(J==max(J)),:],axis=0)
# look at a cutoff value for J --> J_cutoff = 0.41 (from histogram)
mean(pop[flatnonzero(J>0.344),:],axis=0)
idxNB = flatnonzero(mean(pop[flatnonzero(J>0.344),:],axis=0) > 0.54)
hrAxfeats[idxNB]

#### 2012Jul13
# Now beyond simply looking at accuracy, we need to consider WHICH of the classification techniques yeilds the BEST overall CPM, which take into account specificity, diversity, and accuracy (due to class imbalance)
# Also, the accuracies need to be assessed over 10 trials for testsets since there is some variation in accuracy based on WHICH test samples are used w.r.t. each classification model that can be used
knnIDX = zeros_like(pop[0,:]);
svcIDX = zeros_like(pop[0,:]);
nbIDX = zeros_like(pop[0,:]);

knnIDX[idxKNN] = 1;
svcIDX[idxSVC] = 1;
nbIDX[idxNB] = 1;

len(flatnonzero(knnIDX==svcIDX))/(1.0*len(knnIDX))
len(flatnonzero(svcIDX==nbIDX))/(1.0*len(knnIDX))
len(flatnonzero(nbIDX==knnIDX))/(1.0*len(knnIDX))

# working on create a CLASS for NB multinomial version within naivebayes pkg




