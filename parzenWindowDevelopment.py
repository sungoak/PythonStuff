# Parzen_Window_development

# construct first gauss1 gaussian data set of 4 features
a = 2;
part0 = randn(1000,4)
part1 = randn(1000,4) + a*array([1,1/2.0,1/3.0,1/4.0]);
part2 = randn(1000,4) + 2*a*array([1,1/2.0,1/3.0,1/4.0]);
part3 = randn(1000,4) + 3*a*array([1,1/2.0,1/3.0,1/4.0]);

gauss1 = vstack((part0,part1,part2,part3))

# now construct gauss2 which is simply gauss1 features + redundant vectors + noise
gauss2 = hstack((gauss1,2*gauss1+randn(4000,4)))

gaussTargets = stack((ones(1000),2*ones(1000),3*ones(1000),4*ones(1000)))
