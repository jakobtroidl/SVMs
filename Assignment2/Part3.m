addpath(genpath('util'));
addpath('readmnist')

%% Generate at least M=150 disjoint training sets Tk
% by selecting images from the MNIST training set, where each Tk consists
% of not more than N=70 images
M = 150; %  # of training sets
N = 70; %  # of images per training set
train = cell(M,1);
trainL = cell(M,1); % labels
for k = 1:M
    [imgs, labels] = readmnist('training', N, (k-1)*N+1);
    train{k} = im2data(imgs);
    trainL{k} = labels;
end
[test, testL] = readmnist('test');

%% For all k train the SVM and the perceptron (see assignment1) using Tk
% and calculate the test error rate Rk on the MNIST test set for both
% models (using all available images of the two respective classes in the
% testset)
for k = 1:M
    %% SVM
    kernel = @(x1, x2)rbfkernel(x1, x2, sigma);
    C = 50;
    [alpha, w0] = trainSVM(train{k}, trainL{k}, kernel, C);
    
    %% Perceptron
    maxIts = 10000;
    mode = 'batch';
    w = percTrain(train{k}, trainL{k}, maxIts, mode);
    
    %% Comparison
    ySVM = discriminant(alpha, w0, train{k}, trainL{k}, test, kernel);
    yPerc = perc(w, test);
end