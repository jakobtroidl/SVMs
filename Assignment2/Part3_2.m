addpath(genpath('util'));
addpath('readmnist')

% Generate at least M=150 disjoint training sets Tk
% by selecting images from the MNIST training set, where each Tk consists
% of not more than N=70 images

M = 150; %  # of training sets
N = 70; %  # of images per training set
train = zeros(M*N, 2);
trainL = zeros(M, 1); % labels

%% Get only 0 and 1
[imgs, labels] = readmnist('training');
imgs0 = digit(imgs,labels,0);
imgs1 = digit(imgs,labels,1);

%% Get M*N images
imgs = cat(3, ...
    imgs0(:,:, 1:floor(M*N/2)), ...
    imgs1(:,:, 1:M*N - floor(M*N/2)));

%% Transform into data
X = im2data(imgs);
t = [-ones(floor(M*N/2),1); ones(M*N - floor(M*N/2),1)];

%% Permutate randomly
idx = randperm(M*N);
X = X(:,idx);
t = t(idx);

%% Cross validation
Cs = [50];
sigmas = [10];
errors = crossvalidate(X, t, Cs, sigmas, M, N);
errorsavg = mean(errors,3);