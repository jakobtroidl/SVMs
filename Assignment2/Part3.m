addpath(genpath('util'));
addpath('readmnist')

% Generate at least M=150 disjoint training sets Tk
% by selecting images from the MNIST training set, where each Tk consists
% of not more than N=70 images

M = 150; %  # of training sets
N = 70; %  # of images per training set
train = cell(M,1);
trainL = cell(M,1); % labels

[imgs, labels] = readmnist('training');
idx = labels < 2; % only take 0 and 1 digits into account
imgs_01 = imgs(:, :, idx); % filter 0 and 1 images
labels_01 = labels(idx, :); % filter 0 and 1 labels
labels_01 = (labels_01 * 2) - 1; % scale labels to [-1; 1], to use them in trainSVM

curr_start = 1;
for k = 1:M
    curr_end = curr_start + N - 1;
    train{k} = im2data(imgs_01(:, :, curr_start : curr_end));
    trainL{k} = labels_01(curr_start : curr_end, :);
    curr_start = curr_start + N;
end

% extract the test dataset
[test, testL] = readmnist('test');
idx = testL < 2; % only take 0 and 1 digits into account
imgs_test_01 = im2data(test(:, :, idx)); % filter 0 and 1 images
labels_test_01 = testL(idx, :); % filter 0 and 1 labels
labels_test_01 = (labels_test_01 * 2) - 1; % scale labels to [-1; 1]

testSize = size(imgs_test_01, 2);

%% For all k train the SVM and the perceptron (see assignment1) using Tk
% and calculate the test error rate Rk on the MNIST test set for both
% models (using all available images of the two respective classes in the
% testset)

incorr_Perc = zeros(M, 1);
incorr_SVM = zeros(M, 1);

for k = 1:M
    %% SVM
    % kernel = @(x1, x2)rbfkernel(x1, x2, sigma);
    % C = 50;
    % train a linear SVM
    [alpha, w0] = trainSVM(train{k}', trainL{k});
    
    %% Perceptron
    maxIts = 10000;
    mode = 'batch';
    w = percTrain(train{k}, trainL{k}', maxIts, mode);
    
    %% Comparison
    ySVM = discriminant(alpha, w0, train{k}', trainL{k}, imgs_test_01');
    incorr_SVM(k) = sum(sign(ySVM) ~= labels_test_01) / testSize;
   
    yPerc = perc(w, imgs_test_01);
    incorr_Perc(k) = sum(yPerc' ~= labels_test_01) / testSize;
    
end

% calculate the average error for Perc and SVM over all M trained SVMs and Perc's 
avg_error_svm = (sum(incorr_SVM) / M) * 100;
avg_error_perc = (sum(incorr_Perc) / M) * 100;

% plot results
X = categorical({'Avg error SVM','Avg error Perc'});
Y = [avg_error_svm avg_error_perc];
bar(X,Y, 0.4)
title('Average error comparison of Perceptron and SVM (linear kernel, no slack variables)')
ylabel('% of misclassified samples')