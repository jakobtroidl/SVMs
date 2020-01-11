addpath(genpath('util'));
addpath('readmnist')

% Generate at least M=150 disjoint training sets Tk
% by selecting images from the MNIST training set, where each Tk consists
% of not more than N=70 images

M = 3; %  # of training sets
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

%% For all k train the SVM and the perceptron (see assignment1) using Tk
% and calculate the test error rate Rk on the MNIST test set for both
% models (using all available images of the two respective classes in the
% testset)

 %% Compare average error of Linear SVM (no kernel, no slack variable) with perceptron
tic
[errorLinearSVM, ~] = evaluate(train, trainL, imgs_test_01, labels_test_01, 'EvalOnTestSet');
errorPerc = evaluatePercs(train, trainL, imgs_test_01, labels_test_01);
toc

% calculate the average error for Perc and SVM over all M trained SVMs and Perc's 
avg_error_svm = (sum(errorLinearSVM, 3) / M) * 100;
avg_error_perc = (sum(errorPerc, 3) / M) * 100;

% plot results
X = categorical({'Avg error SVM','Avg error Perc'});
Y = [avg_error_svm avg_error_perc];

figure, bar(X,Y, 0.4)
title('Average error comparison of Perceptron and SVM (linear kernel, no slack variables)')
ylabel('% of misclassified samples')

%% Explore 2D parameter space of (sigma, C) of SVM and plot average errors. 
%Analyse the effect of changing C or ? on the average test error rate Ravg of the SVM.

% C = [0.5 1 3 5 7 10 20 50 Inf];
%sigma = [0.5 1 2 3 5 7 10 20 30];
C = [logspace(-1, 2, 2) * 5, Inf];
sigma = logspace(-3, 1, 3) * 5;

% C = [0.5 1 3];
% sigma = [1 2 3];

tic
[errorSVM, numOfSupportVecs] = evaluate(train, trainL, imgs_test_01, labels_test_01, 'EvalOnTestSet', C, sigma);
toc 

avg_error_svm = (sum(errorSVM, 3) ./ M) * 100;

figure, bar3(avg_error_svm);
set(gca,'XtickLabel', sigma);
set(gca,'YtickLabel', C);
ylabel('regularization param. C'), xlabel('sigma'), zlabel('Avg. error in %')
title('Avg. error over the 2D parameter space')

avg_numofSupportVecs = (sum(numOfSupportVecs, 3) ./ M);
figure, bar3(avg_numofSupportVecs);
set(gca,'YtickLabel', C);
set(gca,'XtickLabel', sigma);
xlabel('regularization param. C'), ylabel('sigma'), zlabel('Avg. # of support vectors')
title('Avg. # of support vectors over the 2D parameter space')


%% Analyse the effect of changing C or sigma on the average training error 
% (proportion of false classifications in the training set Tk after training with Tk).

tic
[avgTrainErrorSVM, ~] = evaluate(train, trainL, imgs_test_01, labels_test_01, 'EvalOnTrainSet', C, sigma);
toc 

avg_error_svm = (sum(avgTrainErrorSVM, 3) ./ M) * 100;
figure, bar3(avg_error_svm);
set(gca,'XtickLabel', C);
set(gca,'YtickLabel', sigma);
xlabel('regularization param. C'), ylabel('sigma'), zlabel('Avg. error in %')
title('Avg. training error over the 2D parameter space')

%% Analyse the relation of the average number of support vectors and the average test error rate Ravg of the SVM.


