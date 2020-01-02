clear all;
close all;
addpath(genpath(pwd));

%% Choose a suitable training set of linearly separable data xi ? R2 with 
% targets ti, where i ? {1, ...,N},N >= 100. For example, you can use 
% a linearly separable subset of T of assignment1.
[imgs, labels, imgsTest, labelsTest] = readMNISTauto();

% Extract 
imgs0 = digit(imgs, labels, 0);
imgs1 = digit(imgs, labels, 1);
imgs0test = digit(imgsTest, labelsTest, 0);
imgs1test = digit(imgsTest, labelsTest, 1);

count = 200/2;
countReserve = round(1.1*count);
[imgs0, fa0, s0] = digitSubsetProps(imgs0, countReserve);
[imgs1, fa1, s1] = digitSubsetProps(imgs1, countReserve);
[imgs0test, fa0test, s0test] = digitSubsetProps(imgs0test, countReserve);
[imgs1test, fa1test, s1test] = digitSubsetProps(imgs1test, countReserve);

% Normalize (to range [0,1])
[fa0, fa1, fa0test, fa1test] = normal(fa0, fa1, fa0test, fa1test);
[s0, s1, s0test, s1test] = normal(s0, s1, s0test, s1test);

% Input (X) and labels (t)
X = [fa0' fa1'; s0' s1'];
Xtest = [fa0test', fa1test'; s0test', s1test'];
t = [-ones(size(fa0')) ones(size(fa1'))];
ttest = [-ones(size(fa0')) ones(size(fa1'))];

% Augmented (homogeneous)
h = [X; ones(1,size(X,2))];
htest = [Xtest; ones(1,size(Xtest,2))];

maxIts = 1000;
disp(['Training batch with maxIts=' num2str(maxIts) '...']);
tic;
wbatch = percTrain(h, t, maxIts, false);

toc;
% Find falsely classified samples
ybatch = perc(wbatch, h);
test_errors = ybatch ~= t;
falselyClassifiedIDs = find(test_errors);

% Remove falsely classified samples from X and t
for i = size(falselyClassifiedIDs, 2):-1:1
    X(:, falselyClassifiedIDs(i)) = [];
    t(:, falselyClassifiedIDs(i)) = [];
end

% Remove redundant samples to get desired count
for i = 1:(sum(t(:) == -1) - count)
    X(:, 1) = [];
    t(:, 1) = [];
end
for i = 1:(sum(t(:) == 1) - count)
    X(:, end) = [];
    t(:, end) = [];
end

%% Part 2
X = X';t = t';Xtest = Xtest';ttest = ttest';
sigma = 5;
kernelFunc = @(x1, x2)rbfkernel(x1, x2, sigma);

[alpha, w0] = trainSVM2(X, t, kernelFunc);

% Plot support vectors
figure;
plotdata(X, t);
title('Input data');
sv = X(alpha > 110, :);
scatter(sv(:,1),sv(:,2),'ko');
legend('Zeros','Ones', 'Support vectors');

% Classify test points
y = discriminant2(alpha, w0, X, t, Xtest, kernelFunc);
figure;
plotdata(X, t, [0.75 0.75 0.75], [0.75 0.75 0.75]);
plotdata(Xtest, sign(y - mean(y)));
title('Test data');
miscl = sign(y - mean(y)) ~= ttest;
scatter(Xtest(miscl,1), Xtest(miscl,2), 'ko');
legend('Zeros training','Ones training','Zeros test', ...
    'Ones test','Test misclassifications');


% Plot boundary
w = (alpha .* t)' * X;
figure;
plotdata(X,t);
plotboundary(alpha, w0, X, t);
title('Decision boundary');

% Plot surface
figure;
plotdata(X,t);
plotboundary(alpha, w0, X, t, true);
title('Discriminant function surface');

