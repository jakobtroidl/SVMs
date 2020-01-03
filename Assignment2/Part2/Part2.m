clear all;
close all;
addpath(genpath(pwd));
set(groot, 'defaultTextInterpreter','latex');

%% Choose a suitable training set of linearly separable data xi ? R2 with 
% targets ti, where i ? {1, ...,N},N >= 100. For example, you can use 
% a linearly separable subset of T of assignment1.
[imgs, labels, imgsTest, labelsTest] = readMNISTauto();

% Extract 
imgs0 = digit(imgs, labels, 0);
imgs1 = digit(imgs, labels, 1);
imgs0test = digit(imgsTest, labelsTest, 0);
imgs1test = digit(imgsTest, labelsTest, 1);

count = 200;
countReserve = 200; %round(1.1*count);
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

%% Part 2 - Bullet 1
X = X';t = t';Xtest = Xtest';ttest = ttest';
sigma = 0.2;
C = 50; % regularization parameter
kernelFunc = @(x1, x2)rbfkernel(x1, x2, sigma);

[alpha, w0] = trainSVM2(X, t, kernelFunc, C);

% Classify test points
y = discriminant2(alpha, w0, X, t, Xtest, kernelFunc);
figure;
plotdata(X, t, [0.75 0.75 0.75], [0.75 0.75 0.75]);
plotdata(Xtest, sign(y));
title('Test data');
xlabel('Filled area [-]');
ylabel('Solidity [-]');
miscl = sign(y) ~= ttest;
scatter(Xtest(miscl,1), Xtest(miscl,2), 'ko');
legend('Zeros training','Ones training','Zeros test', ...
    'Ones test','Test misclassifications');

% Plot boundary
figure;
plotdata(X,t);
plotboundary2(alpha, w0, X, t, '', kernelFunc);
title(['Decision boundary for $\sigma$ = ' num2str(sigma)]);
xlabel('Filled area [-]');
ylabel('Solidity [-]');

% Plot surface
figure;
plotdata(X,t);
plotboundary2(alpha, w0, X, t, 'surf', kernelFunc);
title(['Discriminant function surface for $\sigma$ = ' num2str(sigma) ' [-]']);
xlabel('Filled area [-]');
ylabel('Solidity [-]');

%% Part 2 - Bullet 2 - Try different values for sigma.
figure;
plotdata(X,t);
hold on;
sigmaRange = logspace(-1, 1, 5);
for sigma = sigmaRange
    kernelFunc = @(x1, x2)rbfkernel(x1, x2, sigma);
    [alpha, w0] = trainSVM2(X, t, kernelFunc, C);
    plotboundary2(alpha, w0, X, t, 'noMargins', kernelFunc);
end
sigmaString = [];
for i = 1:size(sigmaRange, 2)
    sigmaString = [sigmaString num2str(sigmaRange(i), 2) ', '];
end
title(['Decision boundary for $\sigma$ = ' sigmaString(1:end-2) ' [-]']);
xlabel('Filled area [-]');
ylabel('Solidity [-]');
legend('Zeros','Ones', 'Decision boundaries');
text(0, 0, 'Higher values of $\sigma$ correspond to straighter decision boundaries.')
