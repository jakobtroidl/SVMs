addpath(genpath(pwd));

% Read the data using available packages for your programming language
% resp. simulation software
[imgs, labels, imgsTest, labelsTest] = readMNISTauto();

% Choose two classes (e.g., all images of digits '0' and '1') for the
% following two-class classification task
imgs0 = digit(imgs, labels, 0);
imgs1 = digit(imgs, labels, 1);

% Select a small subset T (with less than 1000 images in total) of
% corresponding images from the MNIST training set
%
% Extract two suitable image features from the subset T. For example
% Matlab?s regionprop-function allows to calculate image features such
% as FilledArea and Solidity from binary images
count = 200;
[imgs0, fa0, s0] = digitSubsetProps(imgs0,count);
[imgs1, fa1, s1] = digitSubsetProps(imgs1,count);

% Normalize (to range [0,1])
[fa0, fa1] = normal(fa0, fa1);
[s0, s1] = normal(s0, s1);

% Input (X) and labels (t)
X = [fa0' fa1'; s0' s1'];
t = [-ones(size(fa0')) ones(size(fa1'))];

% Augmented (homogeneous)
h = [X; ones(1,size(X,2))];
% Plot the input vectors in R2 and visualize corresponding target values
% (e.g. by using color).
figure;
plotPoints(X, t);
hold on;

maxIts = 10000;

disp(['Training online with maxIts=' num2str(maxIts) '...']);
tic;
wonline = percTrain(h, t, maxIts, true);
toc;
yonline = perc(wonline, h);
acconline = sum(yonline == t) / numel(t);

disp(['Training batch with maxIts=' num2str(maxIts) '...']);
tic;
wbatch = percTrain(h, t, maxIts, false);
toc;
ybatch = perc(wbatch, h);
accbatch = sum(ybatch == t) / numel(t);

plotBoundary(wonline, true, '--k');
plotBoundary(wbatch, true, '-k');
legend('Zeros', 'Ones', ['Online (acc=' num2str(acconline) ')'], ['Batch (acc=' num2str(accbatch) ')']);
title('Augmented');
hold off;



f = transFts(X(1,:), X(2,:));

figure;
plotPoints(X, t);
hold on;

disp(['Training transformed online with maxIts=' num2str(maxIts) '...']);
tic;
wonline2 = percTrain(f, t, maxIts, true);
toc;
yonline2 = perc(wonline2, f);
acconline2 = sum(yonline2 == t) / numel(t);

disp(['Training transformed batch with maxIts=' num2str(maxIts) '...']);
tic;
wbatch2 = percTrain(f, t, maxIts, false);
toc;
ybatch2 = perc(wbatch2, f);
accbatch2 = sum(ybatch2 == t) / numel(t);

plotBoundary(wonline2, false, '--k');
plotBoundary(wbatch2, false, '-k');
legend('Zeros', 'Ones', ['Online (acc=' num2str(acconline2) ')'], ['Batch (acc=' num2str(accbatch2) ')']);
title('Transformed');
hold off;