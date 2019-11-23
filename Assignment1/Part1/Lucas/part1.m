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
% Matlab's regionprop-function allows to calculate image features such
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

maxIts = 100;

disp(['Training online with maxIts=' num2str(maxIts) '...']);
tic;
wonline = percTrain(h, t, maxIts, true);
toc;
yonline = perc(wonline, h);

disp(['Training batch with maxIts=' num2str(maxIts) '...']);
tic;
wbatch = percTrain(h, t, maxIts, false);
toc;
ybatch = perc(wbatch, h);

percPlot(X, t, wonline, wbatch, yonline, ybatch, '2d');

% Use  the  feature  transformation
% ?(x) : (x1,x2)?(1,x1,x2,x1^2,x2^2,x1*x2) and plot the data and decision 
% boundary in the original data space R2 (see e.g. Figure 1) after
% training. Hint: Sample the relevant region of the input space using a
% meshgrid. Compute y = w' * ?(x) for all grid points and use a
% contour-function or a surface-plot to visualize the approximation of the
% curvey= 0.
f = transFts(X(1,:), X(2,:));

disp(['Training transformed online with maxIts=' num2str(maxIts) '...']);
tic;
wonline2 = percTrain(f, t, maxIts, true);
toc;
yonline2 = perc(wonline2, f);

disp(['Training transformed batch with maxIts=' num2str(maxIts) '...']);
tic;
wbatch2 = percTrain(f, t, maxIts, false);
toc;
ybatch2 = perc(wbatch2, f);

percPlot(X, t, wonline2, wbatch2, yonline2, ybatch2, 'nd');

% Train the perceptron using all 28?28 = 784 pixels of MNIST images of T as
% input, resulting in augmented input vectors with dimensionality of m=785
% and visualize w1,...,wm as a 28?28 gray-scale image (see Figure 2).
imX = im2data(cat(3,imgs0,imgs1));

disp(['Training pixels online with maxIts=' num2str(maxIts) '...']);
tic;
wonline3 = percTrain(imX, t, maxIts, true);
toc;
yonline3 = perc(wonline3, imX);

disp(['Training pixels batch with maxIts=' num2str(maxIts) '...']);
tic;
wbatch3 = percTrain(imX, t, maxIts, false);
toc;
ybatch3 = perc(wbatch3, imX);

percPlot(X, t, wonline3, wbatch3, yonline3, ybatch3, 'image');