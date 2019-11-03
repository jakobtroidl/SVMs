addpath(genpath(pwd));

% Read the data using available packages for your programming language
% resp. simulation software
[imgs, labels, imgsTest, labelsTest] = readMNISTauto();

% Choose two classes (e.g., all images of digits ?0? and ?1?) for the
% following two-class classification task
imgs0 = digit(imgs, labels, 0);
imgs1 = digit(imgs, labels, 1);

% Select a small subset T( with less than 1000 images in total) of
% corresponding images from the MNIST training set
%
% Extract two suitable image features from the subset T. For example
% Matlab?s regionprop-function allows to calculate image features such
% as FilledArea and Solidity from binary images
count = 999;
if (~exist('fa0','var'))
    [imgs0, fa0, s0] = digitSubsetProps(imgs0,count);
end
if (~exist('fa1','var'))
    [imgs1, fa1, s1] = digitSubsetProps(imgs1,count);
end

% Plot the input vectors in R2 and visualize corresponding target values
% (e.g. by using color).
c = lines(2);
figure;
scatter(fa0, s0, 'filled', 'MarkerFaceColor', c(1,:));
hold on;
scatter(fa1, s1, 'filled', 'MarkerFaceColor', c(2,:));
hold off;
legend('Zeros', 'Ones');

