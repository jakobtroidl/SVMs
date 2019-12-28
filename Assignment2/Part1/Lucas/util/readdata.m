function [X, t, Xtest, imgs, imgsTest] = readdata()

path = 'util/';
pXin = [path 'Xin.mat'];
ptin = [path 'tin.mat'];
pXTin = [path 'XTin.mat'];
pimin = [path 'imin.mat'];
pimTin = [path 'imTin.mat'];

if isfile(pXin) && isfile(ptin) && isfile(pXTin) ...
        && isfile(pimin) && isfile(pimTin)
    
    X = load(pXin).Xin;
    t = load(ptin).tin;
    Xtest = load(pXTin).XTin;
    imgs = load(pimin).imin;
    imgsTest = load(pimTin).imTin;
    return;
    
end

%% Read the data using available packages for your programming language
% resp. simulation software

readDigits = 5000; % Max: 60000
disp(['Loading training set (' num2str(readDigits) ' images)...']);
tic;
imgFile = "train-images.idx3-ubyte";
labelFile = "train-labels.idx1-ubyte";
offset = 0;
[imgs, labels] = readMNIST(imgFile, labelFile, readDigits, offset);
toc;

readDigits = 5000; % Max: 10000
disp(['Loading test set (' num2str(readDigits) ' images)...']);
tic;
imgFile = "t10k-images.idx3-ubyte";
labelFile = "t10k-labels.idx1-ubyte";
offset = 0;
[imgsTest, labelsTest] = readMNIST(imgFile, labelFile, readDigits, offset);
toc;

%% Choose two classes (e.g., all images of digits '0' and '1') for the
% following two-class classification task
imgs0 = digit(imgs, labels, 0);
imgs1 = digit(imgs, labels, 1);
imgs0test = digit(imgsTest, labelsTest, 0);
imgs1test = digit(imgsTest, labelsTest, 1);

%% Select a small subset T (with less than 1000 images in total) of
% corresponding images from the MNIST training set
%
% Extract two suitable image features from the subset T. For example
% Matlab's regionprop-function allows to calculate image features such
% as FilledArea and Solidity from binary images
count = 200;
[imgs0, fa0, s0] = digitSubsetProps(imgs0,count);
[imgs1, fa1, s1] = digitSubsetProps(imgs1,count);
[imgs0test, fa0test, s0test] = digitSubsetProps(imgs0test,count);
[imgs1test, fa1test, s1test] = digitSubsetProps(imgs1test,count);

% Normalize (to range [0,1])
[fa0, fa1, fa0test, fa1test] = normal(fa0, fa1, fa0test, fa1test);
[s0, s1, s0test, s1test] = normal(s0, s1, s0test, s1test);

% Input (X) and labels (t)
X = [fa0 s0; fa1 s1];
Xtest = [fa0test s0test; fa1test s1test];
t = [-ones(size(fa0)); ones(size(fa1))];

imgs = cat(3,imgs0,imgs1);
imgsTest = cat(3,imgs0test,imgs1test);

%% Let user select some points to remove from training set
f = figure;
plotdata(X,t);
title('Select points to remove');
[x,y] = ginput;
close(f);
dis = pdist2([x,y],X,'euclidean');
[~,indexes] = min(dis,[],2);

X(indexes,:) = [];
t(indexes) = [];
imgs(:,:,indexes) = [];

%% Save data to files
Xin = X;
tin = t;
XTin = Xtest;
imin = imgs;
imTin = imgsTest;

save(pXin,'Xin');
save(ptin,'tin');
save(pXTin,'XTin');
save(pimin,'imin');
save(pimTin,'imTin');

end