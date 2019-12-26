addpath(genpath(pwd));

%% Choose a suitable training set of linearly separable data xi ? R2 with 
% targets ti, where i ? {1, ...,N},N ? 100. For example, you can use 
% a linearly separable subset of T of assignment1.
[imgs, labels, imgsTest, labelsTest] = readMNISTauto();

% Extract 
imgs0 = digit(imgs, labels, 0);
imgs1 = digit(imgs, labels, 1);
imgs0test = digit(imgsTest, labelsTest, 0);
imgs1test = digit(imgsTest, labelsTest, 1);

count = 100;
countReserve = round(1.1*count)
[imgs0, fa0, s0] = digitSubsetProps(imgs0, countReserve);
[imgs1, fa1, s1] = digitSubsetProps(imgs1, countReserve);
[imgs0test, fa0test, s0test] = digitSubsetProps(imgs0test, countReserve);
[imgs1test, fa1test, s1test] = digitSubsetProps(imgs1test, countReserve);


% % Normalize (to range [0,1])
[fa0, fa1, fa0test, fa1test] = normal(fa0, fa1, fa0test, fa1test);
[s0, s1, s0test, s1test] = normal(s0, s1, s0test, s1test);

% make dataset linearly seperable
[M, I] = maxk(fa1, 3);

imgs0(:, :, I) = [];
fa0(I) = [];
s0(I) = [];

imgs1(:, :, I) = [];
fa1(I) = [];
s1(I) = [];

imgs0test(:, :, I) = [];
fa0test(I) = [];
s0test(I) = [];

imgs1test(:, :, I) = [];
fa1test(I) = [];
s1test(I) = [];

% Input (X) and labels (t)
X = [fa0' fa1'; s0' s1'];
Xtest = [fa0test', fa1test'; s0test', s1test'];
t = [-ones(size(fa0')) ones(size(fa1'))];

% Augmented (homogeneous)
h = [X; ones(1,size(X,2))];
htest = [Xtest; ones(1,size(Xtest,2))];


%% Plot the input vectors in R2 and visualize corresponding target values 
% (e.g. by using color).
figure;
title('Linearly separable 2-d dataset');
plot(X(1, 1:size(X, 2)/2), X(2, 1:size(X, 2)/2), 'o');
hold on;
plot(X(1, (size(X, 2)/2 + 1):end), X(2, (size(X, 2)/2 + 1):end), 'o');
xlabel('Filled area [-]');
xlabel('Solidity [-]');
legend('Zeros', 'Ones');