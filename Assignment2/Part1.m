addpath(genpath('util'));
addpath('readmnist')

[X, t, Xtest, ttest, imgs, imgsTest] = readdata();
f1 = figure;
plotdata(X, t);
legend('Zeros','Ones');
title('Input data');

[alpha, w0] = trainSVM(X, t);

%% Classify test points
y = discriminant(alpha, w0, X, t, Xtest);
figure;
plotdata(X, t, [0.75 0.75 0.75], [0.75 0.75 0.75]);
plotdata(Xtest, sign(y));
miscl = sign(y) ~= ttest;
scatter(Xtest(miscl,1), Xtest(miscl,2), 'bo');
legend('Zeros training','Ones training','Zeros test', ...
    'Ones test','Test misclassifications', 'Decision boundary', ...
    '|Discriminant|=1', 'Support vectors');
plotboundary(alpha, w0, X, t);


%% Plot boundary
w = (alpha .* t)' * X;
figure;
plotdata(X,t);
plotboundary(alpha, w0, X, t);

%% Plot surface
figure;
plotdata(X,t);
plotboundary(alpha, w0, X, t, [], 'surf');
legend('Zeros', 'Ones', 'Decision boundary', '|Discriminant|=1', ...
    'Support vectors', 'Location', 'best');