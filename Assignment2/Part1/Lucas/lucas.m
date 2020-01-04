addpath(genpath('util'));
addpath('readmnist')

[X, t, Xtest, ttest, imgs, imgsTest] = readdata();
f1 = figure;
plotdata(X, t);
legend('Zeros','Ones');
title('Input data');

[alpha, w0] = trainSVM(X, t);

%% Plot support vectors
sv = X(alpha > 0.00001, :);
scatter(sv(:,1),sv(:,2),'ko');

%% Classify test points
y = discriminant(alpha, w0, X, t, Xtest);
figure;
plotdata(X, t, [0.75 0.75 0.75], [0.75 0.75 0.75]);
plotdata(Xtest, sign(y));
miscl = sign(y) ~= ttest;
scatter(Xtest(miscl,1), Xtest(miscl,2), 'bo');
legend('Zeros training','Ones training','Zeros test', ...
    'Ones test','Test misclassifications');
plotboundary(alpha, w0, X, t);


%% Plot boundary
w = (alpha .* t)' * X;
figure;
plotdata(X,t);
plotboundary(alpha, w0, X, t);

%% Plot surface
figure;
plotdata(X,t);
plotboundary(alpha, w0, X, t, true);