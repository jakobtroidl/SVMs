addpath(genpath('util'));
addpath('readmnist')

[X, t, Xtest, ttest, imgs, imgsTest] = readdata();
f1 = figure;
plotdata(X, t);
legend('Zeros','Ones');
title('Linearly separable training set');
%saveas(gcf,['figures/Training set.png']);

[alpha, w0] = trainSVM(X, t, Inf);

%% Classify test points
y = discriminant(alpha, w0, X, t, Xtest);
figure;
plotdata(X, t, [0.75 0.75 0.75], [0.75 0.75 0.75]);
plotdata(Xtest, sign(y));
miscl = sign(y) ~= ttest;
scatter(Xtest(miscl,1), Xtest(miscl,2), 'bo');
plotboundary(alpha, w0, X, t);
legend('Zeros training','Ones training','Zeros test', ...
    'Ones test','Test misclassifications', 'Decision boundary', ...
    '|Discriminant|=1', 'Support vectors');
%saveas(gcf,['figures/SVM test.png']);
title('Testing SVM on another N=400 long set.');

%% Plot boundary
w = (alpha .* t)' * X;
figure;
plotdata(X,t);
plotboundary(alpha, w0, X, t);
legend('Zeros', 'Ones', 'Decision boundary', '|Discriminant|=1', ...
    'Support vectors', 'Location', 'best');
%saveas(gcf,['figures/SVM train.png']);
title('Trained SVM');

%% Plot surface
figure;
plotdata(X,t);
plotboundary(alpha, w0, X, t, [], 'surf');
legend('Zeros', 'Ones', 'Decision boundary', '|Discriminant|=1', ...
    'Support vectors', 'Location', 'best');
%saveas(gcf,['figures/SVM train surface.png']);
title('Trained SVM with discriminant function surface');