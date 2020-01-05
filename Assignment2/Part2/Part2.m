addpath(genpath('util'));
addpath('readmnist')

[X, t, Xtest, ttest, imgs, imgsTest] = readdata();

%% Part 2 - Bullet 1
sigma = 0.2;
C = 0.1; % regularization parameter
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
title(['Decision boundary for $\sigma$ = ' num2str(sigma) ' and C = ' num2str(C)]);
xlabel('Filled area [-]');
ylabel('Solidity [-]');

% Plot surface
figure;
plotdata(X,t);
plotboundary2(alpha, w0, X, t, 'surf', kernelFunc);
title(['Discriminant function surface for $\sigma$ = ' num2str(sigma) ' [-] and C = ' num2str(C)]);
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
title(['Decision boundary for $\sigma$ = ' sigmaString(1:end-2) ' [-] and C = ' num2str(C)]);
xlabel('Filled area [-]');
ylabel('Solidity [-]');
legend('Zeros','Ones', 'Decision boundaries');
text(0, 0, 'Higher values of $\sigma$ correspond to straighter decision boundaries.')
