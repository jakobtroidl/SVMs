addpath(genpath('util'));
addpath('readmnist')

[X, t, Xtest, ttest, imgs, imgsTest] = readdata();

%% Part 2 - Bullet 1
sigma = 0.2;
C = Inf; % regularization parameter
kernelFunc = @(x1, x2)rbfkernel(x1, x2, sigma);

[alpha, w0] = trainSVM(X, t, kernelFunc, C);

% Classify test points
y = discriminant(alpha, w0, X, t, Xtest, kernelFunc);
figure;
plotdata(X, t, [0.75 0.75 0.75], [0.75 0.75 0.75]);
plotdata(Xtest, ttest);
title('Test data');
xlabel('Filled area [-]');
ylabel('Solidity [-]');
miscl = sign(y) ~= ttest;
scatter(Xtest(miscl,1), Xtest(miscl,2), 'bo');
plotboundary(alpha, w0, X, t, kernelFunc);
legend('Zeros training','Ones training','Zeros test', ...
    'Ones test','Test misclassifications');

% Plot boundary
figure;
plotdata(X,t);
plotboundary(alpha, w0, X, t, kernelFunc);
title(['Decision boundary for $\sigma$ = ' num2str(sigma) ' and C = ' num2str(C)]);
xlabel('Filled area [-]');
ylabel('Solidity [-]');

% Plot surface
figure;
plotdata(X,t);
plotboundary(alpha, w0, X, t, kernelFunc, 'surf');
title(['Discriminant function surface for $\sigma$ = ' num2str(sigma) ' [-] and C = ' num2str(C)]);
xlabel('Filled area [-]');
ylabel('Solidity [-]');

%% Part 2 - Bullet 2 - Try different values for sigma.
figure;
plotdata(X,t);
hold on;
sigmaRange = logspace(-1, 1, 5);
colors = lines(numel(sigmaRange));
legendstrs = cell(2+numel(sigmaRange),1);
legendstrs{1} = 'Zeros';
legendstrs{2} = 'Ones';
for i = 1:numel(sigmaRange)
    sigma = sigmaRange(i);
    kernelFunc = @(x1, x2)rbfkernel(x1, x2, sigma);
    [alpha, w0] = trainSVM(X, t, kernelFunc, C);
    plotboundary(alpha, w0, X, t, kernelFunc, 'nomargins', colors(i,:));
    legendstrs{2+i} = ['$\sigma$ = ' num2str(sigma)];
end
title(['Decision boundary for various \sigma values [-] and C = ' num2str(C)]);
xlabel('Filled area [-]');
ylabel('Solidity [-]');
legend(legendstrs,'Interpreter','latex');
text(0, 0, 'Higher values of $\sigma$ correspond to straighter decision boundaries.')
