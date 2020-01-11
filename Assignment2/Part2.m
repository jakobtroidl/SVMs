addpath(genpath('util'));
addpath('readmnist')

[X, t, Xtest, ttest, imgs, imgsTest] = readdata();

%% Part 2 - Bullet 1
sigma = 0.2; % RBF kernel parameter
C = Inf; % regularization parameter

% Create handle with determined sigma parameter
kernelFunc = @(x1, x2)rbfkernel(x1, x2, sigma);

[alpha, w0] = trainSVM(X, t, kernelFunc, C);

% Classify test points
y = discriminant(alpha, w0, X, t, Xtest, kernelFunc);
figure;
plotdata(X, t, [0.75 0.75 0.75], [0.75 0.75 0.75]);
plotdata(Xtest, ttest);
miscl = sign(y) ~= ttest;
scatter(Xtest(miscl,1), Xtest(miscl,2), 'bo');
plotboundary(alpha, w0, X, t, kernelFunc);
legend('Zeros training','Ones training','Zeros test', ...
    'Ones test','Test misclassifications', 'Decision boundary', ...
    '|Discriminant|=1', 'Support vectors');
title(['Decision boundary \& test data for \sigma = ' num2str(sigma), ...
    ' and C = ' num2str(C)]);

% Plot boundary
figure;
plotdata(X,t);
plotboundary(alpha, w0, X, t, kernelFunc);
legend('Zeros','Ones', 'Decision boundary', '|Discriminant|=1', ...
    'Support vectors');
saveas(gcf,['figures/Decision boundary.png']);
title(['Decision boundary for \sigma = ' num2str(sigma) ' and C = ', ...
    num2str(C)]);

% Plot surface
figure;
plotdata(X,t);
plotboundary(alpha, w0, X, t, kernelFunc, 'surf');
legend('Zeros', 'Ones', 'Decision boundary', '|Discriminant|=1', ...
    'Support vectors', 'Location', 'best');
saveas(gcf,['figures/Discriminant function surface.png']);
title(['Discriminant function surface for \sigma = ' num2str(sigma), ...
    ' and C = ' num2str(C)]);

%% Part 2 - Bullet 2 - Try different values for sigma.
figure;
plotdata(X,t);
hold on;
sigmaRange = logspace(-2, 1, 5);
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
legend(legendstrs,'Interpreter','latex');
text(0, -0.05, 'Higher values of \sigma correspond to straighter decision boundaries.');
saveas(gcf,['figures/Various sigma values.png']);
title(['Decision boundary for various \sigma values and C = ' num2str(C)]);
