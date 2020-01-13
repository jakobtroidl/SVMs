addpath(genpath('util'));
addpath('readmnist')

[X, t, Xtest, ttest, ~, imgsTest] = readdata();

%% Part 2 - Bullet 1
sigma = 0.2; % RBF kernel parameter
C = Inf; % regularization parameter

% Create handle with determined sigma parameter
kernelFunc = @(x1, x2)rbfkernel(x1, x2, sigma);

[alpha, w0] = trainSVM(X, t, C, kernelFunc);

% Plot boundary
figure;
plotdata(X,t);
plotboundary(alpha, w0, X, t, kernelFunc);
legend('Zeros','Ones', 'Decision boundary', '|Discriminant|=1', ...
    'Support vectors');
%saveas(gcf,['figures/Decision boundary.png']);
title(['Decision boundary for \sigma = ' num2str(sigma) ' and C = ', ...
    num2str(C)]);

% Plot surface
figure;
plotdata(X,t);
plotboundary(alpha, w0, X, t, kernelFunc, 'surf');
legend('Zeros', 'Ones', 'Decision boundary', '|Discriminant|=1', ...
    'Support vectors', 'Location', 'best');
%saveas(gcf,['figures/Discriminant function surface.png']);
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
    [alpha, w0] = trainSVM(X, t, C, kernelFunc);
    plotboundary(alpha, w0, X, t, kernelFunc, 'nomargins', colors(i,:));
    legendstrs{2+i} = ['$\sigma$ = ' num2str(sigma)];
end
legend(legendstrs,'Interpreter','latex');
text(0, -0.05, 'Higher values of \sigma correspond to straighter decision boundaries.');
%saveas(gcf,['figures/Various sigma values.png']);
title(['Decision boundary for various \sigma values and C = ' num2str(C)]);

%% Part 3 - Bullet 3 - Slack variables, regularization parameter C
sigma = 0.2; % RBF kernel parameter
C = 0.5; % regularization parameter

% Create handle with determined sigma parameter
kernelFunc = @(x1, x2)rbfkernel(x1, x2, sigma);

[alpha, w0] = trainSVM(X, t, C, kernelFunc);

figure;
plotdata(X,t);
hold on;
plotboundary(alpha, w0, X, t, kernelFunc);
cRange = [0.3 0.1];

for i = 1:numel(cRange)
    kernelFunc = @(x1, x2)rbfkernel(x1, x2, sigma);
    [alpha, w0] = trainSVM(X, t, cRange(i), kernelFunc);
    plotboundary(alpha, w0, X, t, kernelFunc, 'justmargins', colors(i,:));
    %legendstrs{2+i} = ['$\sigma$ = ' num2str(sigma)];
end

legend('Zeros','Ones', ['Decision boundary C=' num2str(0.5)], ...
    ['|Discriminant|=1 C=' num2str(0.5)], 'Support vectors', ...
    ['|Discriminant|=1 C=' num2str(cRange(1))], ['|Discriminant|=1 C=' ...
    num2str(cRange(2))]);
%saveas(gcf,['figures/Slack variable.png']);
title(['Decision boundary for \sigma = ' num2str(sigma) ' and C = ', ...
    num2str(C)]);

%% Part 3 - Bullet 4 - Testing on data set of assignment 1
[imgs, labels] = readmnist('training', 2500);
imgs0 = digit(imgs, labels, 0);
imgs1 = digit(imgs, labels, 1);

count = 200;
[imgs0, fa0, s0] = digitSubsetProps(imgs0,count);
[imgs1, fa1, s1] = digitSubsetProps(imgs1,count);

[fa0, fa1, ~, ~] = normal(fa0, fa1);
[s0, s1, ~, ~] = normal(s0, s1);

X = [fa0 s0; fa1 s1];
t = [-ones(size(fa0'))'; ones(size(fa1'))'];

sigma = 0.2; % RBF kernel parameter
C = 2; % regularization parameter

% Create handle with determined sigma parameter
kernelFunc = @(x1, x2)rbfkernel(x1, x2, sigma);

[alpha, w0] = trainSVM(X, t, C, kernelFunc);


figure;
plotdata(X, t);
plotboundary(alpha, w0, X, t, kernelFunc);
legend('Zeros','Ones', 'Decision boundary', ...
    '|Discriminant|=1', 'Support vectors');
%saveas(gcf,['figures/Test set.png']);
title(['Decision boundary and test data for \sigma = ' num2str(sigma), ...
    ' and C = ' num2str(C)]);
