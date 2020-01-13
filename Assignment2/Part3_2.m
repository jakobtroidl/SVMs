addpath(genpath('util'));
addpath('readmnist')

% Generate at least M=150 disjoint training sets Tk
% by selecting images from the MNIST training set, where each Tk consists
% of not more than N=70 images

M = 150; %  # of training sets
N = 70; %  # of images per training set
train = zeros(M*N, 2);
trainL = zeros(M, 1); % labels

%% Get only 0 and 1
[imgs, labels] = readmnist('training');
imgs0 = digit(imgs,labels,0);
imgs1 = digit(imgs,labels,1);

%% Get M*N images
imgs = cat(3, ...
    imgs0(:,:, 1:floor(M*N/2)), ...
    imgs1(:,:, 1:M*N - floor(M*N/2)));

%% Transform into data
X = im2data(imgs);
t = [-ones(floor(M*N/2),1); ones(M*N - floor(M*N/2),1)];

%% Permutate randomly
idx = randperm(M*N);
X = X(:,idx);
t = t(idx);

%% Cross validation
Cs = [25 50 100 200 400 800];
sigmas = [0.5 5 10 50 100];
errors = crossvalidate(X, t, Cs, sigmas, M, N);
errorsavg = mean(errors,3);

%% Select optimal parameters
[Ce,Ci] = min(errorsavg);
[errorMin,sigmai] = min(Ce);
sigma = sigmas(sigmai);
C = Cs(Ci(sigmai));

%% Plot results
figure;
bar3(errorsavg);
xlabel('\sigma')
ylabel('C');
zlabel('Avg. error in %')
title(['Avg. cross validation error for M=' num2str(M) ' and  N=' num2str(N)])
set(gca,'YtickLabel', Cs);
set(gca,'XtickLabel', sigmas);
x = sigmai;
y = Ci(sigmai);
z = errorMin;
textheight = 0.1;
hold on;
scatter3(x,y,z,'ko','filled');
plot3([x,x],[y,y],[z,z+textheight],'k-');
hold off;
text(sigmai,Ci(sigmai),errorMin+textheight,['min: ' num2str(errorMin)]);
%saveas(gcf,'figures/crossval.png');

%% Get test set for 0 and 1 and transform into data
[imgsT, labelsT] = readmnist('test');
imgsT0 = digit(imgsT,labelsT,0);
imgsT1 = digit(imgsT,labelsT,1);
s = min(size(imgsT0,3), size(imgsT1,3)); % equal number of 0s and 1s
imgsT = cat(3, ...
    imgsT0(:,:, 1:s), ...
    imgsT1(:,:, 1:s));
test = im2data(imgsT)';
testL = [-ones(s,1); ones(s,1)];

%% Compare linear with non-linear SVM
errorLin = zeros(M,1);
errorNon = zeros(M,1);

totalTime = tic();
for k = 1:M
    idx = (k-1)*N+1 : k*N;
    train = X(:,idx)';
    trainL = t(idx);
    
    disp(['Training SVM pair ' num2str(k) ' out of ' num2str(M) '...']);
    tic;
    
    %% Linear
    [alpha, w0] = trainSVM(train, trainL);
    y = discriminant(alpha, w0, train, trainL, test);
    errorLin(k) = mean(sign(y) ~= testL);
    
    %% Non-linear
    kernel = @(x1, x2)rbfkernel(x1, x2, sigma);
    [alpha, w0] = trainSVM(train, trainL, C, kernel);
    y = discriminant(alpha, w0, train, trainL, test, kernel);
    errorNon(k) = mean(sign(y) ~= testL);
    
    %% Finish
    toc;
end
beep;
totalTime = toc(totalTime);
disp(['Done! Total time: ' num2str(totalTime)]);

%% Display results
errorAvgLin = mean(errorLin);
errorAvgNon = mean(errorNon);
disp(['Avg. error for linear SVM: ' num2str(errorAvgLin)]);
disp(['Avg. error for non-linear SVM: ' num2str(errorAvgNon)]);
figure;
bar([errorAvgLin*100,errorAvgNon*100]);
ylabel('Avg. error in %');
set(gca,'XtickLabel', {'Linear SVM', 'Non-linear SVM'});
title(['Avg. error for M=' num2str(M) ' and N=' num2str(N)])
%saveas(gcf,'figures/comp_lin-nonlin.png');