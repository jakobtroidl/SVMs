function errors = crossvalidate(sets, labels, Cs, sigmas, M, N)

if iscell(sets) % if sets is a cell array
    d = size(sets{1},1); % dimensions/features
    n = size(sets,1);

    sizes = zeros(n,1);
    for i = 1:n
        sizes(i) = size(sets{i},2);
    end

    X = zeros(sum(sizes), d);
    t = zeros(sum(sizes, 1), 1);
    sizescum = [0; cumsum(sizes)];
    for i = 1:size(sets,1)
        idx = sizescum(i)+1:sizescum(i+1);
        X(idx, :) = sets{i}';
        t(idx) = labels{i};
    end
    
elseif nargin == 6 % if sets is not a cell array
    n = M;
    X = sets';
    t = labels;
    sizes = zeros(M,1) + N;
    sizescum = [0; cumsum(sizes)];
    
else
    error(['Unsupported input arguments: ' num2str(nargin)]);
end

errors = zeros(numel(Cs), numel(sigmas), n);

totaltime = tic;
for k = 1:n
    
    testI = sizescum(k)+1:sizescum(k+1);
    trainI = setdiff(1:size(X,1), testI);

    Xtest = X(testI,:);
    ttest = t(testI);
    Xtrain = X(trainI,:);
    ttrain = t(trainI);
    
    e = zeros(numel(Cs), numel(sigmas));

    for i = 1:numel(Cs)
        C = Cs(i);
        for j = 1:numel(sigmas)
            sigma = sigmas(j);
            kernel = @(x1, x2)rbfkernel(x1, x2, sigma);
            
            idx = 1+(j-1)+numel(sigmas)*((i-1)+numel(Cs)*(k-1));
            disp(['Starting SVM ' num2str(idx) ' out of ' num2str(n*numel(Cs)*numel(sigmas))]);
            tic;
            
            [alpha, w0] = trainSVM(Xtrain, ttrain, C, kernel);
            y = discriminant(alpha, w0, Xtrain, ttrain, Xtest, kernel);
            
            e(i, j) = sum(sign(y) ~= ttest) / numel(y);
            toc;
        end
    end
    
    errors(:,:,k) = e;
end
elapsed = toc(totaltime);
disp(['Done! Total time: ' num2str(elapsed)]);
beep;

end