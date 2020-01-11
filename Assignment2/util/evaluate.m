function [avgErrorSVM, numOfSupportVecs] = evaluate(trainSamples, trainLabels, testSamples, testLabels, evaluateMode, C, sigma)
% todo
M = length(trainLabels);

% initialize output parameters
if nargin == 5
    avgErrorSVM = zeros(1, 1, M);
    numOfSupportVecs = zeros(1, 1, M);
else
    avgErrorSVM = zeros(length(C), length(sigma), M);
    numOfSupportVecs = zeros(1, 1, M);
end

% evaluate SVM
for k = 1:M
    if nargin == 5
        [currAvgError, currNumOfSupportVecs] = evaluateSVM(trainSamples{k}, trainLabels{k}, testSamples, testLabels, evaluateMode);
        avgErrorSVM(1,1, k) = currAvgError;
        numOfSupportVecs(1, 1, k) = currNumOfSupportVecs;
    else
        for i = 1 : length(C)
            for j = 1 : length(sigma)
                % SVM
                [currAvgError, currNumOfSupportVecs] = evaluateSVM(trainSamples{k}, trainLabels{k}, testSamples, testLabels, evaluateMode, C(i), sigma(j));
                avgErrorSVM(i,j, k) = currAvgError;
                numOfSupportVecs(i, j, k) = currNumOfSupportVecs;
            end
        end
    end
end
end

