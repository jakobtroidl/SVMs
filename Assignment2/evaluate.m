function [avgErrorSVM] = evaluate(trainSamples, trainLabels, testSamples, testLabels, evaluateMode, C, sigma)
% todo
M = length(trainLabels);

% initialize output parameters
if nargin == 5
    avgErrorSVM = zeros(1, 1, M);
else
    avgErrorSVM = zeros(length(C), length(sigma), M);
end

% evaluate SVM
for k = 1:M
    if nargin == 5
        avgErrorSVM(1, 1, k) = evaluateSVM(trainSamples{k}, trainLabels{k}, testSamples, testLabels, evaluateMode);
    else
        for i = 1 : length(C)
            for j = 1 : length(sigma)
                % SVM
                avgErrorSVM(i,j, k) = evaluateSVM(trainSamples{k}, trainLabels{k}, testSamples, testLabels, evaluateMode, C(i), sigma(j));
                
            end
        end
    end
end
end

