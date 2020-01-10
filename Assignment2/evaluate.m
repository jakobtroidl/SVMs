function [avgErrorSVM, avgErrorPerc] = evaluate(trainSamples, trainLabels, testSamples, testLabels, C, sigma)
% todo
M = length(trainLabels);

% initialize output parameters
if nargin == 4
    avgErrorSVM = zeros(1, 1, M);
else
    avgErrorSVM = zeros(length(C), length(sigma), M);
end

% evaluate perceptron
testSize = size(testSamples, 2);
incorr_Perc = zeros(1, 1, M);
for k = 1:M
    %% Perceptron
    maxIts = 10000;
    mode = 'batch';
    w = percTrain(trainSamples{k}, trainLabels{k}', maxIts, mode);
    
    yPerc = perc(w, testSamples);
    incorr_Perc(:, :, k) = sum(yPerc' ~= testLabels) / testSize;
end
avgErrorPerc = incorr_Perc;

% evaluate SVM
for k = 1:M
    if nargin == 4
        avgErrorSVM(1, 1, k) = evaluateSVM(trainSamples{k}, trainLabels{k}, testSamples, testLabels);
    else
        for i = 1 : length(C)
            for j = 1 : length(sigma)
                % SVM
                avgErrorSVM(i,j, k) = evaluateSVM(trainSamples{k}, trainLabels{k}, testSamples, testLabels, C(i), sigma(j));
                
            end
        end
    end
end
end

