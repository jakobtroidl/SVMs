function [avgErrorPerc] = evaluatePercs(trainSamples, trainLabels, testSamples, testLabels)

% evaluate perceptron
M = length(trainLabels);
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
end

