function [incorr_SVM] = evaluateSVM(trainSamples, trainLabels, testSamples, testLabels, C, sigma)
%todo
    testSize = size(testSamples, 2);
    if nargin == 4
        % train linear SVM
        [alpha, w0] = trainSVM(trainSamples', trainLabels);
    else
        % train a SVM with kernel and regularization parameter C
        kernel = @(x1, x2)rbfkernel(x1, x2, sigma);
        [alpha, w0] = trainSVM(trainSamples', trainLabels, kernel, C);
    end

    %% Comparison
    if nargin == 4
        ySVM = discriminant(alpha, w0, trainSamples', trainLabels, testSamples');
    else 
        ySVM = discriminant(alpha, w0, trainSamples', trainLabels, testSamples', kernel);
    end
    incorr_SVM = sum(sign(ySVM) ~= testLabels) / testSize;
end

