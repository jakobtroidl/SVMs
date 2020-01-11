function [incorr_SVM, numOfSupportVecs] = evaluateSVM(trainSamples, trainLabels, testSamples, testLabels, evaluateMode, C, sigma)
%todo
    testSize = size(testSamples, 2);
    if nargin == 5
        % train linear SVM
        [alpha, w0] = trainSVM(trainSamples', trainLabels);
    else
        % train a SVM with kernel and regularization parameter C
        kernel = @(x1, x2)rbfkernel(x1, x2, sigma);
        [alpha, w0] = trainSVM(trainSamples', trainLabels, kernel, C);
    end
    
    if strcmp(evaluateMode, 'EvalOnTestSet')
        evaluateSamples = testSamples;
        evaluateLabels = testLabels;
    else 
        evaluateSamples = trainSamples;
        evaluateLabels = trainLabels;
    end
    
    numOfSupportVecs = sum(alpha > 0.0001);

    %% Comparison
    if nargin == 5
        ySVM = discriminant(alpha, w0, trainSamples', trainLabels, evaluateSamples');
    else 
        ySVM = discriminant(alpha, w0, trainSamples', trainLabels, evaluateSamples', kernel);
    end
    incorr_SVM = sum(sign(ySVM) ~= evaluateLabels) / testSize;
end

