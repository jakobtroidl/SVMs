function [trainImgs, trainLabels, testImgs, testLabels] = readMNISTauto()

if (evalin('base', 'exist(''trainImgs'',''var'') == 1') && ...
    evalin('base', 'exist(''trainLabels'',''var'') == 1'))

    disp('Training set already loaded. Skipping');
    trainImgs = evalin('base','trainImgs');
    trainLabels = evalin('base','trainLabels');
else
    readDigits = 5000; % Max: 60000
    disp(['Loading training set (' num2str(readDigits) ' images)...']);
    tic;
    imgFile = "train-images.idx3-ubyte";
    labelFile = "train-labels.idx1-ubyte";
    offset = 0;
    [trainImgs, trainLabels] = readMNIST(imgFile, labelFile, readDigits, offset);
    assignin('base', 'trainImgs', trainImgs);
    assignin('base', 'trainLabels', trainLabels);
    toc;
end

if (evalin('base', 'exist(''testImgs'',''var'') == 1') && ...
    evalin('base', 'exist(''testLabels'',''var'') == 1'))

    disp('Test set already loaded. Skipping');
    testImgs = evalin('base','testImgs');
    testLabels = evalin('base','testLabels');
else
    readDigits = 5000; % Max: 10000
    disp(['Loading test set (' num2str(readDigits) ' images)...']);
    tic;
    imgFile = "t10k-images.idx3-ubyte";
    labelFile = "t10k-labels.idx1-ubyte";
    offset = 0;
    [testImgs, testLabels] = readMNIST(imgFile, labelFile, readDigits, offset);
    assignin('base', 'testImgs', testImgs);
    assignin('base', 'testLabels', testLabels);
    toc;
end

end