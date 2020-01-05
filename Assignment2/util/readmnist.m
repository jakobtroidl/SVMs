function [imgs, labels] = readmnist(type,readDigits,offset)
% type = 'training' or 'test

if nargin < 3
    offset = 0;
end

if strcmp(type,'training')
    imgFile = "train-images.idx3-ubyte";
    labelFile = "train-labels.idx1-ubyte";
    if nargin < 2
        readDigits = 60000;
    end
elseif strcmp(type,'test')
    imgFile = "t10k-images.idx3-ubyte";
    labelFile = "t10k-labels.idx1-ubyte";
    if nargin < 2
        readDigits = 10000;
    end
end

[imgs, labels] = readMNIST(imgFile, labelFile, readDigits, offset);

end