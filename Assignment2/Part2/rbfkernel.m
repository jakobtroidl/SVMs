% Part 2 - Bullet 1
% Enhance the above functions with RBF-Kernel functions: Either write the 
% function [k] = rbfkernel(x1,x2,sigma) or use an anonymous function for 
% the kernel evaluation of two input vectors and the RBF parameter ?. 
% Use a function handle as an additional parameter of trainSVM 
% and discriminant to pass the kernel to these functions2.

function [k] = rbfkernel(x1, x2, sigma)

    
    aPlus = repmat(x1(:, 1), 1, size(x2, 1));
    aMinus = repmat(x2(:, 1)', size(x1, 1), 1);
    a = aPlus - aMinus;
    
    bPlus = repmat(x1(:, 2), 1, size(x2, 1));
    bMinus = repmat(x2(:, 2)', size(x1, 1), 1);
    b = bPlus - bMinus;
    
    k = exp(-(sqrt(a.^2 + b.^2)) / (sigma^2));
end