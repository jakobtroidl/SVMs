% Part 2 - Bullet 1
% Enhance the above functions with RBF-Kernel functions: Either write the 
% function [k] = rbfkernel(x1,x2,sigma) or use an anonymous function for 
% the kernel evaluation of two input vectors and the RBF parameter ?. 
% Use a function handle as an additional parameter of trainSVM 
% and discriminant to pass the kernel to these functions2.

% Function rbfkernel() implements equation
% k_ij = exp(-|x1_i - x2_j|^2 / sigma^2)
% that is written for 2-dimensional input vectors x1_i, x2_j representing 
% our data samples with filled area & solidity parametes.
% In rbfkernel() we are inputting multiple x1_i and x2_j vectors 
% (data samples) in matrices x1 and x2 and calculating k_ij for each 
% combination of them, creating an output matrix k of k_ij coefficients 
% of dimensions same as matrix x1*x2.

function [k] = rbfkernel(x1, x2, sigma)

    x = size(x1, 1);
    y = size(x2, 1);
    
    aPlus = repmat(x1, [1 1 y]);
    aMinus = repmat(x2,[1 1 x]);
    aMinus = permute(aMinus, [3 2 1]);
    components = permute(aPlus - aMinus, [1 3 2]);
    
    norm = sum(components .^ 2, 3);
    k = exp( - norm / (sigma ^ 2));
     
end