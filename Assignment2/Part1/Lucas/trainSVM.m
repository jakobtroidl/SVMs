function [alpha, w0] = trainSVM(X,t)

d = size(X,1); % dimensions
N = size(X,2); % size

X = [X; ones(1,N)];
d = d+1;

% width of "street" = (x+ - x-) * w/||w||
% constraint = yi(xi*w + b) - 1 >= 0   for any xi
%         OR = yi(xi*w + b) - 1  = 0   for xi in gutter
%
% therefore: width of "street" = (1-b + 1+b) / ||w||
%                              = 2 / ||w||
%
%   maximize:   max 2/||w||   (width of "street")
% = maximize:   max 1/||w||
% = minimize:   min ||w||
% = minimize:   min 1/2 * ||w||^2
%
% LAGRANGE!
%   L   =   1/2 * ||w||^2   -   <sum of all constraints>
%
% constraints depend on i (because xi and yi)
% each constraint has a multiplier ai (alpha of i, normally lambda for the Lagrangian)
% each constraint:
%       ai * <constraint>
%   =   ai * ( yi(xi*w + b) - 1 )

%% H and f for quadprog
% In quadprog, we want to minimize: (1/2 * x' * H * x) + (f' * x)
% In the SVM, we want to minimize:   (1/2 * ||w||^2)
%                                  = (1/2 * w' * w)
%                                  = (1/2 * w' * I * w)   I being the identity matrix
% In other words: x=w, H=I, and f=0
H = eye(d);
H(end,end) = 0;
f = zeros(d,1);



end