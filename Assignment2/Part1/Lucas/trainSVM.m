function [alpha, w0] = trainSVM(X,t)

d = size(X,2); % dimensions
N = size(X,1); % size

% Minimize:
%   1/2 * x' * H * x  +  f' * x
% With constraints:
%   A   * x <=  b
%   Aeq * x  =  beq
xt = (X .* t)';
H = xt' * xt;
f = -ones(N,1);

A = -eye(N,N);
b = zeros(N,1);

Aeq = t';
beq = 0;

[alpha] = quadprog(H,f,A,b,Aeq,beq);

%% Getting w0
% Based on a support vector 's'
[~,s] = max(alpha);
ts = t(s);
xs = X(s,:);
w0 = ts - ((alpha .* t)' * X * xs');

end