function [alpha, w0] = trainSVM(X,t,kernel,C)

d = size(X,2); % dimensions
N = size(X,1); % size

% Minimize:
%   1/2 * x' * H * x  +  f' * x
% With constraints:
%   A   * x <=  b
%   Aeq * x  =  beq
%   lb <= x <= ub

Aeq = t';
beq = 0;

f = -ones(N,1);

if nargin == 2
    xt = (X .* t)';
    H = xt' * xt;
    
    A = -eye(N,N);
    b = zeros(N,1);

    [alpha] = quadprog(H,f,A,b,Aeq,beq);
else
    H = kernel(X, X).*(t*t');
    
    lb = zeros(N, 1);
    ub = ones(N, 1) * C;
    
    [alpha] = quadprog(H,f,[],[],Aeq,beq,lb,ub);
end

%% Getting w0
% Based on a support vector 's'
% Hint: Note that with C<Inf you have to take care about selecting a
% support vector xs with margin |d(xs)| = 1 to calculate w0.

if nargin == 2
    [~,s] = max(alpha);
else
    % prevent the system from a support vector with alpha = C
    idx = alpha < C - 0.01;
    [~,s] = max(alpha .* idx);
end

ts = t(s);
xs = X(s,:);

if nargin == 2
    w0 = ts - ((alpha .* t)' * X * xs');
else
    w0 = ts - ((alpha .* t)' * kernel(X, xs)); % * X * xs'
end

end