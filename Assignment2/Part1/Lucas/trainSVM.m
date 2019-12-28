function [alpha, w0] = trainSVM(X,t)

d = size(X,2); % dimensions
N = size(X,1); % size

X = [X ones(N,1)];
d = d+1;


%% quadprog returns a vector x that minimizes (1/2 * x' * H * x) + (f' * x)
H = eye(d);
H(end,end) = 0;
f = zeros(d,1);

%% Constraints A*x <= b
% In the SVM:     (w' * xi + w0)ti >= 1   (but for us, w and w0 are one)
%             =>  w'*xi*ti >= 1
%             =>  -w'*xi <= ti   (since ti is either -1 or +1)
%             => -xi' * w <= ti   (so it looks more like quadprog)
A = -X;
b = t;

[w,~,~,~,lambda] = quadprog(H, f, A, b);
alpha = lambda.ineqlin;
w0 = w(end);

end