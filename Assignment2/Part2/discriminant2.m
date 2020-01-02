function [y] = discriminant2(alpha, w0, X, t, Xnew, kernel)

% y = d(x) = (w' * x) + w0
% w = sum from i=1 to N of:  ai * xi * ti
% therefore:  y = ((ai * xi * ti)' * x) + w0
y = (((alpha .* t)' * kernel(X, Xnew)) + w0)';

end