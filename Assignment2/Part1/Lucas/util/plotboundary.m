function plotboundary(a, w0, X, t, plotsurf)
% a = alpha
% Had to rename to 'a' because we
%   want to use the command 'alpha'

if nargin < 5
    plotsurf = false;
end

[x, y] = meshgrid(0:0.01:1);
len = size(x,2);
n = numel(x);

Xnew = [reshape(x,n,1) reshape(y,n,1)];
z = discriminant(a, w0, X, t, Xnew);

z = reshape(z, len, len);

contour(x,y,z,[0 0],'-k');
contour(x,y,z+1,[0 0],'--k');
contour(x,y,z-1,[0 0],'--k');

sv = X(a > 0.00001, :);
scatter(sv(:,1),sv(:,2),'ko');

if plotsurf
    surf(x,y,z);
    shading interp;
    alpha 0.5;
    view(3);
end
    
end