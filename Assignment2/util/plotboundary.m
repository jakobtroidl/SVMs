function plotboundary(a, w0, X, t, kernel, mode, color)
% a = alpha
% Had to rename to a because we
%   want to use the command 'alpha'

if nargin < 6
    mode = '';
end

if nargin < 7
    color = [0,0,0];
end

[x, y] = meshgrid(0:0.01:1);
len = size(x,2);
n = numel(x);

Xnew = [reshape(x,n,1) reshape(y,n,1)];
if nargin == 4 || numel(kernel) == 0
    z = discriminant(a, w0, X, t, Xnew);
else
    z = discriminant(a, w0, X, t, Xnew, kernel);
end

z = reshape(z, len, len);

contour(x,y,z,[0 0],'-','Color',color);
if ~strcmp(mode, 'nomargins')
    contour(x,y,z+1,[0 0],'--','Color',color);
    contour(x,y,z-1,[0 0],'--','Color',color);
    
    sv = X(a > 0.00001, :);
    scatter(sv(:,1),sv(:,2),'ko');
end

if strcmp(mode, 'surf')
    surf(x,y,z);
    zlabel('Discriminant output y');
    shading interp;
    alpha 0.5;
    view(3);
end
    
end