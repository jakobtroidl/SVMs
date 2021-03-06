function plotBoundary(w, mode, style)
% mode must be '2d', 'nd' or 'image'


if nargin < 3
    style = '-k';
end

if strcmp(mode,'2d')
    
    bias = -w(3);
    w = w([1 2]);
    
    wlen = sqrt(sum(w .* w));
    p0 = w ./ wlen .* (bias ./ wlen);
    vec = [w(2) -w(1)];
    
    p1 = p0 + vec * 10;
    p2 = p0 - vec * 10;
    plot([p1(1) p2(1)], [p1(2) p2(2)], style);
    
%     plotpc(w, -bias); % Matlab's function
%     hold on;
    
elseif strcmp(mode,'nd')
    [X, Y] = meshgrid(0:0.01:1);
    len = size(X,2);
    x = reshape(X, 1, len, len);
    y = reshape(Y, 1, len, len);
    z = transFts(x, y);
    
    % Dot product
    z = sum(repmat(w', 1,len,len) .* z);
    
    z = reshape(z, len, len);
    
    contour(X,Y,z,[0 0],style);
    colormap([0 0 0]);
    hold on;
    
elseif strcmp(mode, 'image')
    img = data2im(w);
    mini = min(min(img));
    maxi = max(max(img));
    imshow((img - mini) ./ (maxi - mini));
    
else
    error(['Unsupported mode=' mode]);
end

end