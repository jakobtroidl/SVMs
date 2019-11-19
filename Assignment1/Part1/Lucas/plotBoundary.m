function plotBoundary(X, w, force2d)

if nargin <= 2
    force2d = false;
end

% d = 2
if size(X,1) == 2 || force2d
    plotpc(w([1 2]),w(end));
    
elseif (size(X,2) > 2)
    disp('d>2 not yet implemented');
    
end

end