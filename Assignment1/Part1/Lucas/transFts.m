function ftsOut = transFts(x1, x2)
% Use the feature transformation ?(x) : (x1,x2) ? (1,x1,x2,x1^2,x2^2,x1x2)
% and plot the data and decision boundary in the original data space R^2
% (see e.g. Figure 1) after training.
%
% Hint: Sample the relevant region of the inputs pace using a mesh grid.
% Compute y = w' * ?(x) for all grid points and use a contour-function or a
% surface-plot to visualize the approximation of the curve y= 0.

ftsOut = [ones(size(x1)); x1; x2; x1.*x1; x2.*x2; x1.*x2];

end