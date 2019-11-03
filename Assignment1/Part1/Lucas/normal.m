function [a, b] = normal(a, b)

ab = [a; b];
mini = min(ab);
maxi = max(ab);
a = (a - mini) ./ maxi;
b = (b - mini) ./ maxi;

end