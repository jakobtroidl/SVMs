function [a, b, atest, btest] = normal(a, b, atest, btest)

ab = [a; b];
mini = min(ab);
maxi = max(ab);

a = (a - mini) ./ (maxi - mini);
b = (b - mini) ./ (maxi - mini);
atest = 0;
btest = 0;
if nargin > 2
    atest = (atest - mini) ./ (maxi - mini);
    btest = (btest - mini) ./ (maxi - mini);
end

end