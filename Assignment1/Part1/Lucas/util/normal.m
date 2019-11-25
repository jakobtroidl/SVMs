function [a, b, atest, btest] = normal(a, b, atest, btest)

ab = [a; b];
mini = min(ab);
maxi = max(ab);

a = (a - mini) ./ (maxi - mini);
b = (b - mini) ./ (maxi - mini);
atest = (atest - mini) ./ (maxi - mini);
btest = (btest - mini) ./ (maxi - mini);

end