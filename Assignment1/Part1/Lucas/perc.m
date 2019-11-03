function y = perc(w, X)
% Simulates a perceptron.
% The first argument is the weight vector w and the second argument is a
% matrix with input vectors in its columns X.  The output y is a binary
% vector with class labels 1 or -1

X = [X ones(size(X,1),1)];
y = sign(X * w);

end