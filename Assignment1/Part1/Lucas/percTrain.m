function w = percTrain(X, t, maxIts, online)
% Returns a weight vector w corresponding to the decision boundary
% separating the input vectors in X according to their target values t.
%
% The argument maxIts determines an upper limit for iterations of the
% gradient based optimization procedure. If this upper limit is reached
% before a solution vector is found, the function returns the current w,
% otherwise it returns the solution weight vector. online is true if the
% online-version of the optimization procedure is to be used or false for
% the batch-version.

N = size(X,1);
d = size(X,2);
learningRate = 1;
bias = 0.1;

% Add bias
w = [zeros(d,1); -bias];

% Add homogeneous coordinates
X = [X ones(N,1)];

% Permutate
perm = randperm(N);
X = X(perm,:);
t = t(perm);

if (online)
    for it = 1:maxIts
        noMisclass = true;
        for i = 1:N
            xt = X(i,:) .* t(i);
            if (xt * w <= 0) % Misclassified
                w = w + (learningRate * xt');
                noMisclass = false;
            end
        end
        if (noMisclass)
           break; 
        end
    end
        
else % batch
    normalized = (X .* t) * w;
    
end

end