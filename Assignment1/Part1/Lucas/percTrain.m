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

N = size(X,2);
d = size(X,1);

% bias = 0.1;
w = zeros(1,d);

if (online)
    learningRate = 1;
    
    for epoch = 1:maxIts
        for i = 1:N
            xiti = X(:,i) * t(i);
            if w * xiti <= 0 % Misclassified
                w = w + (learningRate * xiti)';
                
%                 hold on;
%                 scatter(X(end-1,i),X(end,i),'ok');
%                 y = sign(w * X);
%                 error = sum(y ~= t);
%                 disp(['Error: ' num2str(error)]);
%                 plotBoundary(w, true);
            end
        end
    end

else % batch
    learningRate = 0.01;
    for ite = 1:maxIts
        xt = X .* t;
        misclass = w * xt <= 0;
        if (sum(misclass) == 0)
            break;
        end
        xt = xt(:,misclass);
        w = w + (learningRate .* sum(xt,2))';
        
%         hold on;
%         y = sign(w * X);
%         error = sum(y ~= t);
%         disp(['Error: ' num2str(error)]);
%         plotBoundary(w, true);
    end
    
end

end