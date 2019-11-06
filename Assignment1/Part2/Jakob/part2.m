% Experimental setup
global d;
d = 2;

x = 0:0.1:5;
y = 2 * x.^2 - 5 * x + 1;

% extracting the training set

x_training = x(1:8:end);
t_training = y(1:8:end);

% compute offsets for t_training using the normal distribution

% create the normal distribution
n_x = -5:.1:5;
n_dist = normpdf(n_x, 0, 4);

% extract 7 random values of the normal distribution
indices = randperm(numel(n_dist), 7);
offset = n_dist(indices);

t_training = t_training + offset;

w = [0 0 0];

error = rss(w)

% compute the residual sum of squares error (RSS) for a given weight vector
% w. the output value is the error
function [error] = rss(w)
    global x_training;
    global t_training;
    error = 0;
    for i = 1:size(x_training, 2)
        ti = t_training(i);
        xi = x_training(i);
        error = error + (ti - dot(w, phi(xi))); 
    end
end

function [res] = phi(x_i)
    global d;
    p = 0:1:d;
    x_i = repmat(x_i, 1, d + 1);
    res = x_i .^ p;
end
