% Experimental setup
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
