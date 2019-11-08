% Experimental setup
N = 7;
d = 2;
threshold = 0.2; % training will terminate when error is lower than threshold
gamma = 0.3; % learning rate

x = 0:0.1:5;
y = 2 * x.^2 - 5 * x + 1;

% extracting the training set
x_training = x(1:(N+1):end);
t = y(1:(N+1):end);

% compute offsets for t_training using the normal distribution
% create the normal distribution
n_x = -5:.1:5;
n_dist = normpdf(n_x, 0, 4);

% extract 7 random values of the normal distribution
indices = randperm(numel(n_dist), N);
offset = n_dist(indices);

t = t + offset;

power = [0; 1; 2]; % powers of the phi function
X = repmat(x_training, d + 1, 1);

% X matrix is created based on the phi function
X = X .^ power;

w = [0; 0; 0];

% compute the error 
o = transpose(w) * X;
error = (t - o) * (t - o)';
