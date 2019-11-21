%% Experimental setup
N = 7;
d = 2;
threshold = 80; % training will terminate when error is lower than threshold
gamma = 0.002; % learning rate
max_iterations = 10000; % max number of training iterations

% generate data for the true target function
x = 0:0.1:5; 
y = 2 * x.^2 - 5 * x + 1;

% extracting the training set
x_training = x(1:(N+1):end);
t = y(1:(N+1):end);

% compute offsets for t_training using the normal distribution
offset = normrnd(0, 4, 1, N);

power = (0:1:d)'; % powers of the phi function
X = repmat(x_training, d + 1, 1);

%% Parameter Optimization
% X matrix is created based on the phi function
X = X .^ power;
t = t + offset;
w = [0; 0; 0];

% compute the error 
e = error(w, X, t);
counter = 0;

% perform training
while counter < max_iterations && e > threshold
    
    % compute current index in [1, N]
    i = mod(counter, N) + 1;
    
    % update weight vector
    w = w + 2 * gamma * (t(i) - w' * X(:, i)) * X(:, i);
    e = error(w, X, t); 
    
    % generate console output
    out = [ 'Iterations: ', num2str(counter), ...
            ' Error: ', num2str(e), ... 
            sprintf(' w: [%d, %d, %d]', w) ];
        
    disp(out);
    counter = counter + 1;
end

% compute final target values
t_out = zeros(size(t));
for i = 1:size(x,2)
    t_out(i) = w' * (repmat(x(i), d + 1, 1) .^ power);
end

% compute w* in closed version
w_closed = pinv(X * X') * X * t';
t_out_closed = zeros(size(t));
for i = 1:size(x,2)
    t_out_closed(i) = w_closed' * (repmat(x(i), d + 1, 1) .^ power);
end

% print results
plot(x, y, 'Color', 'green', 'linewidth', 2);
hold on
plot(x, t_out, 'Color', 'red', 'linewidth', 2);
scatter(x_training, t, 'blue');
hold on
plot(x, t_out_closed, 'Color', 'cyan', 'linewidth', 2);
hold on
legend('true target', 'model online', 'training data', 'model closed')
hold off


function e = error(w, X, t)
% computes the error f a given weight vector
    o = w' * X;
    e = (t - o) * (t - o)';
end
