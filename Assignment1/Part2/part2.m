%% Experimental setup
N = 7;
d = 2;
threshold = 50; % training will terminate when error is lower than threshold
gamma = 0.025; % learning rate
max_iterations = 10000; % max number of training iterations

% generate data for the true target function
x = 0:0.1:5; 
y = 2 * x.^2 - 5 .* x + 1;

% extracting the training set
x_training = x(1:(N+1):end);
t = y(1:(N+1):end);

% compute offsets for t_training using the normal distribution
offset = normrnd(0, 4, 1, N);

power = (d:-1:0)'; % powers of the phi function
X = repmat(x_training, d + 1, 1);

%% Parameter Optimization
% X matrix is created based on the phi function
X = X .^ power;
t = t + offset;
w = [0; 0; 0];

% compute the error 
e_old = realmax;
c = realmax;
e = error(w, X, t);
counter = 0;

% perform training
while counter < max_iterations && e > threshold &&  c > 0.0009
    
    % compute current index in [1, N]
    i = mod(counter, N) + 1;
    
    % update weight vector
    oi = w' * X(:, i);
    w = w + 2 * (gamma * (counter + 1)^(-1/2)) * (t(i) - oi) * X(:, i);
    e = error(w, X, t);
    
    %check for convergence
    c = abs(e_old - e);
    e_old = e;
    
    %generate console output
    out = [ 'Iterations: ', num2str(counter), ...
            ' Error: ', num2str(e), ... 
            sprintf(' w: [%d, %d, %d]', w) ];
        
    disp(out);
    counter = counter + 1;
end

% compute final target values
t_out = zeros(1, size(x, 2));
for i = 1:size(x,2)
    t_out(i) = w' * (repmat(x(i), d + 1, 1) .^ power);
end



% compute w* in closed version
w_closed = pinv(X') * t';
%w_closed2 = X' \ t';
t_out_closed = zeros(1, size(x, 2));
for i = 1:size(x,2)
    t_out_closed(i) = w_closed' * (repmat(x(i), d + 1, 1) .^ power);
end

out = ['Online: ', num2str(error(w, X, t)), 'Closed: ', num2str(error(w_closed, X, t))];
disp(out);

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
