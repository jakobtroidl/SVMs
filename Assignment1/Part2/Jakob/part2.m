%% Experimental setup
N = 7;
d = 2;
threshold = 0.02; % training will terminate when error is lower than threshold
gamma = 0.02; % learning rate

x = 0:0.1:5;
y = 2 * x.^2 - 5 * x + 1;

% extracting the training set
x_training = x(1:(N+1):end);
t = y(1:(N+1):end);

% compute offsets for t_training using the normal distribution
% create the normal distribution
n_x = -5:.1:5;
n_dist = normpdf(n_x, 0, 4);

% extract N random values of the normal distribution
indices = randperm(numel(n_dist), N);
offset = n_dist(indices);

power = [0; 1; 2]; % powers of the phi function
X = repmat(x_training, d + 1, 1);

%% Parameter Optimization
% X matrix is created based on the phi function
X = X .^ power;
t = t + offset;
w = [0; 0; 0];

% compute the error 
e = error(w, X, t);
diff = realmax('double');
counter = 0;

% perform training
while counter < 100
    %i = mod(counter, N) + 1;
   
    for i = 1:7 
        w = w + 2 * gamma * (t(i) - w' * X(:, i)) * X(:, i);
    end
    
    new_error = error(w, X, t); 
    
    out = [ 'Iterations: ', num2str(counter), ...
            ' Error: ', num2str(new_error), ... 
            sprintf(' w: [%d, %d, %d]', w)];
        
    disp(out);
    counter = counter + 1;
    e = new_error;
end

% plot(x, y, 'Color', 'green');
% hold on
% scatter(x_training, t);
% %hold on
% %fplot(@(x) w(1) + x * w(2) + x^2 * w(3),'Color', 'red')
% hold off



function e = error(w, X, t)
% computes the error from a given weight vector
    o = w' * X;
    e = (t - o) * (t - o)';
end
