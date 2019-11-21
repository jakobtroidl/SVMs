clear all;
close all;

% Parameters
groupID = 5;
N = 7;  % training set size
imageSizeX = 29;
imageSizeY = 29;
muX = ceil(imageSizeY/2);
muY = ceil(imageSizeX/2);
sigma = 2;
displayScale = 3;
lambda = 0.1;   % ridge penalty

%% Calculate w* in closed form

x = 0:0.1:5;
y = 2*x.^2 -groupID.*x + 1;
m1 = normrnd(muX, sigma, 1, size(x, 2));
m2 = normrnd(muY, sigma, 1, size(x, 2));


trainDataStep = (size(x, 2) - mod(size(x, 2), N))/N + 1;
if trainDataStep == 0
    disp('Insufficient size of input data vector x');
    return;
end
xTraining = x(1:trainDataStep:trainDataStep*N);
yTraining = y(1:trainDataStep:trainDataStep*N);
tTraining = y(1:trainDataStep:trainDataStep*N);

%Image generation
for img = 1:size(x, 2)
    xImages(:, :, img) = zeros(imageSizeY, imageSizeX);
    for i = 1:imageSizeY
        for j = 1:imageSizeX
            if ((i - m1(img))^2 + (j - m2(img))^2 - (3*x(img))^2) < 0
                xImages(i, j, img) = 1;
            end
        end
    end
    xImagesArray(:, img) = [1; reshape(xImages(:, :, img), [imageSizeX*imageSizeY, 1])];
end

xTrainImages = xImages(:, :, 1:trainDataStep:trainDataStep*N);
xTrainImagesArray = xImagesArray(:, 1:trainDataStep:trainDataStep*N);


displayArray = [];
for img = 1:N
    displayArray = [displayArray xTrainImages(:, :, img)];
    displayArray = [displayArray ones(imageSizeY, displayScale)];
end

figure;
imshow(displayArray);
title([num2str(N) ' training images']);
truesize([imageSizeY*displayScale, imageSizeX*displayScale*N]);

wAsterisk = (xTrainImagesArray*xTrainImagesArray' + lambda*eye(imageSizeX*imageSizeY + 1))\xTrainImagesArray*tTraining';

%% Plot the predicted ?y_i vs. the true y_i for the 7 training images. Compute the training error.

yTrainPredicted = wAsterisk'*xTrainImagesArray;
trainingError = 0.5*sum((yTraining - yTrainPredicted).^2)    % SSE

figure;
plot(x, y);
hold on;
plot(xTraining, yTrainPredicted, 'x');
title(['Comparison of true y to ' num2str(N) ' train images $\hat{y}$ prediction'], 'Interpreter', 'latex');
legend('true y', 'predicted $\hat{y}$', 'Interpreter', 'latex');
xlabel('Input x [-]', 'Interpreter', 'latex');
ylabel('Output y, $\hat{y}$ [-]', 'Interpreter', 'latex');
text(min(x), max(y), ['Training error SSE = ' num2str(trainingError)], 'Interpreter', 'latex');

%% Plot the predicted ?yi vs. the true yi for all 51 images.

yPredicted = wAsterisk'*xImagesArray;

figure;
plot(x, y);
hold on;
plot(x, yPredicted, 'x');
title(['Comparison of true y to all ' num2str(size(x, 2)) ' images $\hat{y}$ prediction'], 'Interpreter', 'latex');
legend('true y', 'predicted $\hat{y}$', 'Interpreter', 'latex');
xlabel('Input x [-]', 'Interpreter', 'latex');
ylabel('Output y, $\hat{y}$ [-]', 'Interpreter', 'latex');
