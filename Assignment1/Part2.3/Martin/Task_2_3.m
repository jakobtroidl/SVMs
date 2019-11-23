clear all;
close all;

%% Parameter settings
% Calculation Parameters
groupID = 5;
N = 7;  % training set size
imageSizeX = 29;
imageSizeY = 29;
muX = ceil(imageSizeY/2);
muY = ceil(imageSizeX/2);
stDeviation = 2;    % default 2
displayScale = 3;
lambda = 0.1;   % ridge penalty
x = 0:0.1:5;    % input for predictor (images)
y = 2*x.^2 -groupID.*x + 1; % target values

% Display parameters
lineWidth = 1;
markerSize = 12;

%% Calculation of influence of increasing variance to the RSS
stDeviationArray = linspace(stDeviation^2, 10*stDeviation^2, 10).^0.5;
errorRSS = stDeviationArray.*0;

trainDataStep = (size(x, 2) - mod(size(x, 2), N))/N + 1;
if trainDataStep == 0
    disp('Insufficient size of input data vector x');
    return;
end
xTraining = x(1:trainDataStep:trainDataStep*N);
yTraining = y(1:trainDataStep:trainDataStep*N);
tTraining = y(1:trainDataStep:trainDataStep*N);

for stDevIndex = 1:size(stDeviationArray, 2)
    % Circle center x, y coordinates including noise
    m1 = normrnd(muX, stDeviationArray(stDevIndex), 1, size(x, 2));
    m2 = normrnd(muY, stDeviationArray(stDevIndex), 1, size(x, 2));

    % Generate images
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
    
    % Calculate w* in closed form
    wEstimate = (xTrainImagesArray*xTrainImagesArray' + lambda*eye(imageSizeX*imageSizeY + 1))\xTrainImagesArray*tTraining';
    
    % Compute the training RSS error
    yTrainPredicted = wEstimate'*xTrainImagesArray;
    trainingErrorRSS = sum((yTraining - yTrainPredicted).^2)    % RSS (Residual Sum of Squares)
    
    % Compute complete sample RSS error
    yPredicted = wEstimate'*xImagesArray;
    errorRSS(stDevIndex) = sum((y - yPredicted).^2);
    
    if stDevIndex == 1  % plot graphs for stDev = 2
        displayArray = [];
        for img = 1:N
            displayArray = [displayArray xTrainImages(:, :, img)];
            displayArray = [displayArray ones(imageSizeY, displayScale)];
        end

        figure;
        imshow(displayArray);
        truesize([imageSizeY*displayScale, imageSizeX*displayScale*N]);
        saveas(gcf,['Training_images.png']);
        title('Training sample images', 'Interpreter', 'latex');
        
        % Plot the predicted ^y_i vs. the true y_i for the training images.
        figure;
        plot(x, y, 'g', 'LineWidth', lineWidth);
        hold on;
        plot(xTraining, yTrainPredicted, 'bx', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
        legend('true target $y$', 'predicted $\hat{y}$', 'Interpreter', 'latex');
        xlabel('Input x [px]', 'Interpreter', 'latex');
        ylabel('Output $y, \hat{y}$ [-]', 'Interpreter', 'latex');
        text(min(x), max(y), ['Training error RSS = ' num2str(trainingErrorRSS, 4) '[-]'], 'Interpreter', 'latex');
        grid on;
        saveas(gcf,['Evaluation_restricted_train_images.png']);
        title('Comparison of true target $y$ to train sample images $\hat{y}$ prediction', 'Interpreter', 'latex');
        
        figure;
        plot(x, y, 'g', 'LineWidth', lineWidth);
        hold on;
        plot(x, yPredicted, 'bx', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
        legend('true target $y$', 'predicted $\hat{y}$', 'Interpreter', 'latex');
        xlabel('Input x [px]', 'Interpreter', 'latex');
        ylabel('Output $y, \hat{y}$ [-]', 'Interpreter', 'latex');
        grid on;
        saveas(gcf,['Evaluation_all_images.png']);
        title('Comparison of true target $y$ to full sample images $\hat{y}$ prediction', 'Interpreter', 'latex');
            
        weightImage = reshape(wEstimate(2:end), imageSizeY, imageSizeX);
        weightImage = weightImage - min(min(weightImage));
        weightImage = weightImage./max(max(weightImage));
        figure;
        imshow(weightImage);
        truesize([imageSizeY*displayScale*2, imageSizeX*displayScale*2]);
        saveas(gcf,['WeightVectorImage.png']);
        title('Weight vector estimate $w*$ image representation', 'Interpreter', 'latex');
    end
end

figure;
plot(stDeviationArray.^2, errorRSS, 'x', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
xlabel('Variance of circle center coordinates $\sigma [px^2]$', 'Interpreter', 'latex');
ylabel('Full image sample error RSS [-]', 'Interpreter', 'latex');
grid on;
saveas(gcf,['VarianceToRSS.png']);
title('Influence of circle coordinates variance to RSS of full image sample', 'Interpreter', 'latex');