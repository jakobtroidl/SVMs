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
displayScale = 5;

x = 0:0.1:5;
y = 2*x.^2 -groupID.*x + 1;
m1 = normrnd(muX, sigma, 1, N);
m2 = normrnd(muY, sigma, 1, N);


trainDataStep = (size(x, 2) - mod(size(x, 2), N))/N;
if trainDataStep == 0
    disp('Insufficient size of input data vector x');
    return;
end
xTraining = x(1:trainDataStep:trainDataStep*N);
tTraining = y(1:trainDataStep:trainDataStep*N);

%Image generation
for img = 1:N
    images(:, :, img) = zeros(imageSizeY, imageSizeX);
    for i = 1:imageSizeY
        for j = 1:imageSizeX
            if ((i - m1(img))^2 + (j - m2(img))^2 - (3*xTraining(img))^2) < 0
                images(i, j, img) = 1;
            end
        end
    end
end

imageArray = [];
for img = 1:N
    imageArray = [imageArray images(:, :, img)];
    imageArray = [imageArray ones(imageSizeY, displayScale)];
end
figure;
imshow(imageArray);
truesize([imageSizeY*displayScale, imageSizeX*displayScale*N]);