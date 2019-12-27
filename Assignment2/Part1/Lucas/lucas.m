addpath(genpath('util'));
addpath('readmnist')

[X, t, Xtest, imgs, imgsTest] = readdata();
figure;
plotdata(X, t);
legend('Zeros','Ones');
title('Input data');