siz = 20;
margin = 2;
count = 10;
digitsImage = zeros(siz*10 + margin*9, siz*count + margin*(count-1));
for dig = 0:9
    y = (dig * siz) + 1 + (dig * margin);
    digits = digit(imgs, labels, dig);
    for column = 1:count
        x = ((column - 1) * siz) + 1 + ((column - 1) * margin);
        digitsImage(y:y+siz-1, x:x+siz-1) = digits(:,:,column);
    end
end
figure;
imshow(digitsImage);