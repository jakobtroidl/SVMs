function imgsOut = digit(imgs, labels, digit)

imgsOut = imgs(:, :, labels == digit);

end