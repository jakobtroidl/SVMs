function imgsOut = getDigit(imgs, labels, digit)

imgsOut = imgs(:, :, labels == digit);

end