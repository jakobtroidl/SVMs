function data = im2data(img)

data = reshape(img, size(img,1) * size(img,2), size(img,3));

end