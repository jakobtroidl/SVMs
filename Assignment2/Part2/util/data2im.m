function img = data2im(data)
% Assumes that images are square

width = sqrt(size(data,2));
img = reshape(data, width, width, size(data,1));

end