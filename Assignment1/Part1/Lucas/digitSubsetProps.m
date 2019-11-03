function [imgsOut, filledArea, solidity] = digitSubsetProps(imgs, count)

s = size(imgs);
s(3) = count;
imgsOut = zeros(s);

filledArea = zeros(count,1);
solidity = zeros(count,1);

idx = 1;
disp(['Extracting properties of ' num2str(count) ' images...']);
tic;
for i = 1:size(imgs,3)
    
    img = imgs(:,:,i);
    props = regionprops(img, 'FilledArea', 'Solidity');
    
    if (size(props,1) > 0)
        filledArea(idx) = props.FilledArea;
        solidity(idx) = props.Solidity;
        idx = idx + 1;
        if (idx > count)
            break
        end
    end
    
end

if (idx <= count)
    error(['Not enough valid images. Required=' num2str(count) ' Gotten=' num2str(idx-1)]);
else
    toc;
end

end