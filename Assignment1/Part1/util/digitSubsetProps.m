function [imgsOut, prop1, prop2] = digitSubsetProps(imgs, count)

s = size(imgs);
s(3) = count;
imgsOut = zeros(s);

prop1 = zeros(count,1);
prop2 = zeros(count,1);

idx = 1;
disp(['Extracting properties of ' num2str(count) ' images...']);
tic;
for i = 1:size(imgs,3)
    
    img = imgs(:,:,i) > 0;
    props = regionprops(img, 'Eccentricity', 'FilledArea', 'Solidity',...
        'Area', 'ConvexArea', 'EquivDiameter', 'Extent');
    
    if (size(props,1) > 0)
        prop1(idx) = props.FilledArea;
        prop2(idx) = props.Solidity;
        imgsOut(:,:,idx) = imgs(:,:,i);
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