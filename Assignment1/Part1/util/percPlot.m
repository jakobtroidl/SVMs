function percPlot(X, t, wonline, wbatch, yonline, ybatch, mode)
% mode can be 'augmented', 'transformed' or 'image'

% Accuracy
erronline = sum(yonline ~= t) / numel(t);
errbatch = sum(ybatch ~= t) / numel(t);

figure;

if ~strcmp(mode,'image')
    % Plot the input vectors in R2 and visualize corresponding target values
    % (e.g. by using color).
    plotPoints(X, t);

    badonline = yonline ~= t;
    Xbadonline = X(:,badonline);
    scatter(Xbadonline(1,:), Xbadonline(2,:), 'dk');

    badbatch = ybatch ~= t;
    Xbadbatch = X(:,badbatch);
    scatter(Xbadbatch(1,:), Xbadbatch(2,:), 'sk');
end

hold on;

if strcmp(mode, 'image')
    subplot(1,2,1);
    title('aaa');
end
plotBoundary(wonline, mode, '--k');

if strcmp(mode, 'image')
    xlabel('Online');
    subplot(1,2,2);
end
plotBoundary(wbatch, mode, '-k');

if ~strcmp(mode, 'image')
   legend('Zeros', 'Ones', 'Misclassified online', 'Misclassified batch', ['Online (err=' num2str(erronline) ')'], ['Batch (err=' num2str(errbatch) ')']); 
   xlabel('Filled area');
   ylabel('Solidity');
else
    xlabel('Batch');
end

hold off;

end