function percPlot(X, t, wonline, wbatch, yonline, ybatch, mode)
% mode can be '2d', 'nd' or 'image'

% Accuracy
acconline = sum(yonline == t) / numel(t);
accbatch = sum(ybatch == t) / numel(t);

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
end
plotBoundary(wonline, mode, '--k');

if strcmp(mode, 'image')
    subplot(1,2,2);
end
plotBoundary(wbatch, mode, '-k');

if ~strcmp(mode, 'image')
   legend('Zeros', 'Ones', 'Incorrect online', 'Incorrect batch', ['Online (acc=' num2str(acconline) ')'], ['Batch (acc=' num2str(accbatch) ')']); 
end

hold off;

end