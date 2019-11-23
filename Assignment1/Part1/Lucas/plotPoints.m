function plotPoints(X, t, y)

ax = X(1, t == -1);
ay = X(2, t == -1);
bx = X(1, t == 1);
by = X(2, t == 1);

c = lines(2);
scatter(ax, ay, 'xr');%, 'MarkerEdgeColor', c(1,:));
hold on;
scatter(bx, by, '+g');%, 'MarkerEdgeColor', c(2,:));
%hold off;
%legend('Zeros', 'Ones');
mini = min(X,[],2);
maxi = max(X,[],2);
hor = (maxi(1) - mini(1)) / 2;
ver = (maxi(2) - mini(2)) / 2;
xlim([mini(1)-hor maxi(1)+hor]);
ylim([mini(2)-ver maxi(2)+ver]);

end