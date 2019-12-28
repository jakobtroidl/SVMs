function plotdata(X, t)

ax = X(t == -1, 1);
ay = X(t == -1, 2);
bx = X(t == 1, 1);
by = X(t == 1, 2);

c = lines(2);
scatter(ax, ay, 'xr');%, 'MarkerEdgeColor', c(1,:));
hold on;
scatter(bx, by, '+g');%, 'MarkerEdgeColor', c(2,:));
%hold off;
%legend('Zeros', 'Ones');
mini = min(X);
maxi = max(X);
hor = (maxi(1) - mini(1)) / 4;
ver = (maxi(2) - mini(2)) / 4;
xlim([mini(1)-hor maxi(1)+hor]);
ylim([mini(2)-ver maxi(2)+ver]);

end