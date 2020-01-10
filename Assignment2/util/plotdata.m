function plotdata(X, t, color1, color2)

if nargin <= 2
    color1 = [1 0 0];
    color2 = [0 1 0];
end

ax = X(t == -1, 1);
ay = X(t == -1, 2);
bx = X(t == 1, 1);
by = X(t == 1, 2);

c = lines(2);
scatter(ax, ay, 'x', 'MarkerEdgeColor', color1);
hold on;
scatter(bx, by, '+', 'MarkerEdgeColor', color2);
%legend('Zeros', 'Ones');
mini = min(X);
maxi = max(X);
hor = (maxi(1) - mini(1)) / 10;
ver = (maxi(2) - mini(2)) / 10;
xlim([mini(1)-hor maxi(1)+hor]);
ylim([mini(2)-ver maxi(2)+ver]);
xlabel('Filled area');
ylabel('Solidity');

end