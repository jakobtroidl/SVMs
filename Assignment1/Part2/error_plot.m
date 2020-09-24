x = 1:0.1:100;
us_y = (0.025 * (x).^(-1/2));
lecture_y = (0.025 ./x);

figure;
plot(x, us_y, 'linewidth', 2)

hold on;

plot(x, lecture_y, 'linewidth', 2);

l = legend('$\gamma (t) = \frac{0.025}{\sqrt{t}}$', '$\gamma (t) = \frac{0.025}{t}$');
set(l, 'Interpreter', 'latex');

hold off;
