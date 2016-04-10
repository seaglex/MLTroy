example1 = [24128	959.92
24129	768.08
24130	3528.9
24131	2721.13
24132	5948.53
24133	5507.6
24134	2786.68
24135	5613.04
24136	9039.69
24137	12443.54
24138	8817.43
24139	8612.77
24140	10274.34
24141	9003.22
24142	8247.58
24143	10237.84
24144	11616.97
24145	18649.17
24146	24796.61
24147	18259.2
24148	15816.96
24149	17381.63
24150	18926.23
24151	23825.51
24152	21350.96
24153	13283.06
24154	15099.5
24155	17226.95
24156	16870.53
24157	7485.88
24158	4658.35
24159	7696.71
24160	10429.12
24161	12235.81
24162	12838.04
24163	10815.97];
x = example1(:, 1);
y = example1(:, 2);

num = length(x);
mx = sum(x) / num;
stdevx = sqrt(x'*x/num - mx^2);
my = sum(y) / num;
stdevy = sqrt(y'*y/num - my^2);
x = (x - mx)./stdevx;
y = (y - my)./stdevy;

% regression
theta = [1, 1, 1, 1]';
sigma2 = 0.1;
[theta, sigma2] = optimize_gp(x, y, theta, sigma2);

min_hist = 0;
nume = 0;
avge = 0;
rmse = 0;
ey = zeros(num, 1);
evy = zeros(num, 1);
for n = min_hist+1: num
    if n == 1
        m1 = 0;
        v1 = sigma2;
    else
        X = x(1:n-1);
        T = y(1:n-1);
        [m1, v1] = nonlinear_gp_regression(X, T, x(n), theta, sigma2);
    end
    ey(n) = m1;
    evy(n) = v1;
    e = y(n)-m1;
    avge = avge + e;
    rmse = rmse + e^2;
    nume = nume + 1;
end
avge = avge / nume;
rmse_gp = sqrt(rmse/nume - avge^2);
plot(x, y, 'r^-');
hold on;
errorbar(x, ey, sqrt(evy));
hold off;
box off;
grid on;

