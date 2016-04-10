N = 100;
eps = 0.05;
C = 2;

N1 = 50;
N2 = 50;
X = [randn(N1, 1)*eps+1, rand(N1, 1)*2*pi];
X = [X; randn(N2, 1)*eps+2, rand(N2, 1)*2*pi];
X = [X(:, 1).*cos(X(:, 2)), X(:, 1).*sin(X(:, 2))];

scatter(X(:, 1), X(:, 2))

A = zeros(N, N);
for n = 1:N
    for m = 1:N
        if n == m
            continue
        end
        d = X(n, :)-X(m, :);
        A(n, m) = exp(-d*d'*C);
    end
end
A(1:N1, N1+1:end) = 0;
A(N1+1:end, 1:N1) = 0;

D = sum(A);
Dr = 1.0 ./ sqrt(D);
Dr = diag(Dr);
L = Dr*A*Dr;

[V, D] = eig(L);
