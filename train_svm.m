% train SVM
% Input:
% X[N, D], X(n, :) is the nth feature vector
% T[N, 1], the label vector (-1/1)
% c, regularization
% kf, kernel function that takes X[N, D] and y[D, 1] or Y[D, M]
% Output:
% model, x' * \sum{alpha_i*y_i*x_i} + b
% Optimization - original:
% min 0.5*w'w + C*\sum{\xi_i}
% s.t. y_i*(w'x_i+b) >= 1-\xi_i
%      \xi_i >= 0
% Optimization - dual:
% min_\alpha -{\sum_i{alpha_i} + 0.5*\sum_{ij}{alpha_i*alpha_j*y_i*y_j*x_i*x_j}
% s.t. 0 <= alpha_i <= C
%      \sum{alpha_i * y_i} = 0
function[model] = train_svm(X, T, c, kf)
if nargin == 3
    kf = @(x, y) x*y
end
[num, dim] = size(X);
K = kf(X, X').*(T*T');
A = [-eye(num); eye(num)];
B = [zeros(num, 1); c*ones(num, 1)];
Aeq = ones(1, num);
Beq = 0;
alpha0 = zeros(num, 1);
alpha = fmincon(@(a) dual_cost(a, K), alpha0, A, B, Aeq, Beq);

I_sv = find(alpha ~= 0);
I = find(alpha ~= c);
B = zeros(length(I), 1);
for n = 1:length(I)
    i = I(n);
    acc = 0;
    for m = 1:length(I_sv)
        j = I_sv(m);
        acc = alpha(j) * y(j) * K(j, i);
    end
    B(n) = acc-1;
end
b = min(B);
if nargin == 3
    w = sum((alpha(I_sv).*T(I_sv)*ones(1, dim)) .* X(I_sv, :))';
    model = struct('name', 'glm', 'w', w, 'b', b);
else
    model = struct('name', 'svm', ...
    'alpha', alpha(I_sv), 'label', T(I_sv), 'sv', X(I_sv, :), 'b', b, ...
    'kernel_function', kf);
end

function[cost] = dual_cost(alpha, K)
cost = 0.5*alpha'*K*alpha - sum(alpha);
