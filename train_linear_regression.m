% train linear regression
% Input:
% X[N, D], X(n, :) is the nth feature vector
% T[N, 1], the label vector
% c, regularization
% Output:
% model, standard linear model w'*x + b
function[model] = train_linear_regression(X, T, c)
num = size(X, 1);
mX = sum(X) / num;
mT = sum(T) / num;
difference = X - ones(num, 1)*mX;
Cxx = difference' * difference;
Cxy = difference' * (T-mT);
w = pinv(Cxx + eye(2)*c) * Cxy;
b = mX*w + mT;
model = struct('w', w, 'b', b);