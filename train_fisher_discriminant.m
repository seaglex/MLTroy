% train fisher's linear discriminant
% Input:
% X[N, D], X(n, :) is the nth feature vector
% T[N, 1], the label vector
% c, regularization
% Output:
% model, standard linear model w'*x + b
function[model] = train_fisher_discriminant(X, T, c)
I = T>0;
numPos = sum(I);
numNeg = length(I) - numPos;
mPos = sum(X(I, :)) / numPos;
mNeg = sum(X(~I, :)) / numNeg;
mAll = sum(X) / length(I);
difference = X(I, :) - ones(numPos, 1)*mPos;
Sw = difference' * difference;
difference = X(~I, :) - ones(numNeg, 1)*mNeg;
Sw = Sw + difference'*difference;

w = pinv(Sw + eye(2)*c) * (mPos-mNeg)';
b = -mAll * w;
model = struct('name', 'glm', 'w', w, 'b', b);