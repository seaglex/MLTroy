function[] = test_linear_classifiers()
c = 0;
model_trainers = {
    @(X, T) train_linear_regression(X, T, c);
    @(X, T) train_fisher_discriminant(X, T, c);
    @(X, T) train_logistic_regression(X, T, c);
    };
fmts = {'g', 'm', 'k'};
names = {'linear regression', 'Fisher''s linear discriminant', ...
    'logistic regression'};
data_generators = {
    @() anti_linear_regression();
    @() anti_fisher_discriminant();
    @() anti_logistic_regression();
    };
titles = {'data1', 'data2', 'data3'};

close all;
AUCs = zeros(length(data_generators), length(model_trainers));
for d = 1:length(data_generators)
    generator = data_generators{d};
    [X, labels] = generator();
    I = labels>0;
    pos = X(I, :);
    neg = X(~I, :);
    figure(d);
    scatter(pos(:, 1), pos(:, 2), 'r.');
    hold on;
    scatter(neg(:, 1), neg(:, 2), 'b.');
    for m = 1:length(model_trainers)
        trainer = model_trainers{m};
        model = trainer(X, labels);
        plot_linear_model(model, 1, fmts{m});
        AUCs(d, m) = auc(labels, linear_model(model, X));
    end
    legend([{'Pos', 'Neg'}, names]);
    title(titles(d));
    grid on;
    box off;
    hold off;
end
AUCs

function[X, labels] = anti_logistic_regression()
num_pos = 1000;
num_neg = 1000;
pos = [get_data([0, 0.2], [1, 0.1], num_pos*9/10);
    get_data([1, -1], [0.1, 0.1], num_pos/10);
    get_data([0, 1], [0.1, 0.1], num_pos/10);
    ];
neg = [get_data([0, -0.2], [1, 0.1], num_neg*9/10);
    get_data([-1, 1], [0.1, 0.1], num_neg/10)
    get_data([0, -1], [0.1, 0.1], num_neg/10)
    ];
X = [pos; neg];
labels = [ones(length(pos), 1); -ones(length(neg), 1)];

function[X, labels] = anti_linear_regression()
num_pos = 1000;
num_neg = 1000;
pos = [get_data([-0.5, 0.5], [0.45, 0.45], num_pos*9/10);
    get_data([0.5, 0.5], [0.45, 0.45], num_pos/10)];
neg = [get_data([0.5, -0.5], [0.45, 0.45], num_neg*9/10);
    get_data([-0.5, -0.5], [0.45, 0.45], num_neg/10)];
X = [pos; neg];
labels = [ones(length(pos), 1); -ones(length(neg), 1)];

function[X, labels] = anti_fisher_discriminant()
num_pos = 200;
num_neg = 200;
pos = [get_data([-0.5, 0.5], [0.01, 0.8], num_pos/2)
    get_data([0.5, 0.5], [0.01, 0.1], num_pos/2)];
neg = [get_data([-0.5, -0.5], [0.01, 0.1], num_neg/2)
    get_data([0.5, -0.5], [0.01, 0.8], num_neg/2)];
X = [pos; neg];
labels = [ones(length(pos), 1); -ones(length(neg), 1)];

function[X] = get_data(means, scales, num)
X = 2*rand(num, 2)-1;
X = X .* [ones(num, 1)*scales(1), ones(num, 1)*scales(2)];
X = X + ones(num, 1) * means;