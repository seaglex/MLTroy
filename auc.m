% Area under (ROC) curve
% Input:
% T[N, 1], labels
% scores[N, 1], scores
% Output:
% AUC, 0.5 for random classificaiton, 1 for perfect classification
function[AUC]=auc(labels, scores)
[scores, I] = sort(scores, 'descend');
labels = labels(I);
I = labels>0;
total = length(labels);
num_pos = sum(I);
num_neg = total - num_pos;
acc_leftneg_numppos = 0.0;  % \sum{ p * (left_neg + num_pos) }
acc_pos = 0.0;
acc_neg = 0.0;
np = 0;
nn = 0;
last_score = scores(1)-1;
for i = 1:length(I)
    if scores(i) ~= last_score
        acc_leftneg_numppos = acc_leftneg_numppos + ...
            np * (total - acc_neg - nn*0.5);
        acc_pos = acc_pos + np;
        acc_neg = acc_neg + nn;
        np = 0;
        nn = 0;
        last_score = scores(i);
    end
    if labels(i) > 0
        np = np + 1;
    else
        nn = nn + 1;
    end
end
acc_leftneg_numppos = acc_leftneg_numppos + ...
    np * (total - acc_neg - nn*0.5);
acc_pos = acc_pos + np;
acc_neg = acc_neg + nn;
if acc_pos * acc_neg == 0
    AUC = 0.5;
    return
end
AUC = acc_leftneg_numppos / (acc_pos*acc_neg) - acc_pos / acc_neg;
return;