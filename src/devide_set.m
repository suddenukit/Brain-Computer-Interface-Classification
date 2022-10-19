function [train_set, test_set] = devide_set(class1, class2, test_num, fold_num)
% This function devide the 2 sample set to training and test set
%
% INPUTS:
% test_num: the number of section which is participated into test set
% fold_num: the number of cross validation folds
len = size(class1, 2);
test_num = mod(test_num, fold_num);
test_len = len/fold_num;
test_set = [class1(:, 1+test_num*test_len : (test_num+1)*test_len) class2(:, 1+test_num*test_len : (test_num+1)*test_len)];
class1(:, 1+test_num*test_len : (test_num+1)*test_len) = [];
class2(:, 1+test_num*test_len : (test_num+1)*test_len) = [];
train_set = [class1 class2];
end