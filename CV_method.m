
function [trData,ttData]=CV_method(data)

[M,N]=size(data);
indices=crossvalind('Kfold',data(1:M,N),5);
for k=1:5
  test = (indices == k);
  train = ~test;
  trData{k}=data(train,:);
%   train_target=target(:,train);
  ttData{k}=data(test,:);
%   test_target=target(:,test);
%  [HammingLoss(1,k),RankingLoss(1,k),OneError(1,k),Coverage(1,k),Average_Precision(1,k),Outputs,Pre_Labels.MLKNN]=MLKNN_algorithm(train_data,train_target,test_data,test_target);
end


end

