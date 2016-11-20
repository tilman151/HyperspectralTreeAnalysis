function SVMModel = svmTrain(X, y)
%SVM_TRAIN Summary of this function goes here
%   Detailed explanation goes here

SVMModel = fitcsvm(X, y);



