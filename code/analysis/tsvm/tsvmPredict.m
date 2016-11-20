function label = tsvmPredict(TSVMModel, X)
%TSVMPREDICT Summary of this function goes here
%   Detailed explanation goes here

label = TSVMModel.f(X);
