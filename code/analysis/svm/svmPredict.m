function label = svmPredict(SVMModel, X)
%SVMPREDICT Summary of this function goes here
%   Detailed explanation goes here

label = predict(SVMModel, X)

