function [ score ] = rateBaseClassifier( baseClassifier, trainFeatures2D , trainLabel1D )
%RATEBASECLASSIFIER rate a classifier with the provided parameter
%
%% Input:
%    baseClassifier ............. the classifier which should be trained
%    trainFeatures2D ............ the features used for training in a 
%                                 nxf matrix with n being the number of
%                                 instances and f the number of features
%    trainLabel1D ............... the labels for the features in
%                                 trainFeatures2D in a nx1 matrix
%
%% Output:
%    score ............ the rating of the classifier
% 
% Version: 2017-01-31
% Author: Tuan Pham Minh
%
    features = permute(trainFeatures2D, [1,3,2]);
    labels = trainLabel1D(subSampleIndices);
    predictedLabels = baseClassifier.classifyOn( features, labels);
    
    
    availableLabels = unique([labels predictedLabels]);
    confMat = confusionmat(labels, predictedLabels, 'order', availableLabels);
    score = Evaluator.getAccuracy(confMat);
end

