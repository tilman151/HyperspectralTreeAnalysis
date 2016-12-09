function [out] = trainBaseClassifier( baseClassifier, trainLabels, trainFeatures )
%TRAINBASECLASSIFIER trains a classifier with the provided parameter
%
%% Input:
%    baseClassifier . the classifier which should be trained
%    trainLabels .... the labels, which are used for training
%    trainFeatures .. the features, which are used to train
%
%% Output:
%    out ............ the classifier
% 
% Version: 2016-12-05
% Author: Tuan Pham Minh
%

    out = baseClassifier.trainOn(trainLabels, trainFeatures);
end

