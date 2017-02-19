function [out] = trainBaseClassifier( baseClassifier, trainFeatures2D , trainLabel1D, trainingInstanceProportion, remainClassDistribution)
%TRAINBASECLASSIFIER trains a classifier with the provided parameter
%
%% Input:
%    baseClassifier ............. the classifier which should be trained
%    trainFeatures2D ............ the features used for training in a 
%                                 nxf matrix with n being the number of
%                                 instances and f the number of features
%    trainLabel1D ............... the labels for the features in
%                                 trainFeatures2D in a nx1 matrix
%    trainingInstanceProportion . a scalar between 0 and 1, which describes
%                                 how much of the data the classifier
%                                 should receive
%
%% Output:
%    out ............ the classifier
% 
% Version: 2016-12-23
% Author: Tuan Pham Minh
%
    subSampleIndices = ...
        subSampleData(trainLabel1D, trainingInstanceProportion, remainClassDistribution);
    randomizedFeatures = permute(trainFeatures2D(subSampleIndices,:), ...
                                 [1,3,2]);
    randomizedLabels = trainLabel1D(subSampleIndices);
    out = baseClassifier.trainOn( randomizedFeatures, randomizedLabels);
end

