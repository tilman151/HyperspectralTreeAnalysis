function [ label ] = classifyOnAll(classifier, evalFeatures)
%CLASSIFYONALL classify the data with the given classifier
%
%% Input:
%    classifier . the classifier which should be used to predict labels
%    fIdx ....... the indices for the features, which should be classified
%    allF ....... all available features
%
%% Output:
%    out ........ a label vector
% 
% Version: 2016-12-05
% Author: Tuan Pham Minh
%
label = classifier.classifyOn(evalFeatures);
end

