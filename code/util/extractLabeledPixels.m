function [featureList, labelList, unlabeledFeatureList] = ...
    extractLabeledPixels(featureMap, labelMap)
%EXTRACTLABELEDPIXELS Extract labeled pixels from the given feature map
%
%    Produce the list of labeled pixels and their corresponding labels from
%    the given spatial maps. The information on their location is lost in
%    that step. Unlabeled pixels are returned as a separate list. Empty 
%    pixels that were used to fill the image are dropped.
%
%% Input:
%    featureMap ........... 
%    labelMap ............. 
%
%% Output:
%    featureList .......... 
%    labelList ............ 
%    unlabeledFeatureList . 
%
% Version: 2016-11-24
% Author: Cornelius Styp von Rekowski
%

% Get logical positions of all labeled pixels (label > 0) and unlabeled 
% pixels (label = 0), excluding empty pixels (label = -1)
labeledPos = labelMap > 0;
unlabeledPos = labelMap == 0;

% Extract list of labels
labelList = labelMap(labeledPos);

% Create empty feature lists
numLabeled = nnz(labeledPos);
numUnlabeled = nnz(unlabeledPos);
numFeatures = size(featureMap, 3);
featureList = zeros(numLabeled, numFeatures);
unlabeledFeatureList = zeros(numUnlabeled, numFeatures);

% For each feature dimension, extract values for the labeled pixels and
% unlabeled pixels
for feature = 1:numFeatures
    singleFeatureMap = featureMap(:, :, feature);
    featureList(:, feature) = singleFeatureMap(labeledPos);
    unlabeledFeatureList(:, feature) = singleFeatureMap(unlabeledPos);
end

end
