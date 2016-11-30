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
%    featureMap ........... Map of features for each image pixel.
%                           Dimensions X x Y x F with X and Y being the
%                           image dimensions and F being the number of
%                           features.
%    labelMap ............. Map of labels for each image pixel. Dimensions
%                           X x Y x 1 with X and Y being the image
%                           dimensions.
%
%% Output:
%    featureList .......... List of features for the labeled samples.
%                           Dimensions L x F with L being the number of
%                           labeled samples and F being the number of
%                           features.
%    labelList ............ List of labels. Dimensions L x 1 with L being
%                           the number of labeled samples.
%    unlabeledFeatureList . List of features for the unlabeled samples.
%                           Dimensions U x F with U being the number of
%                           unlabeled samples and F being the number of
%                           features.
%
% Version: 2016-11-30
% Author: Cornelius Styp von Rekowski
%

% TODO: Maybe it is easier to first reshape to 2D and then extract labeled
% data

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
