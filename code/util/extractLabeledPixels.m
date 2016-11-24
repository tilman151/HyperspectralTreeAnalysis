function [featureList, labelList] = ...
    extractLabeledPixels(featureMap, labelMap)
%EXTRACTLABELEDPIXELS Extract labeled pixels from the given feature map
%
%    Produce the list of labeled pixels and their corresponding labels from
%    the given spatial maps. The information on their location is lost in
%    that step. Unlabeled pixels and empty pixels that were used to fill 
%    the image are dropped.
%
%% Input:
%    featureMap .. 
%    labelMap .... 
%
%% Output:
%    featureList . 
%    labelList ... 
%
% Version: 2016-11-24
% Author: Cornelius Styp von Rekowski
%

% Logical positions of all labeled pixels, excluding unlabeled and empty 
% pixels (0 and -1)
labeledPos = labelMap > 0;

% Extract list of labels
labelList = labelMap(labeledPos);

% Create empty feature list
numLabels = nnz(labeledPos);
numFeatures = size(featureMap, 3);
featureList = zeros(numLabels, numFeatures);

% For each feature dimension, extract values for the labeled pixels
for feature = 1:numFeatures
    singleFeatureMap = featureMap(:, :, feature);
    featureList(:, feature) = singleFeatureMap(labeledPos);
end

end
