function [ features, labels, unlabeledFeatures ] = generateSampleSet( directoryPath, selectionProportion, outputpath)
%GENERATESAMPLESET Generates a sampleset for a given data directory
%
%%  Input:
%       directoryPath ....... a path to the Hyperspectral data directory.
%       selectionProportion . a number betwee 0 and 1, which specifies how
%                             much data should be taken into the set
%       outputpath .......... a path to the file, where the data should be
%                             saved
%                             if none is given the file will not be saved
%
% Version: 2016-12-10
% Author: Tuan Pham Minh


% check for the relevant data files

fileContent = whos('-file', [directoryPath, 'features.mat']);
variableName = {fileContent.name};
matchFunctionHandle = @(filename) matchFilename(filename, 'F2G9_');
nameMask = cellfun(matchFunctionHandle, variableName);
variableName = variableName(nameMask);

% load all data
loadingHandle = @(variableName)removeEmptyInstances([directoryPath 'features.mat'], [directoryPath 'labels.mat'], variableName);
allFeatures = cellfun(loadingHandle, variableName, 'uniformoutput', 0);
allFeatures = horzcat(allFeatures{:});
allFeatures = reshape(allFeatures, 2, [])';

allLabels = allFeatures(:,2);
allFeatures = allFeatures(:,1);

allFeatures = vertcat(allFeatures{:});
allLabels = vertcat(allLabels{:});

% get all classes
classes = unique(allLabels);

% split features by classes
splitFeatures = cell(size(classes));
splitFeatureIndices = cell(size(classes));

for i = 1:length(classes)
    splitFeatures{i} = allFeatures(allLabels == classes(i), :);
    splitFeatureIndices{i} = size(splitFeatures{i}, 1);
end

% delete the previous loaded data to save ram
allFeatures = [];

% generate random indices to select the data
splitFeatureIndices = cellfun(@(numFeatures, proportion) (randperm(numFeatures, floor(proportion * numFeatures))), ...
    splitFeatureIndices, ...
    repmat({selectionProportion}, size(splitFeatureIndices)), ...
    'uniformoutput', 0);
splitFeatures = cellfun(@(features, indices) (features(indices, :)), ...
    splitFeatures, ...
    splitFeatureIndices, ...
    'uniformoutput', 0);

% extract unlabeled features (class 0)
unlabeledFeatures = splitFeatures{classes == 0}; 
splitFeatures(classes == 0) = [];
splitFeatureIndices(classes == 0) = [];

classesWithoutZero = classes(classes ~= 0);

% concatenate the remaining features
features = vertcat(splitFeatures{:});
labels = cellfun(@(f, l)(repmat(l, size(f))), splitFeatureIndices, num2cell(classesWithoutZero), 'uniformoutput', 0);
labels = horzcat(labels{:})';

% save data to the specified file
if(exist('outputpath', 'var'))
    save(outputpath, 'features', 'labels', 'unlabeledFeatures');
end
end

function [output] = removeEmptyInstances(featureFilePath, labelFilePath, variableName)
% load features and labels
feature = load(featureFilePath, variableName);
labels = load(labelFilePath, variableName);
feature = feature.(variableName);
labels = labels.(variableName);

% reshape to get from 3d to 2d
[x,y,numFeatures] = size(feature);
feature = reshape(feature, x*y, numFeatures);
labels = reshape(labels, x*y, 1);
% remove data with no information
feature = feature(labels >= 0, :);

labels = labels(labels >= 0);

output = {feature, labels};
end

function matched = matchFilename(filename, regex)
res = regexp(filename, regex);
matched = ~isempty(res);
end