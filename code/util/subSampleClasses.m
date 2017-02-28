function [ subsampleIndices ] = subSampleClasses( labels, p)
%SUBSAMPLECLASSES generate a list of indices which represents a subset of the
%              given label list
%
%% Input:
%    labels .................. a list with the given labels for each
%                              instance
%    p ....................... the proportion of the subset


numData = numel(labels); %number of labels

    indices = 1:numData;
    availableLabels = num2cell(unique(labels));
    availableLabels = availableLabels(randperm(length(availableLabels), ceil(length(availableLabels)*p)));
    
    splitFunction = @(label) indices(labels == label)';

    subsampleIndices = ...
        cellfun(splitFunction, availableLabels, 'uniformoutput', 0);

    subsampleIndices = cat(1,subsampleIndices{:});

end

