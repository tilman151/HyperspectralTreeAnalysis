function [ subsampleIndices ] = subSampleData( labels, p, ...
                                               remainClassDistribution)
%SUBSAMPLEDATA generate a list of indices which represents a subset of the
%              given label list
%
%% Input:
%    labels .................. a list with the given labels for each
%                              instance
%    p ....................... the proportion of the subset
%    remainClassDistribution . set true if the class
%                              distribution for the 
%                              training subsamples should 
%                              be approximately the same as
%                              the original distribution

numData = numel(labels); %number of labels

if remainClassDistribution
    indices = 1:numData;
    availableLabels = num2cell(unique(labels));

    selectionFunction = @(i,p) i(randperm(numel(i), ceil(numel(i) * p)))';
    splitFunction = @(label) selectionFunction(indices(labels == label), p);

    subsampleIndices = ...
        cellfun(splitFunction, availableLabels, 'uniformoutput', 0);

    subsampleIndices = cat(1,subsampleIndices{:});

else 
    subsampleIndices = randperm(numData, ceil(numData*p));
end

end

