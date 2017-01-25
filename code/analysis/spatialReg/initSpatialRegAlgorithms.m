function [separateImagesMask, imageIdxs, labelCounts, maxLabel] = ...
    initSpatialRegAlgorithms(labelMap)
    %INITSPATIALREGALGORITHMS Summary of this function goes here
    %   Detailed explanation goes here
    
    % Calculate separate image masks regarding -1 columns
    separateImagesMask = calculateImageBoundaryMask(labelMap);
    imageIdxs = 0:max(separateImagesMask(:));
    
    % Create map for counting neighboring labels
    [maxX, maxY] = size(labelMap);
    maxLabel = max(labelMap(:));
    labelCounts = zeros(maxX, maxY, maxLabel);
end

function separateImagesMask = calculateImageBoundaryMask(labelMap)
    dividingColumns = all(labelMap == -1, 1);
    imageIdxs = cumsum(dividingColumns);
    separateImagesMask = repmat(imageIdxs, [size(labelMap, 1) 1]);
end
