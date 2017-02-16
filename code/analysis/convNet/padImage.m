function [featureCube, labelMap] = padImage(featureCube, labelMap, r)
%PADIMAGE Summary of this function goes here
%   Detailed explanation goes here
    
    % Replace fill pixels values with 0
    featureCube = replaceFillPixels(featureCube, labelMap);
    
    % Extend image boundaries to width r
    [featureCube, labelMap] = extendBoundaries(featureCube, labelMap, r);
    
    % Pad the whole image with a frame of size r
    [featureCube, labelMap] = addFrame(featureCube, labelMap, r);
end

function featureCube = replaceFillPixels(featureCube, labelMap)
    % Find fill pixels
    [fillPixelSubs(:, 1), fillPixelSubs(:, 2)] = find(labelMap == -1);
    numFillPixels = size(fillPixelSubs, 1);
    
    % Copy fill pixel subs for each feature dimension
    numFeatures = size(featureCube, 3);
    fillPixelSubs = repmat(fillPixelSubs, [numFeatures, 1]);
    
    % Create third subscript covering each feature dimension
    featureSubs = repmat(1:numFeatures, [numFillPixels, 1]);
    
    % Transform subscripts to indices in featureCube
    fillPixelIndices = sub2ind(size(featureCube), ...
        fillPixelSubs(:, 1), fillPixelSubs(:, 2), featureSubs(:));
    
    % Replace all values by 0 (mean value in the normalized data)
    featureCube(fillPixelIndices) = 0;
end

function [newFeatureCube, newLabelMap] = ...
    extendBoundaries(featureCube, labelMap, r)
    
    [x, y, numFeatures] = size(featureCube);
    
    % Get image boundary indices
    bounds = [find(all(labelMap == -1, 1)), y];
    numBounds = length(bounds) - 1;
    
    % Create new data structures for extended boundary size
    newWidth = y + numBounds*(r-1);
    newFeatureCube = zeros(x, newWidth, numFeatures);
    newLabelMap = -ones(x, newWidth);
    
    % Copy image up to first boundary
    newFeatureCube(:, 1:bounds(1), :) = featureCube(:, 1:bounds(1), :);
    newLabelMap(:, 1:bounds(1)) = labelMap(:, 1:bounds(1));
    
    % Copy all remaining images
    for ii = 1:numBounds
        % Increase boundary size by r-1
        left = bounds(ii) + ii*(r-1) + 1;
        right = bounds(ii + 1) + ii*(r-1);
        
        % Copy image up to next boundary
        newFeatureCube(:, left:right, :) = ...
            featureCube(:, bounds(ii)+1 : bounds(ii+1), :);
        newLabelMap(:, left:right) = ...
            labelMap(:, bounds(ii)+1 : bounds(ii+1));
    end
end

function [newFeatureCube, newLabelMap] = addFrame(featureCube, labelMap, r)
    [x, y, numFeatures] = size(featureCube);
    
    % Calculate new image size with frame of size r
    newHeight = x + 2*r;
    newWidth = y + 2*r;
    
    % Create new data structures for new image size
    newFeatureCube = zeros(newHeight, newWidth, numFeatures);
    newLabelMap = -ones(newHeight, newWidth);
    
    % Copy values
    newFeatureCube(r+1 : r+x, r+1 : r+y, :) = featureCube;
    newLabelMap(r+1 : r+x, r+1 : r+y) = labelMap;
end
