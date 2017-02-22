function regularizedLabelMap = regularize(labelMap, relativeNeighbors)
%REGULARIZE Summary of this function goes here
%   Detailed explanation goes here

    % Get logger
    logger = Logger.getLogger();
    
    % Init all needed data structures
    [separateImagesMask, imageIdxs, labelCounts, maxLabel] = ...
        initSpatialRegAlgorithms(labelMap);
    
    % Handle each partial image separately
    for imageIdx = imageIdxs
        logger.trace('regularize', ['Image ' num2str(imageIdx)]);
        
        % Mask everything but the current image
        maskedLabelMap = labelMap;
        maskedLabelMap(separateImagesMask ~= imageIdx) = -1;
        
        % Handle all pixels of a label at the same time
        for label = 1:maxLabel
            logger.trace('regularize', ['Label ' num2str(label)]);
            
            % Get subscripts of all pixels for this label
            [labelsX, labelsY] = ...
                ind2sub(size(labelMap), find(maskedLabelMap == label));
            
            % Check if there are any pixels for this label
            if ~isempty(labelsX)
                % Get subscripts of all neighbors of the labeled pixels
                neighborSubs = arrayfun(...
                    @(x, y) getValidNeighborSubs(...
                        relativeNeighbors, x, y, size(labelMap)), ...
                    labelsX, labelsY, 'UniformOutput', false);
                clear labelsX labelsY;

                % neighborSubs is cell array -> Concatenate to one list
                neighborSubs = cell2mat(neighborSubs);

                % Accumulate counts for single positions because they can
                % be in the list multiple times
                accumNeighbors = ...
                    accumarray(neighborSubs, 1, size(labelMap));
                
                % Only count neighbors in the current image
                accumNeighbors(maskedLabelMap == -1) = 0;
                
                % Add counts to total sum
                labelCounts(:, :, label) = ...
                    labelCounts(:, :, label) + accumNeighbors;
                clear neighborIdxs;
            end
        end
    end
    
    % For each pixel that is not a fill pixel, assign the label with the 
    % highest count received from the neighborhood
    regularizedLabelMap = labelMap;
    if maxLabel > 0
        [~, maxLabels] = max(labelCounts, [], 3);
        validIdxs = find(labelMap > -1);
        regularizedLabelMap(validIdxs) = maxLabels(validIdxs);
    end
end

function neighbors = getValidNeighborSubs(relativeNeighbors, x, y, ...
    labelMapSize)
    
    % Get positions of neighboring pixels
    numNeighbors = size(relativeNeighbors, 1);
    neighbors = relativeNeighbors + repmat([x y], numNeighbors, 1);

    % Respect image boundaries
    validPositions = min(neighbors, [], 2) > 0 & ...
        neighbors(:, 1) <= labelMapSize(1) & neighbors(:, 2) <= labelMapSize(2);
    neighbors = neighbors(validPositions, :);
end
