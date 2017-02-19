function enrichedLabelMap = propagateLabels(labelMap, clusterIdxMap, ...
    relativeNeighbors, propagationThreshold)
%PROPAGATELABELS Summary of this function goes here
%   Detailed explanation goes here
    
    % Get logger
    logger = Logger.getLogger();
    
    % Init all needed data structures
    [separateImagesMask, imageIdxs, labelCounts, maxLabel] = ...
        initSpatialRegAlgorithms(labelMap);
    
    % Handle each partial image separately
    for imageIdx = imageIdxs
        logger.trace('propagateLabels', ['Image ' num2str(imageIdx)]);
        
        % Mask everything but the current image
        maskedLabelMap = labelMap;
        maskedLabelMap(separateImagesMask ~= imageIdx) = -1;
        
        % Handle all pixels of a label at the same time
        for label = 1:maxLabel
            logger.trace('propagateLabels', ['Label ' num2str(label)]);
            
            % Get indexes of all pixels for this label
            labelIdxs = find(maskedLabelMap == label);
            
            % Check if there are any pixels for this label
            if ~isempty(labelIdxs)
                % Transform indexes to subscripts
                [labelsX, labelsY] = ind2sub(size(labelMap), labelIdxs);

                % Get assigned clusters
                labeledClusters = clusterIdxMap(labelIdxs);
                clear labelIdxs;

                % Get subscripts of all neighbors of the labeled pixels if
                % they are assigned to the same cluster
                neighborSubs = arrayfun(...
                    @(x, y, c) getValidNeighborSubsFromCluster(...
                        relativeNeighbors, x, y, c, clusterIdxMap), ...
                    labelsX, labelsY, labeledClusters, ...
                    'UniformOutput', false);
                clear labelsX labelsY labeledClusters;
                
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
    
    % Calculate number of neighbors needed for propagation
    minNeighbors = propagationThreshold * length(relativeNeighbors);
    
    % For each unlabeled pixel, assign the label with the highest count
    % received from the neighborhood, if the count is high enough
    enrichedLabelMap = labelMap;
    if maxLabel > 0
        [maxCounts, maxLabels] = max(labelCounts, [], 3);
        validIdxs = find(labelMap == 0 & maxCounts > minNeighbors);
        enrichedLabelMap(validIdxs) = maxLabels(validIdxs);
    end
end

function neighbors = getValidNeighborSubsFromCluster(...
    relativeNeighbors, x, y, c, clusterIdxMap)
    
    % Get positions of neighboring pixels
    numNeighbors = size(relativeNeighbors, 1);
    neighbors = relativeNeighbors + repmat([x y], numNeighbors, 1);

    % Respect image boundaries
    [maxX, maxY] = size(clusterIdxMap);
    validPositions = min(neighbors, [], 2) > 0 & ...
        neighbors(:, 1) <= maxX & neighbors(:, 2) <= maxY;
    neighbors = neighbors(validPositions, :);
    
    % Only return neighbors of same cluster
    neighborIdxs = sub2ind(size(clusterIdxMap), ...
        neighbors(:, 1), neighbors(:, 2));
    sameClusterPositions = clusterIdxMap(neighborIdxs) == c;
    neighbors = neighbors(sameClusterPositions, :);
end
