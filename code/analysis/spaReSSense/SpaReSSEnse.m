classdef SpaReSSEnse < Classifier
    %SPARESSENS Spatially regularized semi-supervised ensemble
    %   
    %    Based on the scientific work presented in "Spatially regularized 
    %    semisupervised Ensembles of Extreme Learning Machines for 
    %    hyperspectral image segmentation." [Ayerdi, Marqués, Graña 2015].
    %    The algorithm works as follows:
    %    1. Unsupervised clustering is performed in the spectral domain.
    %    2. For each labeled training sample, the label is propagated to
    %    unlabeled pixels in the spatial neighborhood that belong to the
    %    same cluster.
    %    3. A classifier is trained on this enriched data set.
    %    4. The output image is regularized by assuming spatial smoothness.
    %
    %    Any classifier or ensemble of classifiers following the common 
    %    interface can be used internally. Furthermore, the label 
    %    propagation and the regularization step can optionally be 
    %    disabled.
    %
    %% Properties:
    %    classifier ......... Instance of a Classifier (supervised or 
    %                         semi-supervised, single or ensemble) to be
    %                         used internally.
    %    r .................. Float value. Radius defining the neighborhood
    %                         for label propagation. Needs to be lower than
    %                         half of the image size in either direction.
    %    doLabelPropagation . Boolean. Enables/disables the label
    %                         propagation step.
    %    doRegularization ... Boolean. Enables/disables the regularization
    %                         step.
    %
    %% Methods:
    %    SpaReSSEnse .. Constructor. Can take up to four input arguments:
    %        classifier ......... Set the classifier property.
    %        r .................. [Optional] Set the radius property.
    %        doLabelPropagation . [Optional] Set the label propagation
    %                             property.
    %        doRegularization ... [Optional] Set the regularization
    %                             property.
    %    trainOn ...... See documentation in superclass Classifier.
    %    classifyOn ... See documentation in superclass Classifier.
    %
    % Version: 2016-12-10
    % Author: Cornelius Styp von Rekowski
    %
    
    properties
        % The internally used classifier
        classifier;
        
        % Neighborhood information
        r = 5;
        relativeNeighbors;
        
        % Enabled/Disabled processing steps
        doLabelPropagation = true;
        doRegularization = true;
    end
    
    methods
        function obj = SpaReSSEnse(classifier, r, ...
                doLabelPropagation, doRegularization)
            
            % Classifier to be used internally
            obj.classifier = classifier;
            
            % Radius given?
            if nargin > 1
                obj.r = r;
            end
            
            % Compute relative neighbor positions once
            obj.relativeNeighbors = getRelativeNeighbors(obj.r);
            
            % Processing steps enabled/disabled?
            if nargin > 3
                obj.doLabelPropagation = doLabelPropagation;
                obj.doRegularization = doRegularization;
            end
        end
        
        function obj = trainOn(obj, trainFeatures, trainLabels)
            if obj.doLabelPropagation
                % Perform unsupervised clustering
                clusterIdxMap = clustering(trainFeatures, trainLabels);
                
                % Display clustered map
                visualizeLabels(clusterIdxMap, 'Clusters')
                
                % Propagate labels in spatial neighborhood for matching
                % clusters
                trainLabels = propagateLabels(...
                    trainLabels, clusterIdxMap, obj.relativeNeighbors);
            end
            
            % Train classifier on (enriched) data set
            obj.classifier.trainOn(trainFeatures, trainLabels);
        end
        
        function labels = classifyOn(obj, evalFeatures)
            % Predict labels
            labels = obj.classifier.classifyOn(evalFeatures);
            
            if obj.doRegularization
                % Regularize output labels based on spatial smoothness
                [x, y, ~] = size(evalFeatures);
                labelMap = vecToMap(labels, x, y);
                labels = regularize(labelMap, obj.relativeNeighbors);
            end
        end
    end
    
end


function neighborIdxs = getRelativeNeighbors(r)
    % Create relative indices of neighboring pixels in range
    neighborIdxs = [];
    for addX = -r : r
        % Use euclidean distance (based on sqrt(x^2 + y^2) = r)
        euclY = fix(sqrt(r^2 - addX^2));
        for addY = -euclY : euclY
            neighborIdxs = [neighborIdxs; addX addY];
        end
    end
end

function clusterIdxMap = clustering(featureMap, labelMap)
    % Reshape feature map to vector
    featureVec = mapToVec(featureMap);
    
    % Get number of classes (including unlabeled and empty pixels)
    k = numel(unique(labelMap));
    
    % Perform clustering
    clusterIdx = kmeans(featureVec, k);
    
    % Reshape resulting cluster indices to map representation
    [x, y] = size(labelMap);
    clusterIdxMap = vecToMap(clusterIdx, x, y);
end

function enrichedLabelMap = propagateLabels(labelMap, clusterIdxMap, ...
    relativeNeighbors)
    
    % Create map for counting neighboring labels
    labelCounts = initLabelCounts(labelMap);
    
    % Propagate labels from all labeled pixels
    for labeledIdx = find(labelMap > 0)'
        [labeledX, labeledY] = ind2sub(mapSize, labeledIdx);
        
        % Get assigned cluster and label
        labeledCluster = clusterIdxMap(labeledIdx);
        label = labelMap(labeledIdx);
        
        % Get neighbor indices
        neighbors = getValidNeighborIdxs(...
            relativeNeighbors, labeledX, labeledY, labelMap);
        
        % Get neighbor clusters
        neighborClusters = clusterIdxMap(neighbors);
        
        % Check for matching cluster assignment
        matchingNeighbors = neighbors(neighborClusters == labeledCluster);
        
        % Get subscripts
        [matchingNeighborsX, matchingNeighborsY] = ...
            ind2sub(mapSize, matchingNeighbors);
        
        % Get label count indices
        labelVector = ones(length(matchingNeighborsX), 1) * label;
        labelCountIdxs = sub2ind(size(labelCounts), ...
            matchingNeighborsX, matchingNeighborsY, labelVector);
        
        % Increase label counts
        labelCounts(labelCountIdxs) = labelCounts(labelCountIdxs) + 1;
    end
    
    % For each unlabeled pixel, assign the label with the highest count
    % received from the neighborhood
    enrichedLabelMap = labelMap;
    for unlabeledIdx = find(labelMap == 0)'
        enrichedLabelMap = setMaxLabel(...
            labelCounts, labelMap, unlabeledIdx, enrichedLabelMap);
    end
end

function regularizedLabelMap = regularize(labelMap, relativeNeighbors)
    % Create map for counting neighboring labels
    labelCounts = initLabelCounts(labelMap);
    
    % For each pixel that is not a fill pixel, count all labels occuring in
    % the neighborhood
    % TODO: Is it possible that there are unlabeled pixels?
    for curIdx = find(labelMap > -1)'
        [x, y] = ind2sub(mapSize, curIdx);
        
        % Get neighbor indices
        neighbors = ...
            getValidNeighborIdxs(relativeNeighbors, x, y, labelMap);
        
        % Get neighbor labels
        neighborLabels = labelMap(neighbors);
        
        % Count occurences of labels in neighborhood
        labelCounts(x, y, :) = histcounts(neighborLabels, 1:maxLabel+1);
    end
    
    % For each pixel that is not a fill pixel, assign the label with the 
    % highest count received from the neighborhood
    regularizedLabelMap = labelMap;
    for curIdx = find(labelMap > -1)'
        regularizedLabelMap = setMaxLabel(...
            labelCounts, labelMap, curIdx, regularizedLabelMap);
    end
end

function labelCounts = initLabelCounts(labelMap)
    % Create map for counting neighboring labels
    [maxX, maxY] = size(labelMap);
    maxLabel = max(max(labelMap));
    labelCounts = zeros(maxX, maxY, maxLabel);
end

function neighbors = getValidNeighborIdxs(relativeNeighbors, x, y, ...
    labelMap)
    
    % Get positions of neighboring pixels
    numNeighbors = size(relativeNeighbors, 1);
    neighbors = relativeNeighbors + repmat([x y], numNeighbors, 1);

    % Respect image boundaries
    [maxX, maxY] = size(labelMap);
    validPositions = min(neighbors, [], 2) > 0 & ...
        neighbors(:, 1) <= maxX & neighbors(:, 2) <= maxY;
    neighbors = neighbors(validPositions, :);

    % Transform subscripts to indices
    neighbors = sub2ind(size(labelMap), neighbors(:, 1), neighbors(:, 2));
end

function newMap = setMaxLabel(labelCounts, labelMap, curIdx, newMap)
    [x, y] = ind2sub(size(labelMap), curIdx);
    [count, label] = max(labelCounts(x, y, :));
    if count > 0
        newMap(curIdx) = label;
    end
end
