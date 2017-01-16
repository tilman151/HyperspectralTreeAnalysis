classdef SpatialReg < Classifier
    %SPATIALREG Spatially regularized semi-supervised classifier
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
    %    SpatialReg .... Constructor. Can take up to four input arguments:
    %        classifier ......... Set the classifier property.
    %        r .................. [Optional] Set the radius property.
    %        doLabelPropagation . [Optional] Set the label propagation
    %                             property.
    %        doRegularization ... [Optional] Set the regularization
    %                             property.
    %    toString ...... See documentation in superclass Classifier.
    %    toShortString . See documentation in superclass Classifier.
    %    trainOn ....... See documentation in superclass Classifier.
    %    classifyOn .... See documentation in superclass Classifier.
    %
    % Version: 2016-12-22
    % Author: Cornelius Styp von Rekowski
    %
    
    properties
        % The internally used classifier
        classifier;
        
        % Neighborhood radius
        r = 5;
        
        % Enabled/Disabled processing steps
        doLabelPropagation = true;
        doRegularization = true;
    end
    
    properties(Hidden=true)
        % Cached neighborhood indices
        relativeNeighbors;
    end
    
    methods
        function obj = SpatialReg(classifier, r, ...
                doLabelPropagation, doRegularization)
            
            % Classifier to be used internally
            obj.classifier = classifier;
            
            % Radius given?
            if nargin > 1
                obj.r = r;
            end
            
            % Compute relative neighbor positions for this radius once
            obj.relativeNeighbors = getRelativeNeighbors(obj.r);
            
            % Processing steps enabled/disabled?
            if nargin > 3
                obj.doLabelPropagation = doLabelPropagation;
                obj.doRegularization = doRegularization;
            end
        end
        
        function str = toString(obj)
            % Create output string with class name and classifier
            str = ['SpatialReg (classifier: ' obj.classifier.toString()];
            
            % Append neighborhood radius
            str = [str ', r: ' num2str(obj.r)];
            
            % Append processing steps
            str = [str ', doLabelPropagation: ' ...
                   num2str(obj.doLabelPropagation)];
            str = [str ', doRegularization: ' ...
                   num2str(obj.doRegularization)];
            
            % Close parentheses
            str = [str ')'];
        end
        
        function str = toShortString(obj)
            % Create output string with classifier representation
            str = [obj.classifier.toShortString() '_'];
            
            % Append processing steps
            if obj.doLabelPropagation
                str = [str '_labelPropagation'];
            end
            if obj.doRegularization
                str = [str '_regularization'];
            end
            
            % Append neighborhood radius
            str = [str '_r' num2str(obj.r)];
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            if obj.doLabelPropagation
                % Perform unsupervised clustering
                clusterIdxMap = ...
                    clustering(trainFeatureCube, trainLabelMap);
                
                % Display clustered map
                visualizeLabels(clusterIdxMap, 'Clusters')
                disp('Propagating labels..');
                % Propagate labels in spatial neighborhood for matching
                % clusters
                trainLabelMap = propagateLabels(...
                    trainLabelMap, clusterIdxMap, obj.relativeNeighbors);
                
                % Display enriched label map
                visualizeLabels(trainLabelMap, 'Enriched Training Labels')
            end
            
            % Train classifier on (enriched) data set
            obj.classifier.trainOn(trainFeatureCube, trainLabelMap);
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap)
            
            % Predict labels
            predictedLabelMap = ...
                obj.classifier.classifyOn(evalFeatureCube, maskMap);
            
            if obj.doRegularization
                % Display result before regularization
                visualizeLabels(predictedLabelMap, ...
                    'Predicted labels before regularization')
                
                % Regularize output labels based on spatial smoothness
                predictedLabelMap = ...
                    regularize(predictedLabelMap, obj.relativeNeighbors);
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

function clusterIdxMap = clustering(featureCube, labelMap)
    % Extract valid pixels as list
    featureList = validListFromSpatial(featureCube, labelMap);
    labelList = validListFromSpatial(labelMap, labelMap);
    
    % Get number of classes (excluding fill pixels)
    k = numel(unique(labelList));
    
    % Perform clustering
    clusterIdxList = kmeans(featureList, k, 'MaxIter', 1000);
    
    % Reshape resulting cluster indices to map representation
    clusterIdxMap = rebuildMap(clusterIdxList, labelMap);
end

function enrichedLabelMap = propagateLabels(labelMap, clusterIdxMap, ...
    relativeNeighbors)
    
    % Calculate separate image masks regarding -1 columns
    separateImagesMask = calculateImageBoundaryMask(labelMap);
    
    % Create map for counting neighboring labels
    labelCounts = initLabelCounts(labelMap);
    
    % Propagate labels from all labeled pixels
    for labeledIdx = find(labelMap > 0)'
        [labeledX, labeledY] = ind2sub(size(labelMap), labeledIdx);
        
        % Get assigned cluster and label
        labeledCluster = clusterIdxMap(labeledIdx);
        label = labelMap(labeledIdx);
        
        % Get neighbor indices
        neighbors = getValidNeighborIdxs(...
            relativeNeighbors, labeledX, labeledY, labelMap);
        
        % Mask everything but the image that the current pixel belongs to
        maskedClusterMap = applySeparateImagesMask(...
            clusterIdxMap, separateImagesMask, labeledY);
        
        % Get neighbor clusters
        neighborClusters = maskedClusterMap(neighbors);
        
        % Check for matching cluster assignment
        matchingNeighbors = neighbors(neighborClusters == labeledCluster);
        
        % Get subscripts
        [matchingNeighborsX, matchingNeighborsY] = ...
            ind2sub(size(labelMap), matchingNeighbors);
        
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

function separateImagesMask = calculateImageBoundaryMask(labelMap)
    dividingColumns = all(labelMap == -1, 1);
    imageIdxs = cumsum(dividingColumns);
    separateImagesMask = repmat(imageIdxs, [size(labelMap, 1) 1]);
end

function maskedMap = ...
    applySeparateImagesMask(inputMap, separateImagesMask, curColumn)
    
    otherImgsMask = separateImagesMask ~= separateImagesMask(1, curColumn);
    maskedMap = inputMap;
    maskedMap(otherImgsMask) = -1;
end

function regularizedLabelMap = regularize(labelMap, relativeNeighbors)
    % Calculate separate image masks regarding -1 columns
    separateImagesMask = calculateImageBoundaryMask(labelMap);
    
    % Create map for counting neighboring labels
    [labelCounts, maxLabel] = initLabelCounts(labelMap);
    
    % For each pixel that is not a fill pixel, count all labels occuring in
    % the neighborhood
    for curIdx = find(labelMap > -1)'
        [x, y] = ind2sub(size(labelMap), curIdx);
        
        % Get neighbor indices
        neighbors = ...
            getValidNeighborIdxs(relativeNeighbors, x, y, labelMap);
        
        % Mask everything but the image that the current pixel belongs to
        maskedLabelMap = applySeparateImagesMask(...
            labelMap, separateImagesMask, y);
        
        % Get neighbor labels
        neighborLabels = maskedLabelMap(neighbors);
        
        % Count occurences of labels in neighborhood (ignoring unlabeled
        % and fill pixels)
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

function [labelCounts, maxLabel] = initLabelCounts(labelMap)
    % Create map for counting neighboring labels
    [maxX, maxY] = size(labelMap);
    maxLabel = max(labelMap(:));
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
