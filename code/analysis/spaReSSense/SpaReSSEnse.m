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
    %                         for label propagation.
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
        
        % Neighborhood radius
        r = 5;
        
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
                    trainLabels, clusterIdxMap, obj.r);
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
                regularizedLabelMap = regularize(labelMap);
                labels = mapToVec(regularizedLabelMap);
            end
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

function enrichedLabels = propagateLabels(labelMap, clusterIdxMap, r)
    % Get map dimensions
    mapSize = size(labelMap);
    maxX = mapSize(1);
    maxY = mapSize(2);
    
    % Create map for counting neighboring labels
    maxLabel = max(max(labelMap));
    labelCounts = zeros(maxX, maxY, maxLabel);
    
    % Propagate labels from all labeled pixels
    for labeledIdx = find(labelMap > 0)'
        [idxX, idxY] = ind2sub(mapSize, labeledIdx);
        
        % Get assigned cluster and label
        cLabeled = clusterIdxMap(labeledIdx);
        label = labelMap(labeledIdx);
        
        % Go through all neighbors
        for addX = max(1-idxX, -r) : min(maxX-idxX, r)
            % Use euclidean distance (based on sqrt(x^2 + y^2) = r)
            euclY = fix(sqrt(r^2 - addX^2));
            for addY = max(1-idxY, -euclY) : min(maxY-idxY, euclY)
                propX = idxX + addX;
                propY = idxY + addY;
                
                % Get assigned cluster for the propagating pixel
                cProp = clusterIdxMap(propX, propY);
                
                % Check if pixel is unlabeled and clusters match
                if labelMap(propX, propY) == 0 && cLabeled == cProp
                    % Increase count for this label at the pixel position
                    labelCounts(propX, propY, label) = ...
                        labelCounts(propX, propY, label) + 1;
                end
            end
        end
    end
    
    % For each unlabeled pixel, assign the label with the highest count
    % received from the neighborhood
    enrichedLabels = labelMap;
    for unlabeledIdx = find(labelMap == 0)'
        [idxX, idxY] = ind2sub(mapSize, unlabeledIdx);
        [count, label] = max(labelCounts(idxX, idxY, :));
        if count > 0
            enrichedLabels(unlabeledIdx) = label;
        end
    end
end

function regularizedLabelMap = regularize(labelMap)
    % TODO: Reorder labels to spatial representation
    % TODO: Regularize in spatial domain
end
