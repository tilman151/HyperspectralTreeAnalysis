classdef LabelPropagator < Classifier
    %LABELPROPAGATOR Propagate labels before training a classifier
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
    %
    %    Any classifier inheriting the common interface can be used 
    %    internally.
    %
    %% Properties:
    %    classifier ........... Instance of a Classifier (supervised or 
    %                           semi-supervised, single or ensemble) to be
    %                           used internally.
    %    r .................... Float value. Radius defining the 
    %                           neighborhood for label propagation. Needs 
    %                           to be lower than half of the image size in 
    %                           either direction.
    %    propagationThreshold . Float value. Relative threshold of
    %                           neighbors needed for propagating a label.
    %
    %% Methods:
    %    LabelPropagator . Constructor. Takes Name, Value pair arguments:
    %        Classifier ........... Set the internally used classifier.
    %                               This should be an instance of a class
    %                               inheriting from Classifier.
    %        R .................... Set the radius property. 
    %                               Default: 5
    %        PropagationThreshold . Define the relative threshold of
    %                               neighbors needed for propagating a
    %                               label.
    %                               Default: 0.0
    %        VisualizeSteps ....... Enable/disable visualization of results
    %                               in between single processing steps.
    %                               Default: false
    %    toString ........ See documentation in superclass Classifier.
    %    toShortString ... See documentation in superclass Classifier.
    %    trainOn ......... See documentation in superclass Classifier.
    %    classifyOn ...... See documentation in superclass Classifier.
    %
    % Version: 2017-02-07
    % Author: Cornelius Styp von Rekowski
    %
    
    properties
        % The internally used classifier
        classifier;
        
        % Neighborhood radius
        r;
        
        % Relative propagation threshold
        propagationThreshold;
    end
    
    properties(Hidden=true)
        % Cached neighborhood indices
        relativeNeighbors;
        
        % Visualize results in between single processing steps
        visualizeSteps;
    end
    
    methods
        function obj = LabelPropagator(varargin)
            % Create input parser
            p = inputParser;
            p.addRequired('Classifier');
            p.addParameter('R', 5);
            p.addParameter('PropagationThreshold', 0);
            p.addParameter('VisualizeSteps', false);
            
            % Parse input arguments
            p.parse(varargin{:});
            
            % Save parameters
            obj.classifier = p.Results.Classifier;
            obj.r = p.Results.R;
            obj.propagationThreshold = p.Results.PropagationThreshold;
            obj.visualizeSteps = p.Results.VisualizeSteps;
            
            % Compute relative neighbor positions for this radius once
            obj.relativeNeighbors = getRelativeNeighbors(obj.r);
        end
        
        function str = toString(obj)
            % Create output string with class name
            str = 'LabelPropagator (';
            
            % Append classifier
            str = [str 'classifier: ' obj.classifier.toString()];
            
            % Append neighborhood radius
            str = [str ', r: ' num2str(obj.r)];
            
            % Append propagation threshold
            str = [str ', propagationThreshold: '...
                   num2str(obj.propagationThreshold)];
            
            % Close parentheses
            str = [str ')'];
        end
        
        function str = toShortString(obj)
            % Create output string with classifier representation
            str = [obj.classifier.toShortString() '_'];
            
            % Append propagated labels suffix
            str = [str '_propagatedLabels'];
            
            % Append neighborhood radius
            str = [str '_r' num2str(obj.r)];
            
            % Append propagation threshold
            if obj.propagationThreshold > 0.0
                str = [str '_pT' num2str(obj.propagationThreshold)];
            end
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % Get logger
            logger = Logger.getLogger();
            
            % Perform unsupervised clustering
            logger.info('LabelPropagator', 'Clustering...');
            clusterIdxMap = ...
                clustering(trainFeatureCube, trainLabelMap);

            % Display clustered map
            if obj.visualizeSteps
                visualizeLabels(clusterIdxMap, 'Clusters');
            end

            % Propagate labels in spatial neighborhood for matching
            % clusters
            logger.info('LabelPropagator', 'Propagating labels...');
            trainLabelMap = propagateLabels(...
                trainLabelMap, clusterIdxMap, obj.relativeNeighbors,...
                obj.propagationThreshold);

            % Display enriched label map
            if obj.visualizeSteps
                visualizeLabels(trainLabelMap, ...
                    'Enriched Training Labels');
            end
            
            % Train classifier on (enriched) data set
            logger.info('LabelPropagator', 'Training classifier...');
            obj.classifier.trainOn(trainFeatureCube, trainLabelMap);
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap)
            
            % Predict labels using the internal classifier
            predictedLabelMap = ...
                obj.classifier.classifyOn(evalFeatureCube, maskMap);
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
