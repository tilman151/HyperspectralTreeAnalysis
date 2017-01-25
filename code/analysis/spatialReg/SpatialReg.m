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
    %    classifier ........... Instance of a Classifier (supervised or 
    %                           semi-supervised, single or ensemble) to be
    %                           used internally.
    %    r .................... Float value. Radius defining the 
    %                           neighborhood for label propagation. Needs 
    %                           to be lower than half of the image size in 
    %                           either direction.
    %    labelPropagation ..... Boolean. Enables/disables the label
    %                           propagation step.
    %    outputRegularization . Boolean. Enables/disables the output
    %                           regularization step.
    %    propagationThreshold . Float value. Relative threshold of
    %                           neighbors needed for propagating a label.
    %
    %% Methods:
    %    SpatialReg .... Constructor. Takes Name, Value pair arguments:
    %        Classifier ........... Set the internally used classifier.
    %        R .................... Set the radius property. 
    %                               Default: 5
    %        LabelPropagation ..... Enable/disable label propagation.
    %                               Default: true
    %        OutputRegularization . Enable/disable output regularization.
    %                               Default: true
    %        PropagationThreshold . Define the relative threshold of
    %                               neighbors needed for propagating a
    %                               label.
    %                               Default: 0.0
    %        VisualizeSteps ....... Enable/disable visualization of results
    %                               in between single processing steps.
    %                               Default: false
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
        r;
        
        % Enabled/Disabled processing steps
        labelPropagation;
        outputRegularization;
        
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
        function obj = SpatialReg(varargin)
            % Create input parser
            p = inputParser;
            p.addRequired('Classifier');
            p.addParameter('R', 5);
            p.addParameter('LabelPropagation', true);
            p.addParameter('OutputRegularization', true);
            p.addParameter('PropagationThreshold', 0);
            p.addParameter('VisualizeSteps', false);
            
            % Parse input arguments
            p.parse(varargin{:});
            
            % Save parameters
            obj.classifier = p.Results.Classifier;
            obj.r = p.Results.R;
            obj.labelPropagation = p.Results.LabelPropagation;
            obj.outputRegularization = p.Results.OutputRegularization;
            obj.propagationThreshold = p.Results.PropagationThreshold;
            obj.visualizeSteps = p.Results.VisualizeSteps;
            
            % Compute relative neighbor positions for this radius once
            obj.relativeNeighbors = getRelativeNeighbors(obj.r);
        end
        
        function str = toString(obj)
            % Create output string with class name and classifier
            str = ['SpatialReg (classifier: ' obj.classifier.toString()];
            
            % Append neighborhood radius
            str = [str ', r: ' num2str(obj.r)];
            
            % Append processing steps
            str = [str ', labelPropagation: ' ...
                   num2str(obj.labelPropagation)];
            str = [str ', outputRegularization: ' ...
                   num2str(obj.outputRegularization)];
            
            % Append propagation threshold
            str = [str ', propagationThreshold: '...
                   num2str(obj.propagationThreshold)];
            
            % Close parentheses
            str = [str ')'];
        end
        
        function str = toShortString(obj)
            % Create output string with classifier representation
            str = [obj.classifier.toShortString() '_'];
            
            % Append processing steps
            if obj.labelPropagation
                str = [str '_labelPropagation'];
            end
            if obj.outputRegularization
                str = [str '_outputRegularization'];
            end
            
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
            
            if obj.labelPropagation
                % Perform unsupervised clustering
                logger.info('SpatialReg', 'Clustering...');
                clusterIdxMap = ...
                    clustering(trainFeatureCube, trainLabelMap);
                
                % Display clustered map
                if obj.visualizeSteps
                    visualizeLabels(clusterIdxMap, 'Clusters');
                end
                
                % Propagate labels in spatial neighborhood for matching
                % clusters
                logger.info('SpatialReg', 'Propagating labels...');
                trainLabelMap = propagateLabels(...
                    trainLabelMap, clusterIdxMap, obj.relativeNeighbors,...
                    obj.propagationThreshold);
                
                % Display enriched label map
                if obj.visualizeSteps
                    visualizeLabels(trainLabelMap, ...
                        'Enriched Training Labels');
                end
            end
            
            % Train classifier on (enriched) data set
            logger.info('SpatialReg', 'Training classifier...');
            obj.classifier.trainOn(trainFeatureCube, trainLabelMap);
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap)
            
            % Get logger
            logger = Logger.getLogger();
            
            % Predict labels
            logger.info('SpatialReg', 'Predicting labels...');
            predictedLabelMap = ...
                obj.classifier.classifyOn(evalFeatureCube, maskMap);
            
            if obj.outputRegularization
                % Display result before regularization
                if obj.visualizeSteps
                    visualizeLabels(predictedLabelMap, ...
                        'Predicted labels before regularization');
                end
                
                % Regularize output labels based on spatial smoothness
                logger.info('SpatialReg', 'Regularizing output...');
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
