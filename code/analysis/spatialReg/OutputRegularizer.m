classdef OutputRegularizer
    %OUTPUTREGULARIZER Summary of this class goes here
    %   
    %    Based on the scientific work presented in "Spatially regularized 
    %    semisupervised Ensembles of Extreme Learning Machines for 
    %    hyperspectral image segmentation." [Ayerdi, Marqués, Graña 2015].
    %    The algorithm works as follows:
    %    1. A classifier predicts labels for pixels of an input image.
    %    2. The output image is regularized by assuming spatial smoothness.
    %       Every pixel receives the label that is most prominent in its
    %       neighborhood.
    %    
    %    Any previously trained classifier inheriting the common interface
    %    can be used internally.
    %
    %% Properties:
    %    classifier ........... Instance of a Classifier (supervised or 
    %                           semi-supervised, single or ensemble) to be
    %                           used internally.
    %    r .................... Float value. Radius defining the 
    %                           neighborhood for regularization. Needs 
    %                           to be lower than half of the image size in 
    %                           either direction.
    %
    %% Methods:
    %    OutputRegularizer . Constructor. Takes Name, Value pair arguments:
    %        ModelFile .......... Set the internally used classifier by
    %                             providing the path to a previously 
    %                             trained model.
    %        R .................. Set the radius property. 
    %                             Default: 5
    %        VisualizeSteps ..... Enable/disable visualization of results
    %                             in between single processing steps.
    %                             Default: false
    %    toString .......... See documentation in superclass Classifier.
    %    toShortString ..... See documentation in superclass Classifier.
    %    trainOn ........... See documentation in superclass Classifier.
    %    classifyOn ........ See documentation in superclass Classifier.
    %
    % Version: 2017-02-07
    % Author: Cornelius Styp von Rekowski
    %
    
    properties
        % The internally used pretrained classifier
        classifier;
        
        % Neighborhood radius
        r;
    end
    
    properties(Hidden=true)
        % Cached neighborhood indices
        relativeNeighbors;
        
        % Visualize results in between single processing steps
        visualizeSteps;
    end
    
    methods
        function obj = OutputRegularizer(varargin)
            % Create input parser
            p = inputParser;
            p.addRequired('ModelFile');
            p.addParameter('R', 5);
            p.addParameter('VisualizeSteps', false);
            
            % Parse input arguments
            p.parse(varargin{:});
            
            % Load model
            obj.classifier = Classifier.loadFrom(p.Results.ModelFile);
            
            % Save other parameters
            obj.r = p.Results.R;
            obj.visualizeSteps = p.Results.VisualizeSteps;
            
            % Compute relative neighbor positions for this radius once
            obj.relativeNeighbors = getRelativeNeighbors(obj.r);
        end
        
        function str = toString(obj)
            % Create output string with class name
            str = 'OutputRegularizer (';
            
            % Append classifier
            str = [str 'classifier: ' obj.classifier.toString()];
            
            % Append neighborhood radius
            str = [str ', r: ' num2str(obj.r)];
            
            % Close parentheses
            str = [str ')'];
        end
        
        function str = toShortString(obj)
            % Create output string with classifier representation
            str = [obj.classifier.toShortString() '_'];
            
            % Append regularized output suffix
            str = [str '_regularizedOutput'];
            
            % Append neighborhood radius
            str = [str '_r' num2str(obj.r)];
        end
        
        function obj = trainOn(obj, ~, ~)
            % Model has already been trained -> nothing to do
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap)
            
            % Get logger
            logger = Logger.getLogger();
            
            % Predict labels
            logger.info('OutputRegularizer', 'Predicting labels...');
            predictedLabelMap = ...
                obj.classifier.classifyOn(evalFeatureCube, maskMap);
            
            % Display result before regularization
            if obj.visualizeSteps
                visualizeLabels(predictedLabelMap, ...
                    'Predicted labels before regularization');
            end

            % Regularize output labels based on spatial smoothness
            logger.info('OutputRegularizer', 'Regularizing output...');
            predictedLabelMap = ...
                regularize(predictedLabelMap, obj.relativeNeighbors);
        end
    end
    
end
