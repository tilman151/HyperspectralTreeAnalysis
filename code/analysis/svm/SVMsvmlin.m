classdef SVMsvmlin < Classifier
    %SVMsvmlin Linear Support Vector Machine for multiclass problems
    %
    %    This class uses SVMlin as its underlying SVM implementation.
    %    In order to setup SVMlin, two steps are required:
    %    1. Compile the library: `make`
    %    2. Compile mex interface: 
    %            `mex -largeArrayDims svmlin_mex.cpp ssl.o`
    %
    %% Properties:
    %    coding ........ Name of the multiclass coding design.
    %
    %% Methods:
    %    SVMlibsvm ..... Constructor. Can take Name, Value pair arguments 
    %                    that change the multiclass strategy and the 
    %                    internal parameters of the SVM. 
    %                    Possible arguments:
    %        Coding .......... Coding design for the multiclass model.
    %                          'onevsone'(default) | 'onevsall'
    %    toString ...... See documentation in superclass Classifier.
    %    toShortString . See documentation in superclass Classifier.
    %    trainOn ....... See documentation in superclass Classifier.
    %    classifyOn .... See documentation in superclass Classifier.
    %
    % Version: 2017-02-23
    % Author: Cornelius Styp von Rekowski
    %
    
    properties
        % Parameters
        coding;
    end
    
    properties(Hidden=true)
        % Trained models
        models;
    end
    
    methods
        function obj = SVMsvmlin(varargin)
            % Create input parser
            p = inputParser;
            p.addParameter('Coding', 'onevsone');
            
            % Parse input arguments
            p.parse(varargin{:});
            
            % Save parameters
            obj.coding = p.Results.Coding;
        end
        
        function str = toString(obj)
            % Create output string with class name and multiclass coding
            str = ['SVM [SVMlin] (Coding: ' obj.coding ')'];
        end
        
        function str = toShortString(obj)
            % Create output string with class name and multiclass coding
            str = ['SVMlin_' obj.coding];
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % Get logger
            logger = Logger.getLogger();
            
            % Extract labeled pixels
            featureList = validListFromSpatial(...
                trainFeatureCube, trainLabelMap, true);
            labelList = validListFromSpatial(...
                trainLabelMap, trainLabelMap, true);
            
            % Train model
            switch obj.coding
                case 'onevsone'
                    obj.models = train1vs1Svmlin(featureList, labelList);
                case 'onevsall'
                    obj.models = train1vsAllSvmlin(featureList, labelList);
                otherwise
                    logger.error('SVM', 'Invalid multiclass coding!');
                    exit;
            end
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap, ~)
            
            % Extract list of unlabeled pixels
            featureList = validListFromSpatial(evalFeatureCube, maskMap);
            
            % Predict labels
            switch obj.coding
                case 'onevsone'
                    predictedLabelList = ...
                        predict1vs1Svmlin(featureList, obj.models);
                case 'onevsall'
                    predictedLabelList = ...
                        predict1vsAllSvmlin(featureList, obj.models);
                otherwise
                    logger.error('SVM', 'Invalid multiclass coding!');
                    exit;
            end
            
            % Rebuild map representation
            predictedLabelMap = rebuildMap(predictedLabelList, maskMap);
        end
    end
    
end

