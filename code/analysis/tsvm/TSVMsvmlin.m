classdef TSVMsvmlin < Classifier
    %TSVMsvmlin transductive Support Vector Machine using SVMlin
    %
    %    Support vector machine that uses unlabeled data as additional
    %    information. Also known as Semi-Supervised SVM (S3VM).
    %    This class uses SVMlin as its underlying SVM implementation.
    %    In order to setup SVMlin, two steps are required:
    %    1. Compile the library: `make`
    %    2. Compile mex interface: 
    %            `mex -largeArrayDims svmlin_mex.cpp ssl.o`
    %
    %% Properties:
    %    coding .......... Name of the multiclass coding design.
    %    unlabeledRate ... Rate of the available unlabeled samples to be 
    %                      used for training.
    %
    %% Methods:
    %    TSVMsvmlin .... Constructor. Can take Name, Value pair arguments 
    %                    that change the multiclass strategy and the 
    %                    internal parameters of the SVM. 
    %                    Possible arguments:
    %        Coding .......... Coding design for the multiclass model.
    %                          'onevsall'(default) | 'onevsone'
    %        UnlabeledRate ... Rate of the available unlabeled samples to
    %                          be used for training.
    %                          Default: 1.0
    %    toString ...... See documentation in superclass Classifier.
    %    toShortString . See documentation in superclass Classifier.
    %    trainOn ....... See documentation in superclass Classifier.
    %    classifyOn .... See documentation in superclass Classifier.
    %
    % Version: 2017-02-24
    % Author: Cornelius Styp von Rekowski
    %
    
    properties
        % Parameters
        coding;
        unlabeledRate;
    end
    
    properties(Hidden=true)
        % Trained binary models
        models;
    end
    
    methods
        function obj = TSVMsvmlin(varargin)
            % Create input parser
            p = inputParser;
            p.addParameter('Coding', 'onevsall');
            p.addParameter('UnlabeledRate', 1.0);
            
            % Parse input arguments
            p.parse(varargin{:});
            
            % Save parameters
            obj.coding = p.Results.Coding;
            obj.unlabeledRate = p.Results.UnlabeledRate;
        end
        
        function str = toString(obj)
            % Create output string with class name
            str = 't-SVM [SVMlin] (';
            
            % Append multiclass coding
            str = [str ', Coding: ' obj.coding];
            
            % Append unlabeled rate
            str = [str ', UnlabeledRate: ' num2str(obj.unlabeledRate)];
            
            % Close parentheses
            str = [str ')'];
        end
        
        function str = toShortString(obj)
            % Create output string with class name
            str = 'tSVM_';
            
            % Append multiclass coding
            str = [str '_' obj.coding];
            
            % Append unlabeled rate
            str = [str '_ur' num2str(obj.unlabeledRate)];
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % Get logger
            logger = Logger.getLogger();
            
            % Extract valid pixels as lists
            featureList = validListFromSpatial(...
                trainFeatureCube, trainLabelMap);
            labelList = validListFromSpatial(...
                trainLabelMap, trainLabelMap);
            
            % Sample rate of unlabeled instances
            if obj.unlabeledRate < 1.0
                [featureList, labelList] = sampleUnlabeledRate(...
                    featureList, labelList, obj.unlabeledRate);
            end
            
            % Train model
            switch obj.coding
                case 'onevsone'
                    obj.models = ...
                        train1vs1Svmlin(featureList, labelList, true);
                case 'onevsall'
                    obj.models = ...
                        train1vsAllSvmlin(featureList, labelList, true);
                otherwise
                    logger.error('t-SVM', 'Invalid multiclass coding!');
                    exit;
            end
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap, ~)
            
            % Extract unlabeled pixels as list
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


function [featureList, labelList] = ...
    sampleUnlabeledRate(featureList, labelList, unlabeledRate)
    
    % Get logger
    logger = Logger.getLogger();
    
    % Split lists into labeled and unlabeled instances
    unlabeledFeatureList = featureList(labelList == 0, :);
    labeledFeatureList = featureList(labelList > 0, :);
    labeledLabelList = labelList(labelList > 0);

    % Calculate desired number of unlabeled instances
    numUnlabeled = size(unlabeledFeatureList, 1);
    numRateUnlabeled = floor(unlabeledRate * numUnlabeled);

    logger.debug('t-SVM', ...
        ['Sampling ' num2str(unlabeledRate) ...
         ' of the available unlabeled instances: '...
         num2str(numRateUnlabeled) '/' num2str(numUnlabeled)]);

    % Create subsampled feature and label lists
    featureList = [labeledFeatureList; ...
        unlabeledFeatureList(...
            randsample(numUnlabeled, numRateUnlabeled), :)];
    labelList = [labeledLabelList; zeros(numRateUnlabeled, 1)];
end
