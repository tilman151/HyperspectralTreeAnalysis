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
                    obj.models = trainOneVsOne(featureList, labelList);
                otherwise
                    logger.error('SVM', ['Currently only one-vs-one '...
                        'coding is supported in SVMlin!']);
                    exit;
            end
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap, ~)
            
            % Extract list of unlabeled pixels
            featureList = validListFromSpatial(evalFeatureCube, maskMap);
            
            % Predict labels
            predictedLabelList = predictOneVsOne(featureList, obj.models);
            
            % Rebuild map representation
            predictedLabelMap = rebuildMap(predictedLabelList, maskMap);
        end
    end
    
end


function models = trainOneVsOne(featureList, labelList)
    % Get logger
    logger = Logger.getLogger();
    
    % Get list of classes
    classes = unique(labelList(labelList > 0));
    numClasses = length(classes);
    
    % Create class combinations and cell array for binary classifiers
    classpairs = nchoosek(1:numClasses, 2);
    numBinaryClassifiers = size(classpairs, 1);
    models = cell(numBinaryClassifiers, 3);

    % Train binary classifiers
    for ii = 1 : numBinaryClassifiers
        % Get classes for this classifier
        classpair = classpairs(ii, :);
        c1 = classes(classpair(1));
        c2 = classes(classpair(2));
        
        logger.trace('SVM 1vs1', ...
            ['Training ' num2str(c1) ' vs. ' num2str(c2)]);
        
        % Concatenate features and labels for the two classes
        binaryFeatureList = [...
            featureList(labelList == c1, :); ...
            featureList(labelList == c2, :)];
        binaryLabelList = [...
            ones(sum(labelList == c1), 1); ...
            -ones(sum(labelList == c2), 1)];
        
        % Make feature list sparse
        binaryFeatureList = sparse(binaryFeatureList);
        
        % Train and store model (-A 1 sets algorithm to SVM)
        modelWeights = svmlin('-A 1', binaryFeatureList, binaryLabelList);
        models(ii, :) = {c1, c2, modelWeights};
    end
end

function predictedLabelList = predictOneVsOne(featureList, models)
    % Obtain vote from each model
    votes = cellfun(@(c1, c2, m) applyModel(c1, c2, m, featureList), ...
        models(:, 1), models(:, 2), models(:, 3), 'UniformOutput', false);
    
    % Reshape votes to numSamples x numModels
    votes = cell2mat(votes);
    votes = reshape(votes, [size(featureList, 1), size(models, 1)]);
    
    % Decide for class with maximum number of votes
    maxClass = max(votes(:));
    voteCounts = histc(votes, 1:maxClass, 2);
    [~, predictedLabelList] = max(voteCounts, [], 2);
end

function predictedLabelList = applyModel(c1, c2, model, featureList)
    % Make feature list sparse
    featureList = sparse(featureList);
    
    % Predict labels
    [~, predictedLabelList] = svmlin([], featureList, [], model);
    
    % Assign classes based on predictions
    predictedLabelList(predictedLabelList > 0) = c1;
    predictedLabelList(predictedLabelList <= 0) = c2;
end
