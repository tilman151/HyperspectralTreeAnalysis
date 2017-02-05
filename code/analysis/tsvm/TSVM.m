classdef TSVM < Classifier
    %TSVM transductive Support Vector Machine
    %
    %    Support vector machine that uses unlabeled data as additional
    %    information. Also known as Semi-Supervised SVM (S3VM).
    %    This class requires a compiled version of the SVM-Light MATLAB
    %    interface, which can be downloaded from here: 
    %    https://sourceforge.net/projects/mex-svm/files/svm_mex601r14.zip/download
    %
    %% Properties:
    %    kernel .......... Name of the kernel function.
    %    polynomialOrder . Order of the polynomial kernel function.
    %    coding .......... Name of the multiclass coding design.
    %
    %% Methods:
    %    TSVM .......... Constructor. Can take Name, Value pair arguments 
    %                    that change the multiclass strategy and the 
    %                    internal parameters of the SVM. 
    %                    Possible arguments:
    %        KernelFunction .. Kernel function for the SVM.
    %                          'linear'(default) | 'polynomial'
    %        PolynomialOrder . Positive integer specifying the degree of
    %                          polynomial to be used for polynomial
    %                          kernel. This parameter is used only if
    %                          you set 'KernelFunction' to 'polynomial'.
    %                          Default: 3
    %        Coding .......... Coding design for the multiclass model.
    %                          'onevsone'(default) | 'onevsall'
    %    toString ...... See documentation in superclass Classifier.
    %    toShortString . See documentation in superclass Classifier.
    %    trainOn ....... See documentation in superclass Classifier.
    %    classifyOn .... See documentation in superclass Classifier.
    %
    % Version: 2017-02-05
    % Author: Cornelius Styp von Rekowski
    %
    
    properties
        % Parameters
        kernel;
        polynomialOrder
        coding;
    end
    
    properties(Hidden=true)
        % Trained binary models
        models;
    end
    
    methods
        function obj = TSVM(varargin)
            % Create input parser
            p = inputParser;
            p.addParameter('KernelFunction', 'linear');
            p.addParameter('PolynomialOrder', []);
            p.addParameter('Coding', 'onevsone');
            
            % Parse input arguments
            p.parse(varargin{:});
            
            % Save parameters
            obj.kernel = p.Results.KernelFunction;
            obj.coding = p.Results.Coding;
            
            if strcmp(obj.kernel, 'polynomial')
                if isempty(p.Results.PolynomialOrder)
                    obj.polynomialOrder = 3;
                else
                    obj.polynomialOrder = p.Results.PolynomialOrder;
                end
            end
        end
        
        function str = toString(obj)
            % Create output string with class name and kernel function
            str = ['t-SVM (KernelFunction: ' obj.kernel];
            
            % Append polynomial order if kernel is polynomial
            if strcmp(obj.kernel, 'polynomial')
                str = [str ', PolynomialOrder: ' ...
                       num2str(obj.polynomialOrder)];
            end
            
            % Append multiclass coding
            str = [str ', Coding: ' obj.coding];
            
            % Close parentheses
            str = [str ')'];
        end
        
        function str = toShortString(obj)
            % Create output string with class name and kernel function
            str = ['tSVM_' obj.kernel];
            
            % Append polynomial order if kernel is polynomial
            if strcmp(obj.kernel, 'polynomial')
                str = [str num2str(obj.polynomialOrder)];
            end
            
            % Append multiclass coding
            str = [str '_' obj.coding];
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % Get logger
            logger = Logger.getLogger();
            
            % Extract valid pixels as lists
            featureList = validListFromSpatial(...
                trainFeatureCube, trainLabelMap);
            labelList = validListFromSpatial(...
                trainLabelMap, trainLabelMap);
            
            % Build additional parameters
            params = '-v 0';
            switch obj.kernel
                case 'linear'
                    params = [params ' -t 0'];
                case 'polynomial'
                    params = [params ' -t 1 -d ' ...
                              num2str(obj.polynomialOrder)];
            end
            
            % Train model
            switch obj.coding
                case 'onevsone'
                    obj.models = ...
                        trainOneVsOne(featureList, labelList, params);
                otherwise
                    logger.error('t-SVM', ['Currently only one-vs-one '...
                        'coding is supported for t-SVMs!']);
                    exit;
            end
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap)
            
            % Extract unlabeled pixels as list
            featureList = validListFromSpatial(evalFeatureCube, maskMap);
            
            % Predict labels
            predictedLabelList = predictOneVsOne(featureList, obj.models);
            
            % Rebuild map representation
            predictedLabelMap = rebuildMap(predictedLabelList, maskMap);
        end
    end
    
end

function models = trainOneVsOne(featureList, labelList, params)
    % Get logger
    logger = Logger.getLogger();
    
    % Get neutral data
    neutralFeatureList = featureList(labelList == 0, :);
    neutralLabelList = zeros(size(neutralFeatureList, 1), 1);
    
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
        
        logger.trace('t-SVM 1vs1', ...
            ['Training ' num2str(c1) ' vs. ' num2str(c2)]);
        
        % Concatenate features and labels for the two classes
        binaryFeatureList = [...
            featureList(labelList == c1, :); ...
            featureList(labelList == c2, :); ...
            neutralFeatureList];
        binaryLabelList = [...
            ones(sum(labelList == c1), 1); ...
            -ones(sum(labelList == c2), 1); ...
            neutralLabelList];
        
        % Train and store model
        models(ii, :) = {c1, c2, ...
            svmlearn(binaryFeatureList, binaryLabelList, params)};
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
    % Create empty label list, because svmclassify needs one
    labelList = zeros(size(featureList, 1), 1);
    
    % Predict labels
    [~, predictedLabelList] = svmclassify(featureList, labelList, model);
    
    % Assign classes based on predictions
    predictedLabelList(predictedLabelList > 0) = c1;
    predictedLabelList(predictedLabelList <= 0) = c2;
end
