classdef BestClassifierEnsemble < Classifier
    %BASICENSEMBLE Classifier that uses other classifiers to predict labels
    %
    %% Properties:
    %    baseClassifiers ............... Classifiers which are used to
    %                                    predict the label
    %    trainingInstanceProportions ... The proportion, which is
    %                                    selected from the given training
    %                                    instances for a specific 
    %                                    classifier
    %
    %% Methods:
    %    BasicEnsemble ...... Constructor.
    %       Input:
    %           baseClassifiers ............... a Nx1 cell array with
    %                                           instantiated classifiers
    %           numClassifiers ................  a Nx1 array with the
    %                                           number of copies for each
    %                                           base classifier
    %           trainingInstanceProportions ...  a Nx1 array with a
    %                                           number between 0 and 1. For
    %                                           example a 0.8 says that all
    %                                           classifiers are trained with
    %                                           80% of the training 
    %                                           instances
    %           votingMode .................... VotingMode enum to specify
    %                                           the voting method to get to 
    %                                           a conclusion
    %           validationSubsampleProportion . VotingMode enum to specify
    %                                           the voting method to get to 
    %                                           a conclusion
    %           remainClassDistribution ....... set true if the class
    %                                           distribution for the 
    %                                           training subsamples should 
    %                                           be approximately the same
    %                                           as the original 
    %                                           distribution
    %           multithreadingTraining ........ set to true if training
    %                                           should use the
    %                                           multithreaded 
    %                                           implementation
    %           multithreadingClassification .. set to true if
    %                                           classification should use 
    %                                           the multithreaded 
    %                                           implementation
    %           
    %           
    %    toString ...... See documentation in superclass Classifier.
    %    toShortString . See documentation in superclass Classifier.
    %    trainOn ....... See documentation in superclass Classifier.
    %    classifyOn .... See documentation in superclass Classifier.
    %
    % Version: 2017-14-02
    % Author: Tuan Pham Minh
    %
    
    properties
        samplesetPath;
        multithreadingClassification;
        featureExtractors;
        baseClassifiers;
        numClassifiersStrings;
        baseClassifiersStrings;
        baseClassifiersShortStrings;
        votingMode;
        weights;
    end
    
    methods
        
        function obj = BestClassifierEnsemble(baseClassifierPaths, ...
                votingMode, multithreadingClassification, samplesetPath)
            'begin'
            
            obj.samplesetPath = samplesetPath;
            
            'begin load'
            [baseClassifiers, featureExtractors, accuracies] = ...
                obj.loadClassifier(baseClassifierPaths); 
            'loading finished'
            obj.featureExtractors = featureExtractors;
            
            obj.votingMode = votingMode;
            obj.multithreadingClassification = ...
                multithreadingClassification;
            
            obj.baseClassifiersStrings = cellfun(@(o)o.toString(), ...
                                                 baseClassifiers, ...
                                                 'uniformoutput', 0);
            obj.baseClassifiersShortStrings = ...
                cellfun(@(o)o.toShortString(), ...
                        baseClassifiers, ...
                        'uniformoutput', 0);
            
            % copy the classifier
            obj.baseClassifiers = baseClassifiers;
            
            if obj.votingMode ~= VotingMode.Majority
                obj.weights = ones(size(accuracies));
            elseif obj.votingMode == VotingMode.Presidential
                [~, bestClassifier] = max(accuracies);
                obj.weights = ones(size(accuracies));
                obj.weights(bestClassifier) = numel(accuracies) - 0.5;
            else
                obj.weights = normalizeRating( accuracies, [0.1,0.9] );
            end
            'end'
        end
        
        
        function str = toString(obj)
            paramStr = ['voting: ' ...
                         char(obj.votingMode)];
            str = ['BasicEnsemble ', paramStr, '[' ];
            stringParts = ...
                cellfun(@(c) ['( classifier:[' c '] )'], ...
                        obj.baseClassifiersStrings, ...
                        'uniformoutput', 0);
            stringParts{end}(end-2:end) = [];
            str = [str horzcat(stringParts{:}) ']'];
        end
        
        function str = toShortString(obj)
            
            paramStr = char(obj.votingMode);
            
            str = ['BasicEnsemble--' paramStr];
            stringParts = ...
                cellfun(@(c) [c '--'], ...
                        obj.baseClassifiersShortStrings, ...
                        'uniformoutput', 0);
            stringParts{end}(end-1:end) = [];
            str = [str horzcat(stringParts{:})];
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap)
            
            % let each classifier predict
            concatenatedLabels = ...
                obj.classifyOnEachClassifier(evalFeatureCube, maskMap);
            [x,y,nClassifier] = size(concatenatedLabels);
            concatenatedLabels = ...
                reshape(concatenatedLabels, x*y, nClassifier);
            
            predictedLabelMap = zeros(x*y, 1);
            
            weights = obj.weights;
            
            parfor i = 1:numel(maskMap)
                w = weights;
                predictions = concatenatedLabels(i,:);
                labels = unique(predictions);
                labelWeights = zeros(size(labels));
                for labelIdx = 1:numel(labels)
                    l = labels(labelIdx);
                    labelWeights(labelIdx) = sum(w(predictions == l));
                end
                [~,labelIndex] = max(labelWeights);
                predictedLabelMap(i) = labels(labelIndex);
            end
            predictedLabelMap = reshape(predictedLabelMap, x, y);
        end
        
        function concatenatedLabels = ...
                    classifyOnEachClassifier(obj, evalFeatureCube, maskMap)
                    
            if obj.multithreadingClassification
                classifier = obj.baseClassifiers;
                accumulatedLabels = cell(length(classifier),1);
                parfor i = 1:length(classifier)
                    newFeatureCube = applyFeatureExtraction(...
                        evalFeatureCube, ...
                        maskMap, ...
                        extractors, ...
                        obj.samplesetPath);
                    accumulatedLabels{i} = ...
                        classifier{i}.classifyOn(...
                            newFeatureCube, maskMap);
                end
            else
                classifier = obj.baseClassifiers;
                accumulatedLabels = cell(length(classifier),1);
                for i = 1:length(classifier)
                    newFeatureCube = applyFeatureExtraction(...
                        evalFeatureCube, ...
                        maskMap, ...
                        obj.featureExtractors{i}, ...
                        obj.samplesetPath);
                    accumulatedLabels{i} = ...
                        classifier{i}.classifyOn(...
                            newFeatureCube, maskMap);
                end
            end
            
            % concatenate the labels along the third dimension
            concatenatedLabels = cat(3, accumulatedLabels{:});
            
        end
        
        function [bC, fE, w] = loadClassifier(obj, baseClassifierPaths)
            for i = 1:numel(baseClassifierPaths)
               p = baseClassifierPaths{i};
               [d,f,e] = fileparts(p);
               
               bC = cell(size(baseClassifierPaths));
               fE = cell(size(baseClassifierPaths));
               w = cell(size(baseClassifierPaths));
               
               if(isempty(f))
                   searchStr = [d '/model_*'];
                   modelFilePaths = dir(searchStr);
                   modelFilePaths = {modelFilePaths.name};
                   accuracies = zeros(size(modelFilePaths));
                   for j = 1:numel(modelFilePaths)
                      modelFile = load([d '/' modelFilePaths{j}]);
                      cMat = modelFile.confMat;
                      accuracies(j) =  ...
                          Evaluator.getAccuracy(cMat(2:end, 2:end));
                   end
                   [~, bestClassifierIndex] = max(accuracies);
                   f = modelFilePaths{bestClassifierIndex};
               else
                   f = [f e];
                   e = '';
               end
               
               modelFile = load([d '/' f]);
               classifier = modelFile.model;
               cMat = modelFile.confMat;
               accuracy = Evaluator.getAccuracy(cMat(2:end, 2:end));
               
               featureExtractorFile = load([d '/FeatureExtractors.mat']);
               featureExtractors = featureExtractorFile.EXTRACTORS;
               
               bC{i} = classifier;
               fE{i} = featureExtractors;
               w{i} = accuracy;
            end
        end
    end
    
end


        
function rawFeatures = applyFeatureExtraction(rawFeatures, maskMap, ...
                                extractors, samplesetPath)
    for i = 1:numel(extractors)
        rawFeatures = extractors{i}.extractFeatures(... 
                            rawFeatures, ...
                            maskMap, samplesetPath);
    end 
end
