classdef RandomForest < Classifier
    %RANDOM FOREST Standard implementation of a random forest
    %    
    %    Using https://de.mathworks.com/help/stats/treebagger.html. The
    %    internally used object is a TreeBagger. It is possible to extract 
    %    other useful information from that object, see 
    %    https://de.mathworks.com/help/stats/compacttreebagger-class.html.
    %
    %% Properties:
    %    numTrees ..... The number of trees used in the ensemble
    %    treeEnsemble . The ensemble itself, which is a TreeBagger object
    %
    %% Methods:
    %    RandomForest .. Initialise forest with given number of trees
    %    toString ...... See documentation in superclass Classifier.
    %    toShortString . See documentation in superclass Classifier.
    %    trainOn ....... See documentation in superclass Classifier. 
    %    classifyOn .... See documentation in superclass Classifier.
    %
    % Version: 2017-01-13
    % Author: Viola Hauffe
    %
 
    properties
        % Number of trees in the ensemble
        numTrees;
        
        % The tree ensemble 
        treeEnsemble;
        
        % Additional specifications possible (eg. predicition of class
        % values as input)
    end
    
    methods
       
        function obj = RandomForest(numTrees)
            % Specify how many trees should be learned
            obj.numTrees = numTrees;
        end
        
        function str = toString(obj)
           str = ['RandomForest (numTrees: ' int2str(obj.numTrees) ')'];  
        end
        
        function str = toShortString(obj)
            str = ['RandomForest_' int2str(obj.numTrees)];
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % Extract labeled pixels
            featureList = validListFromSpatial(...
                trainFeatureCube, trainLabelMap, true);
            labelList = validListFromSpatial(...
                trainLabelMap, trainLabelMap, true);
            
            % Train the ensemble with the given number of trees.
            obj.treeEnsemble = ...
                TreeBagger(obj.numTrees, featureList, labelList);
        end
        
        function predictedLabelMap = classifyOn(...
                obj, evalFeatureCube, maskMap, ~)
            
            % Extract list of unlabeled pixels
            featureList = validListFromSpatial(evalFeatureCube, maskMap);
            
            % Predict labels using the ensemble
            predictedLabelList = obj.treeEnsemble.predict(featureList);
            
            % TreeBagger output is a cell array -> transform to matrix
            predictedLabelList = cellfun(@(x) str2num(x), predictedLabelList);
            
            % Rebuild map representation
            predictedLabelMap = rebuildMap(predictedLabelList, maskMap);
        end
    end
    
end
