classdef RotationForest < ExampleClassifier
    %ROTATIONFOREST Summary of this class goes here
    %   Detailed explanation goes here
    
 
    properties
        %number of trees in the ensemble
        numTrees;
        
        %the tree ensemble 
        treeEnsemble;
        
        %additional specifications possible (eg. predicition of class
        %values as input
    end
    
    methods
       
        function obj = RandomForest(numTrees)
             %specify how many trees should be learned
            obj.numTrees = numTrees;
        end
        function obj = trainOn(obj,trainFeatures,trainLabels)
            [featureList, labelList, ~] = ...
                extractLabeledPixels(trainFeatures, trainLabels);
            
             %learn the ensemble with the given number of trees
            obj.treeEnsemble = TreeBagger(obj.numTrees,featureList,labelList);
        end
        
        function labels = classifyOn(obj,evalFeatures)
             % Transform input map to vector
            featureList = mapToVec(evalFeatures);
            
            %predict label using the ensemble
            labels = predict(obj.treeEnsemble, featureList);
        end
    end
    
end

