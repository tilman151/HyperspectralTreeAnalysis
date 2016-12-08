classdef RandomForest < Classifier
    %RANDOM FOREST using https://de.mathworks.com/help/stats/treebagger.html
    %
    %%Properties:
    %
    %numTrees - the number of trees used in the ensemble
    %treeEnsemble - the ensemble itself, which is a TreeBagger object
    %It's possible to extract other useful information from that object,
    %see https://de.mathworks.com/help/stats/compacttreebagger-class.html
    %
    %%Methods:
    %
    %RandomForest - initialise forest with given number of trees
    %
    %trainOn - See documentation in superclass Classifier. 
    %classifyOn - See documentation in superclass Classifier.
 
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
