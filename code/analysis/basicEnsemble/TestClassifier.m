classdef TestClassifier < Classifier
    %SVM Standard Support Vector Machine for multiclass problems
    %
    %    See detailed explanations on the used algorithms and possible
    %    arguments on the following two documentation pages:
    %    - https://de.mathworks.com/help/stats/fitcecoc.html
    %    - https://de.mathworks.com/help/stats/templatesvm.html
    %
    %% Properties:
    %    template ... SVM template for classification.
    %    coding ..... Coding design for the multiclass model.
    %    model ...... Trained model.
    %
    %% Methods:
    %    SVM ........ Constructor. Can take Name, Value pair arguments that
    %                 change the multiclass strategy and the internal
    %                 parameters of the SVM. Possible arguments:
    %        KernelFunction . Kernel function for the SVM.
    %                         'linear'(default) | 'gaussian' | 'rbf' | 
    %                         'polynomial'
    %        Coding ......... Coding design for the ECOC (error-correcting 
    %                         output codes) multiclass model.
    %                         'onevsone'(default) | 'allpairs' |
    %                         'binarycomplete' | 'denserandom' | 'onevsall'
    %                         | 'ordinal' | 'sparserandom' | 
    %                         'ternarycomplete'.
    %    trainOn .... See documentation in superclass Classifier.
    %    classifyOn . See documentation in superclass Classifier.
    %
    % Version: 2016-11-24
    % Author: Cornelius Styp von Rekowski
    %
    
    properties
        baseClassifier;
        trainingInstanceProportions;
    end
    
    methods
        function obj = TestClassifier(baseClassifier, numClassifier, trainingInstanceProportions)
            
            % generate indices to copy the baseclassifier 
            % calculate the points, where new classifier starts
            classifierIndexStartPoint = [1 (cumsum(numClassifier(1:end-1)) + 1)];
            % calculate the indices, which is then used to copy the objects
            classifierIndices = zeros(1, sum(numClassifier));
            classifierIndices(classifierIndexStartPoint) = 1;
            classifierIndices = cumsum(classifierIndices);
            % copy the classifier
            obj.baseClassifier = baseClassifier(classifierIndices);
            obj.trainingInstanceProportions = num2cell(trainingInstanceProportions(classifierIndices));
        end
        
        function obj = trainOn(obj, trainFeatures, trainLabels)
            obj.baseClassifier{1} = trainFeatures;
        end
        
        function labels = classifyOn(obj, evalFeatures)
        end
    end
    
end

