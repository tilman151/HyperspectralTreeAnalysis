classdef TemplateEnsemble < Classifier
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
        model;
        templates;
    end
    
    methods
        function obj = BasicEnsemble(templates)
            obj.templates = templates;
        end
        
        function obj = trainOn(obj, trainFeatures, trainLabels)
            % Extract labeled pixels
            [featureList, labelList, ~] = ...
                extractLabeledPixels(trainFeatures, trainLabels);
            
            % Train multiclass model
            obj.model = fitcensemble(featureList, labelList, ...
                'Learners', obj.templates);
        end
        
        function labels = classifyOn(obj, evalFeatures)
            % Transform input map to vector
            featureList = mapToVec(evalFeatures);
            
            % Predict labels
            labels = obj.model.predict(featureList);
        end
    end
    
end

