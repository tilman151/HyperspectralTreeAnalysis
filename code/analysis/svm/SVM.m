classdef SVM < Classifier
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
        % SVM template
        template;
        
        % Coding design
        coding;
        
        % Trained model
        model;
    end
    
    methods
        function obj = SVM(varargin)
            % Create input parser
            p = inputParser;
            p.addParameter('KernelFunction', 'linear');
            p.addParameter('Coding', 'onevsone');
            
            % Parse input arguments
            p.parse(varargin{:});
            
            % Create SVM template
            obj.template = templateSVM(...
                'KernelFunction', p.Results.KernelFunction);
            
            % Save coding for the following training
            obj.coding = p.Results.Coding;
        end
        
        function obj = trainOn(obj, trainFeatures, trainLabels)
            % Extract labeled pixels
            [featureList, labelList] = ...
                extractLabeledPixels(trainFeatures, trainLabels);
            
            % Train multiclass model
            obj.model = fitcecoc(featureList, labelList, ...
                'Coding', obj.coding, 'Learners', obj.template);
        end
        
        function labels = classifyOn(obj, evalFeatures)
            % Transform input map to vector
            featureList = mapToVec(evalFeatures);
            
            % Predict labels
            labels = obj.model.predict(featureList);
        end
    end
    
end

