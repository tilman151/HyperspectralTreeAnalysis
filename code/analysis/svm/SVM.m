classdef SVM < Classifier
    %SVM Standard Support Vector Machine for multiclass problems
    %
    %    See detailed explanations on the used algorithms and possible
    %    arguments here: https://de.mathworks.com/help/stats/fitcecoc.html
    %
    %% Properties:
    %    template ... 
    %    coding ..... 
    %    model ...... 
    %
    %% Methods:
    %    SVM ........ 
    %    trainOn .... 
    %    classifyOn . 
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
        function obj = SVM(kernel, coding)
            % TODO: allow Name Value pairs instead -> use inputParser
            if nargin < 2
                kernel = 'linear';
                coding = 'onevsone';
            end
            
            % Create SVM template
            obj.template = templateSVM('KernelFunction', kernel);
            
            obj.coding = coding;
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

