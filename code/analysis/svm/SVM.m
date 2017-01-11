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
    %        KernelFunction .. Kernel function for the SVM.
    %                          'linear'(default) | 'gaussian' | 'rbf' | 
    %                          'polynomial'
    %        PolynomialOrder . Positive integer specifying the degree of
    %                          polynomial to be used for polynomial
    %                          kernel. This parameter is used only if
    %                          you set 'KernelFunction' to 'polynomial'.
    %                          Default: 3
    %        Coding .......... Coding design for the ECOC (error-correcting 
    %                          output codes) multiclass model.
    %                          'onevsone'(default) | 'allpairs' |
    %                          'binarycomplete' | 'denserandom' | 
    %                          'onevsall' | 'ordinal' | 'sparserandom' | 
    %                          'ternarycomplete'.
    %    toString ... See documentation in superclass Classifier.
    %    trainOn .... See documentation in superclass Classifier.
    %    classifyOn . See documentation in superclass Classifier.
    %
    % Version: 2016-12-22
    % Author: Cornelius Styp von Rekowski
    %
    
    properties
        % SVM template
        template;
        
        % Coding design
        coding;
    end
    
    properties(Hidden=true)
        % Trained model
        model;
    end
    
    methods
        function obj = SVM(varargin)
            % Create input parser
            p = inputParser;
            p.addParameter('KernelFunction', 'linear');
            p.addParameter('PolynomialOrder', 3);
            p.addParameter('Coding', 'onevsone');
            
            % Parse input arguments
            p.parse(varargin{:});
            
            % Create SVM template
            obj.template = templateSVM(...
                'KernelFunction', p.Results.KernelFunction, ...
                'PolynomialOrder', p.Results.PolynomialOrder);
            
            % Save coding for the following training
            obj.coding = p.Results.Coding;
        end
        
        function str = toString(obj)
            % Create output string with class name and kernel function
            kernel = obj.template.ModelParams.KernelFunction;
            str = ['SVM (KernelFunction: ' kernel];
            
            % Append polynomial order if kernel is polynomial
            if strcmp(kernel, 'polynomial')
                order = obj.template.ModelParams.KernelPolynomialOrder;
                str = [str ', PolynomialOrder: ' num2str(order)];
            end
            
            % Append multiclass coding
            str = [str ', Coding: ' obj.coding];
            
            % Close parentheses
            str = [str ')'];
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % Extract labeled pixels
            featureList = validListFromSpatial(...
                trainFeatureCube, trainLabelMap, true);
            labelList = validListFromSpatial(...
                trainLabelMap, trainLabelMap, true);
            
            % Train multiclass model
            obj.model = fitcecoc(featureList, labelList, ...
                'Coding', obj.coding, 'Learners', obj.template);
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap)
            
            % Extract list of unlabeled pixels
            featureList = validListFromSpatial(evalFeatureCube, maskMap);
            
            % Predict labels
            predictedLabelList = obj.model.predict(featureList);
            
            % Rebuild map representation
            predictedLabelMap = rebuildMap(predictedLabelList, maskMap);
        end
    end
    
end

