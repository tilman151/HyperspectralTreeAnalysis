classdef SVMlibsvm < Classifier
    %SVMlibsvm Standard Support Vector Machine for multiclass problems
    %
    %    This class uses LIBSVM as its underlying SVM implementation.
    %    It requires a compiled version of LIBSVM, which can be downloaded
    %    here: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
    %
    %% Properties:
    %    kernel .......... Name of the kernel function.
    %    polynomialOrder . Order of the polynomial kernel function.
    %    coding .......... Name of the multiclass coding design.
    %
    %% Methods:
    %    SVMlibsvm ..... Constructor. Can take Name, Value pair arguments 
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
        % Trained model
        model;
    end
    
    methods
        function obj = SVMlibsvm(varargin)
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
            str = ['SVM [LIBSVM] (KernelFunction: ' obj.kernel];
            
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
            str = ['LibSVM_' obj.kernel];
            
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
            
            % Extract labeled pixels
            featureList = validListFromSpatial(...
                trainFeatureCube, trainLabelMap, true);
            labelList = validListFromSpatial(...
                trainLabelMap, trainLabelMap, true);
            
            % Build additional parameters
            params = '-q';
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
                    obj.model = svmtrain(labelList, featureList, params);
                otherwise
                    logger.error('SVM', ['Currently only one-vs-one '...
                        'coding is supported in LIBSVM!']);
                    exit;
            end
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap, ~)
            
            % Extract list of unlabeled pixels
            featureList = validListFromSpatial(evalFeatureCube, maskMap);
            labelList = zeros(size(featureList, 1), 1);
            
            % Predict labels
            predictedLabelList = ...
                svmpredict(labelList, featureList, obj.model);
            
            % Rebuild map representation
            predictedLabelMap = rebuildMap(predictedLabelList, maskMap);
        end
    end
    
end

