classdef TSVM < Classifier
    %TSVM transductive Support Vector Machine
    %
    %    Support vector machine that uses unlabeled data as additional
    %    information. Also known as Semi-Supervised SVM (S3VM).
    %
    %% Properties:
    %    C1 .............. Misclassification penalty (labeled data).
    %    C2 .............. Misclassification penalty (neutral data).
    %    KernelName ...... Name of the kernel function.
    %    KernelFunction .. Kernel function handle for the SVM.
    %    PolynomialOrder . Order of the polynomial kernel function.
    %    Coding .......... Name of the multiclass coding design.
    %    CodingFunction .. Function handle for the multiclass coding.
    %    f ............... Classification function handle of the trained 
    %                      model.
    %    SX .............. Support vector points.
    %    SY .............. Support vector labels.
    %    SA .............. Support vector alphas (Lagrange multipliers).
    %
    %% Methods:
    %    TSVM ....... Constructor. Can take Name, Value pair arguments that
    %                 change the multiclass strategy and the internal
    %                 parameters of the SVM. Possible arguments:
    %        C1 .............. Misclassification penalty (labeled data).
    %        C2 .............. Misclassification penalty (neutral data).
    %        KernelFunction .. Kernel function for the SVM.
    %                          'linear'(default) | 'sigmoidal' | 'rbf' | 
    %                          'polynomial'
    %        PolynomialOrder . Positive integer specifying the degree of
    %                          polynomial to be used for polynomial
    %                          kernel. This parameter is used only if
    %                          you set 'KernelFunction' to 'polynomial'.
    %                          Default: 3
    %        Coding .......... Coding design for the multiclass model.
    %                          'onevsone'(default) | 'onevsall'
    %    toString ... See documentation in superclass Classifier.
    %    trainOn .... See documentation in superclass Classifier.
    %    classifyOn . See documentation in superclass Classifier.
    %
    % Version: 2016-12-22
    % Author: Cornelius Styp von Rekowski
    %
    
    properties
        % Parameters
        C1;
        C2;
        KernelName;
        PolynomialOrder;
        Coding;
    end
    
    properties(Hidden=true)
        % Function handles
        KernelFunction;
        CodingFunction;
        
        % Output results
        f; 
        SX; 
        SY;
        SA;
    end
    
    methods
        function obj = TSVM(varargin)
            % Create input parser
            p = inputParser;
            p.addParameter('C1', 10);
            p.addParameter('C2', 10);
            p.addParameter('KernelFunction', 'linear');
            p.addParameter('PolynomialOrder', 3);
            p.addParameter('Coding', 'onevsone');
            
            % Parse input arguments
            p.parse(varargin{:});
            
            % Save parameters
            obj.C1 = p.Results.C1;
            obj.C2 = p.Results.C2;
            
            % Parse kernel function
            obj.KernelName = p.Results.KernelFunction;
            switch p.Results.KernelFunction
                case 'linear'
                    obj.KernelFunction = kernel_gen_lin;
                case 'sigmoidal'
                    obj.KernelFunction = kernel_gen_sig;
                case 'rbf'
                    obj.KernelFunction = kernel_gen_rbf;
                case 'polynomial'
                    obj.KernelFunction = kernel_gen_pol(...
                        [0, p.Results.PolynomialOrder]);
                    obj.PolynomialOrder = p.Results.PolynomialOrder;
                otherwise
                    disp(['Unrecognized option for KernelFunction: '...
                          p.Results.KernelFunction]);
                    disp('Falling back to default linear kernel.');
                    obj.KernelName = 'linear';
                    obj.KernelFunction = kernel_gen_lin;
            end
            
            % Parse multiclass coding
            obj.Coding = p.Results.Coding;
            switch p.Results.Coding
                case 'onevsone'
                    obj.CodingFunction = @svm_ovo;
                case 'onevsall'
                    obj.CodingFunction = @svm_ova;
                otherwise
                    disp(['Unrecognized option for Coding: '...
                          p.Results.Coding]);
                    disp('Falling back to default one vs one coding.');
                    obj.Coding = 'onevsone';
                    obj.CodingFunction = @svm_ovo;
            end
        end
        
        function str = toString(obj)
            % Create output string with class name and kernel function
            str = ['t-SVM (KernelFunction: ' obj.KernelName];
            
            % Append polynomial order if kernel is polynomial
            if obj.PolynomialOrder
                str = [str ', PolynomialOrder: ' ...
                       num2str(obj.PolynomialOrder)];
            end
            
            % Append misclassification penalties
            str = [str ', C1: ' num2str(obj.C1) ', C2: ' num2str(obj.C2)];
            
            % Append multiclass coding
            str = [str ', Coding: ' obj.Coding];
            
            % Close parentheses
            str = [str ')'];
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % Extract valid pixels as lists
            featureList = validListFromSpatial(...
                trainFeatureCube, trainLabelMap);
            labelList = validListFromSpatial(...
                trainLabelMap, trainLabelMap);
            
            % Extract labeled pixels
            labeledFeatureList = featureList(labelList > 0, :);
            labeledLabelList = labelList(labelList > 0);
            
            % Extract unlabeled pixels
            unlabeledFeatureList = featureList(labelList == 0, :);
            
            [obj.f, obj.SX, obj.SY, obj.SA, ~] = obj.CodingFunction(...
                labeledFeatureList, labeledLabelList, ...
                unlabeledFeatureList, ...
                obj.C1, obj.C2, obj.KernelFunction);
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap)
            
            % Extract unlabeled pixels as list
            featureList = validListFromSpatial(evalFeatureCube, maskMap);
            
            % Predict labels
            featureRows = num2cell(featureList, 2);
            predictedLabelList = cellfun(obj.f, featureRows);
            
            % Rebuild map representation
            predictedLabelMap = rebuildMap(predictedLabelList, maskMap);
        end
    end
    
end

