classdef TSVM < Classifier
    %TSVM transductive Support Vector Machine
    %
    %    Support vector machine that uses unlabeled data as additional
    %    information. Also known as Semi-Supervised SVM (S3VM).
    %
    %% Properties:
    %    C1 ......... Misclassification penalty (labeled data).
    %    C2 ......... Misclassification penalty (neutral data).
    %    k .......... Kernel function handle for the SVM.
    %    f .......... Classification function handle of the trained model.
    %    SX ......... Support vector points.
    %    SY ......... Support vector labels.
    %    SA ......... Support vector alphas (Lagrange multipliers).
    %
    %% Methods:
    %    TSVM ....... Constructor. Can take Name, Value pair arguments that
    %                 change the multiclass strategy and the internal
    %                 parameters of the SVM. Possible arguments:
    %        C1 ............. Misclassification penalty (labeled data).
    %        C2 ............. Misclassification penalty (neutral data).
    %        KernelFunction . Kernel function for the SVM.
    %                         'linear'(default) | 'sigmoidal' | 'rbf' | 
    %                         'polynomial'
    %        Coding ......... Coding design for the multiclass model.
    %                         'onevsone'(default) | 'onevsall'
    %    trainOn .... See documentation in superclass Classifier.
    %    classifyOn . See documentation in superclass Classifier.
    %
    % Version: 2016-11-24
    % Author: Cornelius Styp von Rekowski
    %
    
    properties
        % Parameters
        C1;
        C2;
        KernelFunction;
        Coding;
        
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
            p.addParameter('Coding', 'onevsone');
            
            % Parse input arguments
            p.parse(varargin{:});
            
            % Save parameters
            obj.C1 = p.Results.C1;
            obj.C2 = p.Results.C2;
            
            % Parse kernel function
            switch p.Results.KernelFunction
                case 'linear'
                    obj.KernelFunction = kernel_gen_lin;
                case 'sigmoidal'
                    obj.KernelFunction = kernel_gen_sig;
                case 'rbf'
                    obj.KernelFunction = kernel_gen_rbf;
                case 'polynomial'
                    obj.KernelFunction = kernel_gen_pol;
                otherwise
                    disp(['Unrecognized option for KernelFunction: '...
                          p.Results.KernelFunction]);
                    disp('Falling back to default linear kernel.');
                    obj.KernelFunction = kernel_gen_lin;
            end
            
            % Parse multiclass coding
            switch p.Results.Coding
                case 'onevsone'
                    obj.Coding = @svm_ovo;
                case 'onevsall'
                    obj.Coding = @svm_ova;
                otherwise
                    disp(['Unrecognized option for Coding: '...
                          p.Results.Coding]);
                    disp('Falling back to default one vs one coding.');
                    obj.Coding = @svm_ovo;
            end
        end
        
        function obj = trainOn(obj, trainFeatures, trainLabels)
            % Extract labeled pixels
            [featureList, labelList, unlabeledFeatureList] = ...
                extractLabeledPixels(trainFeatures, trainLabels);
            
            [obj.f, obj.SX, obj.SY, obj.SA, ~] = obj.Coding(...
                featureList, labelList, unlabeledFeatureList, ...
                obj.C1, obj.C2, obj.KernelFunction);
        end
        
        function labels = classifyOn(obj, evalFeatures)
            % Transform input map to vector
            featureList = mapToVec(evalFeatures);
            
            % Predict labels
            featureRows = num2cell(featureList, 2);
            labels = cellfun(obj.f, featureRows);
        end
    end
    
end

