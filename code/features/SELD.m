classdef SELD < TransformationFeatureExtractor   
    %SELD semisupervised local discriminant analysis
    %
    %    SELD finds and applies a feature transformation that preserves 
    %    local neighborhoods and maximizes class separation. The algorithm 
    %    was proposed by Wenzhi et al. (2013).
    %
    %% Properties:
    %    k ...... number of neighbors considered in the unsupervised local 
    %             linear feature extraction method
    %    numDim . desired number of dimensions of the extracted features
    %
    %% Methods:
    %    SELD ....... Constructor. Internally saves the parameters.
    %                 k ...... See properties.
    %                 numDim . See properties.
    %    calculateTransformation ... See documentation in superclass
    %                                TransformationFeatureExtractor.
    %    applyTransformation ....... See documentation in superclass
    %                                TransformationFeatureExtractor.
    %    getTransformationFilename . See documentation in superclass
    %                                TransformationFeatureExtractor.
    %    extractFeatures ........... See documentation in superclass
    %                                TransformationFeatureExtractor.
    %                                Returns (width x height x numDim) cube 
    %                                with the extracted features of the  
    %                                (weight x height) input instances.
    %
    % Version: 2016-12-12
    % Author: Marianne Stecklina
    %
    
    properties
        k;
        numDim;
    end
    
    methods
        function obj = SELD(k, numDim)
           obj.k = k; 
           obj.numDim = numDim;
        end
        
        function transformationMatrix = calculateTransformation(obj, sampleSet)
            allFeatures = [sampleSet.features; sampleSet.unlabeledFeatures];
            zeroLabels  = zeros(size(sampleSet.labels));
            allLabels   = [sampleSet.labels zeroLabels];
            [transformationMatrix, ~] = seld.SELD(allFeatures, ...
                allLabels, obj.numDim, obj.k);
        end
        
        function features = applyTransformation(obj, originalFeatures, ...
            transformationMatrix)         
            [width, height, numFeatures] = size(originalFeatures);
        
            reshapedFeatures = reshape(originalFeatures, ...
                width * height, numFeatures); 
            
            features = reshape(reshapedFeatures * transformationMatrix, ...
                width, height, obj.numDim);
        end
        
        function name = getTransformationFilename(obj)
            name = ['SELD_' int2str(obj.numDim) '_' int2str(obj.k) '.mat'];  
        end
    end
    
end