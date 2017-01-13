classdef PCA < TransformationFeatureExtractor
    %PCA principal component analysis
    %
    %    PCA finds and applies an orthogonal transformation in order to
    %    obtain linearly uncorrelated features.
    %
    %% Properties:
    %    numDim . desired number of dimensions of the extracted features
    %
    %% Methods:
    %    PCA ........ Constructor. Internally saves the parameters.
    %                 numDim . See properties.
    %    toString .................. See documentation in superclass
    %                                FeatureExtractor.
    %    toShortString ............. See documentation in superclass
    %                                FeatureExtractor.
    %    calculateTransformation ... See documentation in superclass
    %                                TransformationFeatureExtractor.
    %    applyTransformation ....... See documentation in superclass
    %                                TransformationFeatureExtractor.
    %    getTransformationFilename . See documentation in superclass
    %                                TransformationFeatureExtractor.
    %    extractFeatures ........... See documentation in superclass
    %                                FeatureExtractor.
    %                                Returns (width x height x numDim) cube 
    %                                with the extracted features of the  
    %                                (weight x height) input instances.
    %
    % Version: 2017-01-13
    % Author: Marianne Stecklina
    %
    
    properties
        numDim;
    end
    
    methods
        function obj = PCA(numDim)
           obj.numDim = numDim;
        end
        
        function str = toString(obj)
            str = ['PCA (numDim: ' int2str(obj.numDim) ')'];
        end
        
        function str = toShortString(obj)
            str = ['PCA_' int2str(obj.numDim)];
        end
        
        function transformationMatrix = calculateTransformation(obj, ...
                sampleSet)
            allFeatures = [sampleSet.features; ...
                sampleSet.unlabeledFeatures];
            [transformationMatrix, ~] = pca(allFeatures);
        end
        
        function features = applyTransformation(obj, originalFeatures, ...
            transformationMatrix)         
            [width, height, numFeatures] = size(originalFeatures);
        
            reshapedFeatures = reshape(originalFeatures, ...
                width * height, numFeatures); 
            
            features = reshape(reshapedFeatures * ...
                transformationMatrix(:, 1:obj.numDim), width, height, ...
                obj.numDim);
        end
        
        function name = getTransformationFilename(obj)
            name = 'PCA.mat';  
        end
    end
    
end

