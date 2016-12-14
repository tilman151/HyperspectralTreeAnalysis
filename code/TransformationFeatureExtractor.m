classdef TransformationFeatureExtractor < FeatureExtractor
    %TRANSFORMATIONFEATUREEXTRACTOR Abstract transformation feature
    %extractor superclass
    %
    %   Each feature extractor that calculates and applies a transformation
    %   on the original features should inherit this class.
    %
    %% Abstract Methods:
    %   calculateTransformation ... calculates and permanently saves a
    %                               transformation matrix based on a 
    %                               sample set
    %       sampleSet . struct containing features (n x numFeatures),
    %                   labels (n x 1) and unlabeledFeatures
    %                   (n x numFeatures)
    %   applyTransformation ....... applies a transformation to the 
    %                               original features
    %       originalFeatures . (width x height x numFeatures) cube
    %                          containing the data
    %   getTransformationFilename . returns the name of the file that 
    %                               contains the transformation matrix 
    %
    %% Methods
    %   extractFeatures . loads or (in case it does not exsit yet) 
    %                     calculates the transformation matrix and applies
    %                     it to the original features
    %
    % Version: 2016-12-12
    % Author: Marianne Stecklina
    %
    
    methods (Abstract)
        transformationMatrix = calculateTransformation(obj, sampleSet);
        features = applyTransformation(obj, originalFeatures, ...
            transformationMatrix);
        name = getTransformationFilename(obj)
    end
    
    methods
        function features = extractFeatures(obj, originalFeatures, ...
                sampleSetPath)
            if ~exist(getTransformationFilename(), 'file')
                sampleSet = load(sampleSetPath);
                transformationMatrix = calculateTransformation(obj, ...
                    sampleSet);
            else
                transformationMatrix = load(getTransformationFilename());
            end
            features = applyTransformation(obj, originalFeatures, ...
                transformationMatrix);
        end
    end
    
end

