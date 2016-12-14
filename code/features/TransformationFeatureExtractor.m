classdef TransformationFeatureExtractor < FeatureExtractor
    %TRANSFORMATIONFEATUREEXTRACTOR Abstract transformation feature
    %extractor superclass
    %
    %   Each feature extractor that calculates and applies a
    %   transformation on the original features should inherit this class.
    %
    %% Abstract Methods:
    %   calculateTransformation ... calculates and permanently saves a
    %                               transformation matrix based on a 
    %                               sample set
    %   applyTransformation ....... applies a transformation to the 
    %                               original features
    %   getTransformationFilename . returns the name of the file that 
    %                               contains the transformation matrix 
    %
    %% Methods
    %   extractFeatures . loads or (in case it does not exsit yet) 
    %                     calculates the transformation matrix and applies
    %                     it to the original features
    %
    % Version: 2016-12-09
    % Author: Marianne Stecklina
    %
    
    methods (Abstract)
        transformationMatrix = calculateTransformation(obj, sampleSet);
        features = applyTransformation(obj, originalFeatures, ...
            transformationMatrix);
        name = getTransformationFilename(obj)
    end
    
    methods
        function features = extractFeatures(obj, originalFeatures)
            if ~exist(getTransformationFilename(), 'file')
                sampleSet = load(['../data/ftp-iff2.iff.fraunhofer.de/' ...
                    'Data/FeatureExtraction/sampleSet.mat']);
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

