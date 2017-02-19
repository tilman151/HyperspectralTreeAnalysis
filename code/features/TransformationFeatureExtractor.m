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
        function features = extractFeatures(obj, originalFeatures, ...
                                            ~, sampleSetPath)
                                        
            exportClassPath =['../data/ftp-iff2.iff.fraunhofer.de/',...
                'FeatureExtraction/TransformationMatrices/',...
                class(obj), '/'];
%             exportClassPath =['../../FeatureExtraction/TransformationMatrices/',...
%                 class(obj), '/'];

            exportPath = [exportClassPath, ...
                obj.getTransformationFilename()];
            
            if ~exist(exportPath, 'file')
                
                sampleSet = load(sampleSetPath);
                transformationMatrix = ...
                    obj.calculateTransformation(sampleSet);
                
                if ~exist(exportClassPath, 'dir')
                   mkdir(exportClassPath);
                end
                
                save(exportPath, 'transformationMatrix')
            else
                transformationMatrix = ...
                    load(exportPath);
                transformationMatrix = ...
                    transformationMatrix.transformationMatrix;
            end
            features = obj.applyTransformation(originalFeatures, ...
                transformationMatrix);
        end
    end
    
end

