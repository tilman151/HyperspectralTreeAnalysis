classdef MulticlassLda < TransformationFeatureExtractor
    %MulticlassLda multi class linear discriminance analysis
    %
    %    MulticlassLda finds a transformationmatrix, which transforms the
    %    data into a space, where the classes are separated
    %
    %% Properties:
    %    numDim . desired number of dimensions of the extracted features
    %
    %% Methods:
    %    MulticlassLda ............. Constructor. 
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
    %                                TransformationFeatureExtractor.
    %                                Returns (width x height x numDim) cube 
    %                                with the extracted features of the  
    %                                (weight x height) input instances.
    %
    % Version: 2016-12-14
    % Author: Tuan Pham Minh
    %
    
    methods
        
        function transformationMatrix = calculateTransformation(obj, ...
                sampleSet)
            [~, transformationMatrix] = ...
                multiclassLda(sampleSet.features, sampleSet.labels);
        end
        
        function str = toString(obj)
            str = 'MulticlassLda';
        end
        
        function str = toShortString(obj)
            str = obj.toString();
        end
        
        function features = applyTransformation(obj, originalFeatures, ...
            transformationMatrix)         
            [width, height, numFeatures] = size(originalFeatures);
        
            reshapedFeatures = reshape(originalFeatures, ...
                width * height, numFeatures); 
            
            features = reshape(reshapedFeatures * ...
                transformationMatrix, width, height, []);
        end
        
        function name = getTransformationFilename(obj)
            name = 'MulticlassLda.mat';  
        end
    end
end

function [ transformedFeatures, transformationMatrix ] = multiclassLda( rawFeatures, classes)
%MULTICLASSLDA calculate the multiclass lda transformation matrix
%
%% Input:
%    rawFeatures ......... a 3-dimensional matrix with the dimensions 
%                          X x Y x Z, where X is the width, Y is the height
%                          and Z is the number of features per pixel
%    classes ............. a 2-dimensional matrix with the dimensions 
%                          X x Y, where X is the width, Y is the height and
%                          contains the classes corresponding to the
%                          features in rawFeatures
%                       
%
%% Output:
%    transformedFeatures . a 3-dimensional matrix with the same size as
%                          rawFeatures, but filled with the transformed
%                          data
% 
% Version: 2016-11-21
% Author: Tuan Pham Minh
%


% % get the dimensions of the raw features
% [x,spectralBands] = size(rawFeatures);
% % transform the data into a 2-dimensional matrix, such that each pixel and
% % its spectral bands are represented by a row
% reshapedFeatures = reshape(rawFeatures, x, spectralBands);
% % transform the classes into a 2-dimensional matrix, so that it matches the
% % structure of transformedFeatures
% reshapedClasses = reshape(classes, x, 1);
% get the number of classes in the data set
[numClasses, ~] = size(unique(classes));
% calculate the transformed features with the help of the multiclass lda
% implementation of Sultan Alzahrani
[transformedFeatures, transformationMatrix] = ...
                FDA(rawFeatures', classes, numClasses - 1);

% % bring the transformed features into the 3-dimensional form
% transformedFeatures = ...
%                 reshape(reshapedTransformedFeatures, x, y, numClasses - 1);

end