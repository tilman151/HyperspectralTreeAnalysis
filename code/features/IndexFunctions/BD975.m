classdef BD975 < FeatureExtractor    
    %ContinuumRemoval continuum removal
    %
    %    BD975 is a wrapper class for BD975 and calculates the BD975 index
    %
    %% Methods:
    %    ContinuumRemoval. Constructor.
    %    extractFeatures . See documentation in superclass
    %                      TransformationFeatureExtractor.
    %                      Returns (width x height x numDim) cube 
    %                      with the extracted features of the  
    %                      (weight x height) input instances.
    %
    % Version: 2016-12-14
    % Author: Tuan Pham Minh

    methods
        function features = extractFeatures(obj, originalFeatures)
            features = bd975(originalFeatures);
        end
        
        function str = toString(obj)
            str = class(obj);
        end
        
        function str = toShortString(obj)
            str = obj.toString();
        end
    end
end

function [ result ] = bd975( rawFeatures)
%BD975 Summary of this function goes here
%   Detailed explanation goes here
    continuumRemover = ContinuumRemoval(true);
    result = continuumRemover.extractFeatures(rawFeatures);
    result = result(:,:,154);
end

