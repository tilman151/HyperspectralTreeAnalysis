classdef NDVI < FeatureExtractor    
    %ContinuumRemoval continuum removal
    %
    %    NDVI is a wrapper class for ndvi and calculates the NDVI index
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
        function features = extractFeatures(obj, originalFeatures, ~, ~)
            features = ndvi(originalFeatures);
        end
        
        function str = toString(obj)
            str = class(obj);
        end
        
        function str = toShortString(obj)
            str = obj.toString();
        end
    end
end
function [ result ] = ndvi( rawFeatures )
%NDVI Summary of this function goes here
%   Detailed explanation goes here

    r800 = rawFeatures(:,:,107);
    r670 = rawFeatures(:,:,72);
    result = (r800 - r670)./(r800 + r670);
    
    isNanMask = isnan(result);
    result(isNanMask) = 0;
end

