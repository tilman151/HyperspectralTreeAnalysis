classdef MCAR1 < FeatureExtractor    
    %ContinuumRemoval continuum removal
    %
    %    MCAR1 is a wrapper class for mcar1 and calculates the MCAR1 index
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
            features = mcar1(originalFeatures);
        end
        
        function str = toString(obj)
            str = class(obj);
        end
        
        function str = toShortString(obj)
            str = obj.toString();
        end
    end
end
function [ result ] = mcar1( rawFeatures )
%MCAR1 Summary of this function goes here
%   Detailed explanation goes here

    r700 = rawFeatures(:,:,80);
    r670 = rawFeatures(:,:,72);
    r550 = rawFeatures(:,:,41);
    result = ((r700 - r670) - 0.2*(r700-r550)).*r700./r670;
    
    isNanMask = isnan(result);
    isInfMask = isinf(result);
    result(isNanMask | isInfMask) = 0;
end

