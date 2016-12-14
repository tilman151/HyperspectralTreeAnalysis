classdef FeatureExtractor
    %FEATUREEXTRACTOR Abstract feature extractor superclass
    %
    %   Each feature extractor should inherit this class.
    %
    %% Abstract Methods:
    %   extractFeatures . loads or (in case it does not exsit yet) 
    %                     calculates the transformation matrix and applies
    %                     it to the original features
    %
    % Version: 2016-12-09
    % Author: Marianne Stecklina
    %
    
    methods (Abstract)
        features = extractFeatures(obj, originalFeatures)
    end
    
end

