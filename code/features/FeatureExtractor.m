classdef FeatureExtractor
    %FEATUREEXTRACTOR Abstract feature extractor superclass
    %
    %   Each feature extractor should inherit this class.
    %
    %% Abstract Methods:
    %   toString ........ returns a string representation of the object
    %   extractFeatures . loads or (in case it does not exist yet) 
    %                     calculates the transformation matrix and applies
    %                     it to the original features
    %
    % Version: 2017-01-13
    % Author: Marianne Stecklina
    %
    
    methods (Abstract)
        str = toString(obj)
        features = extractFeatures(obj, originalFeatures, sampleSetPath)
    end
    
end

