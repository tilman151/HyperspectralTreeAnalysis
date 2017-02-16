classdef Indices < FeatureExtractor    
    %Indices calculate indices
    %
    %    Indices is a wrapper class to compute multiple indices with the
    %    help of FeatureExtractionMerger
    %
    %% Properties:
    %   featureExtractionMerger . the emrger, which is used to concatenate
    %                               all the indices
    %
    %% Methods:
    %    Indices ......... Constructor.
    %    toString ........ See documentation in superclass
    %                      FeatureExtractor.
    %    toShortString ... See documentation in superclass
    %                      FeatureExtractor.
    %    extractFeatures . See documentation in superclass
    %                      FeatureExtractor.
    %
    %
    % Version: 2017-01-17
    % Author: Tuan Pham Minh
    
    properties
        featureExtractionMerger;
    end

    methods
        
        function obj = Indices()
            indexExtractors = {DI1, GNDVI, MCAR1, NDVI, PRI, WI};
            obj.featureExtractionMerger = ...
                FeatureExtractionMerger(indexExtractors);
        end
        
        function str = toString(obj)
            str = 'Indices';
        end
        
        function str = toShortString(obj)
            str = obj.toString();
        end
        
        function features = extractFeatures(obj, originalFeatures, ...
                                            maskMap, samplsesetPath)
            features = ...
                obj.featureExtractionMerger.extractFeatures(...
                    originalFeatures, maskMap, samplsesetPath ...
                );
        end
    end
    
end