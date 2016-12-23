classdef (Abstract) Classifier < matlab.mixin.Copyable
    %CLASSIFIER Abstract classifier superclass
    %
    %    Every classifier should inherit this class and overwrite the
    %    abstract methods.
    %
    %% Abstract Methods:
    %    trainOn ..... Train a model on the given data and return it. The 
    %                  function should make sure to train a fresh model on 
    %                  each call.
    %        trainFeatureCube ... Feature cube of dimensions X x Y x F with
    %                             X and Y being the image dimensions and F
    %                             being the number of features.
    %        trainLabelMap ...... Label Map o dimensions X x Y.
    %    classifyOn .. Classify the given data.
    %        evalFeatureCube .... Feature cube of dimensions X x Y x F with
    %                             X and Y being the image dimensions and F
    %                             being the number of features.
    %        maskMap ............ Map that indicates, which pixels are fill
    %                             pixels (-1) with dimensions X x Y.
    %        predictedLabelMap .. Output map of features, having dimensions
    %                             X x Y.
    %
    % Version: 2016-12-22
    % Author: Tilman Krokotsch
    %
    
    methods (Abstract)
        obj = trainOn(obj, trainFeatureCube, trainLabelMap);
        predictedLabelMap = classifyOn(obj, evalFeatureCube, maskMap);
    end
    
end

