classdef (Abstract) Classifier
    %CLASSIFIER Abstract classifier superclass
    %
    %    Every classifier should inherit this class and overwrite the
    %    abstract methods.
    %
    %% Abstract Methods:
    %    trainOn ..... This method takes a feature cube of dimensions 
    %                  X x Y x F and a label map of dimensions X x Y as
    %                  input, with X x Y being the image dimensions and F
    %                  being the number of features.
    %                  A model is trained on the given data and returned.
    %                  The function should make sure to train and a fresh 
    %                  model on each call.
    %    classifyOn .. This method takes a feature cube of dimensions
    %                  X x Y x F as input, with X x Y being the image 
    %                  dimensions and F being the number of features.
    %                  The output should be a vector of labels with an
    %                  entry for each image pixel -> (X * Y) x 1.
    %
    % Version: 2016-11-24
    % Author: Tilman Krokotsch
    %
    
    methods (Abstract)
        obj = trainOn(obj,trainFeatures,trainLabels);
        labels = classifyOn(obj,evalFeatures);
    end
    
end

