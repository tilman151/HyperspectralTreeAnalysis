classdef (Abstract) Classifier < matlab.mixin.Copyable
    %CLASSIFIER Abstract classifier superclass
    %
    %    Every classifier should inherit this class and overwrite the
    %    abstract methods.
    %
    %% Abstract Methods:
    %    toString ...... Return a string representation of the object.
    %    toShortString . Return a short string representation without
    %                    special characters that can for example be used as
    %                    a directory name.
    %    trainOn ....... Train a model on the given data and return it. The 
    %                    function should make sure to train a fresh model 
    %                    on each call.
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
    %    saveTo ...... Save the model to a given folder. The model will be
    %                  stored in the file model.mat.
    %    loadFrom .... Load the model from a given folder. The model is
    %                  expected to be in the file model.mat
    %
    % Version: 2016-12-22
    % Author: Tilman Krokotsch
    %
    
    methods (Abstract)
        str = toString(obj);
        str = toShortString(obj);
        obj = trainOn(obj, trainFeatureCube, trainLabelMap);
        predictedLabelMap = classifyOn(obj, evalFeatureCube, maskMap);
    end
    
    methods
        function model = saveTo(model, folder)
            save(fullfile(folder, 'model.mat'), 'model');
        end
    end
    
    methods (Static)
        function model = loadFrom(folder)
            load(fullfile(folder, 'model.mat'), 'model');
        end
    end
    
end

