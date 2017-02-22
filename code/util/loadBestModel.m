function model = loadBestModel(experimentPath)
%LOADBESTMODEL Find and load the best model from the experiment
%    
%    Find and load the best model regarding accuracy from the trained 
%    models in the given experiment folder.
%
%% Input:
%    experimentPath .. Path to the experiment folder
%
%% Output:
%    bestModel ....... The loaded best model regarding accuracy
%
% Version: 2017-02-21
% Author: Cornelius Styp von Rekowski
% 
    
    % Get list of all trained models
    modelFiles = dir(fullfile(experimentPath, 'model_*.mat'));
    
    % Find the best model regarding accuracy
    bestAccuracy = -1;
    for k = 1:length(modelFiles)
        % Load confusion matrix
        load(fullfile(experimentPath, modelFiles(k).name), 'confMat');
        
        % Calculate accurcay
        accuracy = Evaluator.getAccuracy(confMat);
        
        % Update best fold
        if accuracy > bestAccuracy
            bestAccuracy = accuracy;
            bestK = k;
        end
    end
    
    % Load best model
    load(fullfile(experimentPath, modelFiles(bestK).name), 'model');
end

