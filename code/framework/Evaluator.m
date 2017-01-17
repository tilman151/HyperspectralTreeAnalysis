classdef Evaluator
    %EVALUATOR Computes evaluation measures on a confusion matrix
    
    methods (Static)
        
        function allMeasures = getAllMeasures(confMat)
            allMeasures.Accuracy = ...
                Evaluator.getAccuracy(confMat);
            allMeasures.Precisions = ...
                Evaluator.getPrecisions(confMat);
            allMeasures.Sensitivities = ...
                Evaluator.getSensitivities(confMat);
            allMeasures.Specificities = ...
                Evaluator.getSpecificities(confMat);
            allMeasures.PositiveLikelihoods = ...
                Evaluator.getPositiveLikelihoods(0, ...
                                             allMeasures.Sensitivities, ...
                                             allMeasures.Specificities);
            allMeasures.NegativeLikelihoods = ...
                Evaluator.getNegativeLikelihoods(0, ...
                                             allMeasures.Sensitivities, ...
                                             allMeasures.Specificities);
            allMeasures.FScores = ...
                Evaluator.getFScores(confMat);
        end
        
        function accuracy = getAccuracy(confMat)
            accuracy = sum(diag(confMat)) / sum(sum(confMat));
        end
        
        function precisions = getPrecisions(confMat)
            precisions = diag(confMat) ./ sum(confMat, 1)';
        end
        
        function sensitivities = getSensitivities(confMat)
            sensitivities = (diag(confMat) ./ sum(confMat, 2));
        end
        
        function specificities = getSpecificities(confMat)
            trueNegatives = repmat(sum(diag(confMat)), ...
                                   size(confMat, 1), ...
                                   1);
            trueNegatives = trueNegatives - diag(confMat);
            
            allNegatives = sum(confMat, 2);
            allNegatives = repmat(allNegatives, 1, numel(allNegatives));
            allNegatives = allNegatives - diag(diag(allNegatives));
            allNegatives = sum(allNegatives, 1);
            
            specificities = trueNegatives ./ allNegatives';
        end
        
        function posLikelihoods = getPositiveLikelihoods(confMat, ...
                                                         sensitivities, ...
                                                         specificities)
            if nargin == 1
                posLikelihoods = Evaluator.getSensitivities(confMat) ./ ...
                                 (1-Evaluator.getSpecificities(confMat));
            elseif nargin == 3
                posLikelihoods = sensitivities ./ (1 - specificities);
            end
        end
        
        function negLikelihoods = getNegativeLikelihoods(confMat, ...
                                                         sensitivities, ...
                                                         specificities)
            if nargin == 1
                negLikelihoods = ...
                           (1 - Evaluator.getSensitivities(confMat)) ./ ...
                           Evaluator.getSpecificities(confMat);
            elseif nargin == 3
                negLikelihoods = (1 - sensitivities) ./ specificities;
            end
        end
        
        function fScores = getFScores(confMat)
            precisions = Evaluator.getPrecisions(confMat);
            sensitivities = Evaluator.getSensitivities(confMat);
            fScores = 2 * (precisions .* sensitivities) ./ ...
                          (precisions + sensitivities);
        end
        
    end
    
end

