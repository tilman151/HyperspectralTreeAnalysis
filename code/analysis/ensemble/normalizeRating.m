function [ normalizedScores ] = normalizeRating( scores, projectedRange )
%NORMALIZERATING normalize scores of the classifier
%
%% Input:
%    scores         ............. the scores of the base classifiers
%
%% Output:
%    normalizedScores ........... the rating of the classifier
% 
% Version: 2017-01-31
% Author: Tuan Pham Minh


if min(scores) ~= max(scores)
    normalizedScores = (scores - min(scores))/(max(scores) - min(scores));
    normalizedScores = ...
        (normalizedScores) * ...
        (max(projectedRange) - min(projectedRange)) + min(projectedRange);
else
    normalizedScores = ones(size(scores)) * min(projectedRange);
end

end

