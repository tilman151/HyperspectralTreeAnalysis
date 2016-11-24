function featureVec = mapToVec(featureMap)
%MAPTOVEC Reshape a given feature map to a single vector
%
%    Input has dimensions X * Y * F, output has dimensions (X * Y) * F
%
%% Input:
%    featureMap . 
%
%% Output:
%    featureVec . 
%
% Version: 2016-11-24
% Author: Cornelius Styp von Rekowski
%

x = size(featureMap, 1);
y = size(featureMap, 2);
f = size(featureMap, 3);
featureVec = reshape(featureMap, x*y, f);

end

