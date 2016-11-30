function featureVec = mapToVec(featureMap)
%MAPTOVEC Reshape a given feature map to a single vector
%
%    Input has dimensions X x Y x F, output has dimensions (X * Y) x F. This 
%    also works for label vectors having F=1.
%
%% Input:
%    featureMap . Map of features for each image pixel. Dimensions X x Y x F
%                 with X and Y being the image dimensions and F being the number
%                 of features.
%
%% Output:
%    featureVec . Reshaped feature map with dimensions (X * Y) x F.
%
% Version: 2016-11-29
% Author: Cornelius Styp von Rekowski
%

[x, y, f] = size(featureMap);
featureVec = reshape(featureMap, x*y, f);

end

