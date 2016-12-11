function map = vecToMap(vector, x, y)
%VECTOMAP Reshape a given vector back to the spatial map representation
%
%    A given vector of features or labels with dimensions (X * Y) x F is
%    reshaped to the original map representation with dimensions X x Y x F.
%
%% Input:
%    vector .. Vector of features or labels. Dimensions (X * Y) x F with X 
%              and Y being the image dimensions and F being the number of 
%              features (F = 1 for labels).
%
%% Output:
%    map ..... Reshaped map with dimensions X x Y x F.
%
% Version: 2016-12-10
% Author: Cornelius Styp von Rekowski
%

% TODO: Re-insert empty pixels (class -1). This needs a given mask.
map = reshape(vector, x, y, []);

end

