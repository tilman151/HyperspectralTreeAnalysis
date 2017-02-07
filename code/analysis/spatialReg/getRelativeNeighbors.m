function neighborIdxs = getRelativeNeighbors(r)
%GETRELATIVENEIGHBORS Summary of this function goes here
%   Detailed explanation goes here

    % Create relative indices of neighboring pixels in range
    neighborIdxs = [];
    for addX = -r : r
        % Use euclidean distance (based on sqrt(x^2 + y^2) = r)
        euclY = fix(sqrt(r^2 - addX^2));
        for addY = -euclY : euclY
            neighborIdxs = [neighborIdxs; addX addY];
        end
    end
end

