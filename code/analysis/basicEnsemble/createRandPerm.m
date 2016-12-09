function [ permutation ] = createRandPerm(numInstances, proportion)
%CREATERANDPERM creates a random permutation with a given size
%
%   This function calculates a random permutation for a given number of
%   instances and a proportion, which describes how many instances should
%   be included in the permutation
%
%% Input:
%    numInstances . a scalar with the number of instances
%    proportion ... a scalar between 0 and 1, which describes how many
%                   instances should be included in the permutation
%
%% Output:
%    permutation .. a vector, with ceil(numInstances * proportion) elements
% 
% Version: 2016-12-05
% Author: Tuan Pham Minh
%

    permutation = randperm(numInstances, ceil(numInstances * proportion));
end

