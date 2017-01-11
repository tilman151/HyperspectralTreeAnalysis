
function [data_new_hang, index_hang]=array_hang(X)
%%% Author: Mao Shasha,skymss0828@gmail.com,2008.6.2 %%%%

%%% for row vector of X %%%
rng('shuffle'); 
index_hang=randperm(size(X,1));    
data_new_hang=X(index_hang,:);
