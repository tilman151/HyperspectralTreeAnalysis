

function [data_new_lie, index_lie]=array_lie(X)
%%% Author: Mao Shasha,skymss0828@gmail.com,2008.6.2 %%%%

%%% for column vector of X %%%%
rng('shuffle'); 
index_lie=randperm(size(X,2));    
data_new_lie=X(:,index_lie);
