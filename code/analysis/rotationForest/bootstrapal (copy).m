
function [Xpart,Ypart,indexselect]=bootstrapal(Xini,Yini,Ratio)
%%% skymss,2011.1.22 (email: skymss@126.com) %%%%
%
% bootstrapal: bootstrap algorithm used in bagging ensemble;
%
% Function: [Xpart,Ypart,indexselect]=bootstrapal(Xini,Yini,Ratio)
%
% Input:  Xini: the original samples;
%         Yini: the labels of original samples;
%         Ratio: the proportion of resampling new samples from original samples;
% Ourput: Xpart: the new samples;
%         Ypart: the labels of Xpart;
%         indexselect: the index of selected samples;
%
% Author: Mao Shasha (skymss0828@gmail.com),2011.1.22 %

if (nargin<2 || nargin>3)
    help bootstrapal
else
    [N,~]=size(Xini);
    A=randperm(N); 

    numberselect=ceil(N*Ratio);
    Aselect=(A(1:numberselect))';

    Xpartini=Xini(Aselect,:);
    Ypartini=Yini(Aselect,:);

    numberleaver=N-numberselect;

    index1=(1:numberselect)';
    index1 = index1./numberselect;

    index2=rand(numberleaver,1);
    index4=ones(numberleaver)';
    
    for i=1:numberleaver
        index3=index2(i);
        for j=1:numberselect
            if (index1(j)>=index3 && j==1)
                index4(i,1)=j;
            else
                if (index1(j)>=index3 && index1(j-1)<index3)
                    index4(i,1)=j;
                end
            end
        end
    end
    Xpart=[Xpartini;Xpartini(index4,:);];
    Ypart=[Ypartini;Ypartini(index4,:);];
    indexselect=[Aselect;Aselect(index4);];
end
