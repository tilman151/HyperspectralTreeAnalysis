function [acraw, aclda,aclpp, acildlp, acnldlp]=repeat5n(spectraldata,grd,traingrd,class_num,no_dim)

neighbor=8;
acraw=[];
aclda=[];
aclpp=[];
acildlp=[];
acnldlp=[];


for i=1:3
    [araw, alda , alpp, aildlp, anldlp]=compareall(spectraldata, grd, traingrd, i, neighbor, 0.5, class_num);
    acraw=[acraw,araw];
    aclda=[aclda,alda];
    aclpp=[aclpp,alpp];  
    acildlp=[acildlp,aildlp];
    acnldlp=[acnldlp,anldlp];
end

for i=4:no_dim
    [araw, alpp, aildlp, anldlp]=compareall2(spectraldata, grd, traingrd, i, neighbor, 0.5, class_num);
    acraw=[acraw,araw];
    aclda=[aclda,0];
    aclpp=[aclpp,alpp];  
    acildlp=[acildlp,aildlp];
    acnldlp=[acnldlp,anldlp];

end



