function [labeled, unlabeled]= getlabelmatrix2(groundtruth, precent)

B= groundtruth;
precent=1-precent;
% mask=zeros(size(groundtruth));
[x1,y1]=find(groundtruth==1);
[x2,y2]=find(groundtruth==2);
[x3,y3]=find(groundtruth==3);
[x4,y4]=find(groundtruth==4);
[x5,y5]=find(groundtruth==5);
[x6,y6]=find(groundtruth==6);
[x7,y7]=find(groundtruth==7);
[x8,y8]=find(groundtruth==8);
[x9,y9]=find(groundtruth==9);
[x10,y10]=find(groundtruth==10);
[x11,y11]=find(groundtruth==11);
[x12,y12]=find(groundtruth==12);
[x13,y13]=find(groundtruth==13);
[x14,y14]=find(groundtruth==14);
[x15,y15]=find(groundtruth==15);
[x16,y16]=find(groundtruth==16);

Tx=[];Ty=[];

 
xx1=randperm(size(x1,1));
for i=1:ceil(size(x1,1)*precent)
    Tx=[Tx,x1(xx1(i))];
    Ty=[Ty,y1(xx1(i))];
end

xx2=randperm(size(x2,1));
for i=1:ceil(size(x2,1)*precent)
    Tx=[Tx,x2(xx2(i))];
    Ty=[Ty,y2(xx2(i))];
end


xx3=randperm(size(x3,1));
for i=1:ceil(size(x3,1)*precent)
    Tx=[Tx,x3(xx3(i))];
    Ty=[Ty,y3(xx3(i))];
end

xx4=randperm(size(x4,1));
for i=1:ceil(size(x4,1)*precent)
    Tx=[Tx,x4(xx4(i))];
    Ty=[Ty,y4(xx4(i))];
end

xx5=randperm(size(x5,1));
for i=1:ceil(size(x5,1)*precent)
    Tx=[Tx,x5(xx5(i))];
    Ty=[Ty,y5(xx5(i))];
end


xx6=randperm(size(x6,1));
for i=1:ceil(size(x6,1)*precent)
    Tx=[Tx,x6(xx6(i))];
    Ty=[Ty,y6(xx6(i))];
end


xx7=randperm(size(x7,1));
for i=1:ceil(size(x7,1)*precent)
    Tx=[Tx,x7(xx7(i))];
    Ty=[Ty,y7(xx7(i))];
end


xx8=randperm(size(x8,1));
for i=1:ceil(size(x8,1)*precent)
    Tx=[Tx,x8(xx8(i))];
    Ty=[Ty,y8(xx8(i))];
end


xx9=randperm(size(x9,1));
for i=1:ceil(size(x9,1)*precent)
    Tx=[Tx,x9(xx9(i))];
    Ty=[Ty,y9(xx9(i))];
end


xx10=randperm(size(x10,1));
for i=1:ceil(size(x10,1)*precent)
    Tx=[Tx,x10(xx10(i))];
    Ty=[Ty,y10(xx10(i))];
end


xx11=randperm(size(x11,1));
for i=1:ceil(size(x11,1)*precent)
    Tx=[Tx,x11(xx11(i))];
    Ty=[Ty,y11(xx11(i))];
end


xx12=randperm(size(x12,1));
for i=1:ceil(size(x12,1)*precent)
    Tx=[Tx,x12(xx12(i))];
    Ty=[Ty,y12(xx12(i))];
end




xx13=randperm(size(x13,1));
for i=1:ceil(size(x13,1)*precent)
    Tx=[Tx,x13(xx13(i))];
    Ty=[Ty,y13(xx13(i))];
end



xx14=randperm(size(x14,1));
for i=1:ceil(size(x14,1)*precent)
    Tx=[Tx,x14(xx14(i))];
    Ty=[Ty,y14(xx14(i))];
end



xx15=randperm(size(x15,1));
for i=1:ceil(size(x15,1)*precent)
    Tx=[Tx,x15(xx15(i))];
    Ty=[Ty,y15(xx15(i))];
end


xx16=randperm(size(x16,1));
for i=1:ceil(size(x16,1)*precent)
    Tx=[Tx,x16(xx16(i))];
    Ty=[Ty,y16(xx16(i))];
end


T=[];

for i=1:size(Tx,2)
    T=[T,groundtruth(Tx(i),Ty(i))];
    groundtruth(Tx(i),Ty(i))=0;    
end
% size(groundtruth,1)
labeled=groundtruth;
unlabeled=B-groundtruth;
