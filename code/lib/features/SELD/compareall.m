%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             
%           Feature extraction for remote sensing Combining the class
%           discrimination and spatial information
%                             
%           
%       Copyright notes
%       Author: Wenzhi Liao, IPI, Telin, Ghent University, Belgium
%       Date: 25/11/2010
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [araw, alda , alpp, aildlp, anldlp, outraw, outlda, outlpp, outildlp, outnldlp]=compareall(spectraldata, ground, traingrd, nu_features, k,rate,nu_class)
% p = fileparts(mfilename('fullpath'));
% addpath(fullfile(p,'../algorithm'));
dimension=nu_features;% the number of extracted features 

labeled=reshape(ground,size(ground,1)*size(ground,2),1);

% parameters for SVM classifier
class_num=nu_class;% the number of the class
classifier_type='svm';% the classifier used to classification

%%%remove the noisy
spectraldata=double(spectraldata);
spectraldataa=spectraldata;
spectraldata=spectraldata(:,:,1:216);
spectraldata11=spectraldata(:,:,1:148);
spectraldata12=spectraldata(:,:,166:end);
spectraldata=cat(3,spectraldata11,spectraldata12);
spectraldata21=spectraldata(:,:,1:103);
spectraldata22=spectraldata(:,:,113:end);
spectraldata=cat(3,spectraldata21,spectraldata22);
spectraldata=spectraldata(:,:,6:end);

X=reshape(spectraldata,size(spectraldata,1)*size(spectraldata,2),size(spectraldata,3));

%============================
%feature extraction by each method
[tlda,fealda]=lda(X, labeled, dimension);
[tlpp,fealpp]=lpp(X, dimension,k);%%
[tildlp,feaildlp]=ILDLP(X, labeled, dimension, rate, k);
[tnldlp,feanldlp]=SELD(X, labeled, dimension, k);



for i=1:dimension
    Extractedfeatureslda(:,:,i)=reshape(fealda(:,i),size(ground,1),size(ground,2));
    Extractedfeatureslpp(:,:,i)=reshape(fealpp(:,i),size(ground,1),size(ground,2));
    Extractedfeaturesildlp(:,:,i)=reshape(feaildlp(:,i),size(ground,1),size(ground,2));
    Extractedfeaturesnldlp(:,:,i)=reshape(feanldlp(:,i),size(ground,1),size(ground,2));
end


% input data for SVM classification
inputraw=spectraldataa;
inputlda=Extractedfeatureslda;
inputlpp=Extractedfeatureslpp;
inputildlp=Extractedfeaturesildlp;
inputnldlp=Extractedfeaturesnldlp;


% load the trainmask
labels=ground;
mask=traingrd;%train SVM


% do classification 

c=classifier(classifier_type);

craw=train(c,inputraw,labels,mask,[100,0.1]);
outraw=classify(craw,inputraw);

clda=train(c,inputlda,labels,mask,[100,0.1]);
outlda=classify(clda,inputlda);

clpp=train(c,inputlpp,labels,mask,[100,0.1]);
outlpp=classify(clpp,inputlpp);

cildlp=train(c,inputildlp,labels,mask,[100,0.1]);
outildlp=classify(cildlp,inputildlp);

cnldlp=train(c,inputnldlp,labels,mask,[100,0.1]);
outnldlp=classify(cnldlp,inputnldlp);


%%%calculate the total accuracy
araw=sum(outraw==labels)/sum(labels>0);

alda=sum(outlda==labels)/sum(labels>0);

alpp=sum(outlpp==labels)/sum(labels>0);

aildlp=sum(outildlp==labels)/sum(labels>0);

anldlp=sum(outnldlp==labels)/sum(labels>0);


%%% accuracy for each class
for i=1:class_num
     acraw(i)=size(find(outraw==labels & outraw==i & labels==i & mask==0),1)/size(find(labels==i & mask==0),1);
     
     aclda(i)=size(find(outlda==labels & outlda==i & labels==i & mask==0),1)/size(find(labels==i & mask==0),1);
     
     aclpp(i)=size(find(outlpp==labels & outlpp==i & labels==i & mask==0),1)/size(find(labels==i & mask==0),1);
    
     acildlp(i)=size(find(outildlp==labels & outildlp==i & labels==i & mask==0),1)/size(find(labels==i & mask==0),1);
     
     acnldlp(i)=size(find(outnldlp==labels & outnldlp==i & labels==i & mask==0),1)/size(find(labels==i & mask==0),1);
       
end

disp('The classification accuracy for each class: (Raw, LDA, LPP, ILDLP, NLDLP)')
for i=1:class_num
   disp(['The classify accuracy of class ' num2str(i),sprintf(' is: %.2f%',acraw(i))])
    
   disp(['The classify accuracy of class ' num2str(i),sprintf(' is: %.2f%',aclda(i))])
   
   disp(['The classify accuracy of class ' num2str(i),sprintf(' is: %.2f%',aclpp(i))])
 
   disp(['The classify accuracy of class ' num2str(i),sprintf(' is: %.2f%',acildlp(i))])
   
   disp(['The classify accuracy of class ' num2str(i),sprintf(' is: %.2f%',acnldlp(i))])
      
end

disp('The average classification accuracy for each class: (Raw, LDA, LPP, ILDLP, NLDLP)')
sum(acraw)/class_num
sum(aclda)/class_num
sum(aclpp)/class_num
sum(acildlp)/class_num
sum(acnldlp)/class_num



