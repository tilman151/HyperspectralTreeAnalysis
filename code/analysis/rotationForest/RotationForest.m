classdef RotationForest < ExampleClassifier
    %ROTATIONFOREST Summary of this class goes here
    %   Detailed explanation goes here
    
 
    properties
        %number of trees in the ensemble
        numTrees;
        
        %the tree ensemble 
        treeEnsemble;
        
        %additional specifications possible (eg. predicition of class
        %values as input
    end
    
    methods
       
        function obj = RandomForest(numTrees)
             %specify how many trees should be learned
            obj.numTrees = numTrees;
            
        end
        
        function obj = trainOn(obj,trainFeatures,trainLabels)
           [featureList, labelList, ~] = ...
                extractLabeledPixels(trainFeatures, trainLabels);
            trainXOrg=featureList;
         %   numberfeature=size(trainXOrg,2);
        trainYOrg=labelList;
        %numbertrain=length(trainYOrg);
        testYOrg=dnatestlabels;
        numbertest=length(testYOrg);
        
        class=unique(testYOrg);
        numberclass=length(class);
            
            L=obj.numTrees; %% number of ensemble individuals;
for l=1:L
    %%% obtain the new samples by rotation forest %%%
    K=13; 
    ratio=0.75;
    [R_new,R_coeff,trainRFnew,testXRFnew]=RotationFal(trainXOrg, trainYOrg, [], K, ratio);
    
    %%% learn for the new samples by classifiers(learners) %%%
    knn=13; % parameter of classifiers;
    prelabeltest(:,l) = Nearest_Neighbor(trainRFnew, trainYOrg, testXRFnew, knn);
end
%%% voting %%%
numberindex=[];
value=[];
preENCRF=[];
for i=1:numbertest  
    prelabelES=[];
    prelabelES= prelabeltest(i,:); 
    for j=1:numberclass
        index=[];
        index=find(prelabelES==class(j));
        numberindex(i,j)=length(index);
    end
    [value(i,1) indexmax(i,1)]=max(numberindex(i,:));
    preENCRF(i,1)=class(indexmax(i,1));
end
      
%%% compute the accuracy rate of ensemble %%%
accuracyrate=sum(preENCRF==testYOrg)/numbertest;
        

        end
        
        function labels = classifyOn(obj,evalFeatures)
             % Transform input map to vector
            featureList = mapToVec(evalFeatures);
            
            %predict label using the ensemble
            labels = predict(obj.treeEnsemble, featureList);
        end
    end
    
end



% load blfwdbc.mat
% dataX=wdbcsamples;
% dataY=wdbclabels;
% [number numberfeature]=size(dataX);
% numbertrain=100;
% numbertest=number-numbertrain;
% trainXOrg=dataX(1:numbertrain,:);
% trainYOrg=dataY(1:numbertrain,:);
% testXOrg=dataX(1+numbertrain:end,:);
% testYOrg=dataY(1+numbertrain:end,:);
%%% number of classes of samples %%




