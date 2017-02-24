function [w,o,time_secs]=svmlin(cmdline,X,Y,w)
% Matlab Interface to SVMlin
% Usage: [w,o]=svmlin(cmdline,X,Y,w)
% 
% Inputs:
%
% cmdline: a string containing command line options
%          e.g. cmdline='-A 1 -W 10.0' where the options are,  
%          -A algorithm : set algorithm (default 1)
%                         0 -- Regularized Least Squares Classification (RLSC)
%                         1 -- SVM (L2-SVM-MFN) (default choice)
%                         2 -- Multi-switch Transductive SVM (using L2-SVM-MFN)
%                         3 -- Deterministic Annealing Semi-supervised VM (using L2-SVM-MFN)
%          -W regularization parameter lambda 
%          -U regularization parameter lambda_u (see notes below)
%          -S maximum number of switches in TSVM (default 10000)
%          -R positive class fraction of unlabeled data  (default 0.5)
%          -Cp relative cost for positive examples (only available with -A 1)
%          -Cn relative cost for negative examples (only available with -A 1)
% X : data matrix (in sparse format) with examples as rows and features as columns  
% Y : label vector (same length as number of examples)
% w : (test mode) A trained weight vector is given for prediction on X.
%      In test mode, if Y is given then accuracy of prediction is printed
%      out. If Y is not available, use Y=[]
%
% Outputs: 
% w: trained weight vector
% o: (soft) predictions of w on X
%
% Example Usage:
%
% Training: 
%   For training a TSVM with maximum switching heuristic (10000
%   switches allowed per iteration), lambda=0.001, lambda_u=1 and positive
%   class ratio 0.2 (fraction of unlabeled data expected to be positive)
%           [w,o]=svmlin('-A 2 -W 0.001 -U 1 -R 0.2 -S 100000'); 
%
% Prediction: 
%           [w,o]=svmlin([],X,[],w); % the output and input w vectors are
%                                    % the same in the prediction mode. 
%                                    % The soft predictions are saved
%                                    % in the vector o.
% Evaluation: 
%           [w,o]=svmlin([],X,Y,w); % outputs accuracy too comparing the 
%                                   predictions o with the true labels Y.
%
%
% Notes: In this code, we need to take a transpose of X to convert to 
%        C compressed row format from the matlab compressed column format.
%        Addionally, a vector of ones is appended as an additional column
%        of X to incorporate the bias feature. Some checks are made to
%        ensure that the mex file runs smoothly. These modifications cause
%        this code to run slower than the C code (taking away loading time).
%        However, the time reported is the time the mex code takes which
%        should be very similar what one would get thru the C code.
%
% Author: Vikas Sindhwani (vikass@cs.uchicago.edu)
%         SVMlin (v1.0) August 2006
%
%
if ~(issparse(X))
    error('data matrix needs to be sparse. Use X=sparse(X)');
end

param=[1.0 1.0 1.0 10000 -1 1.0 1.0]';
if nargin<=3 % training
R=cmdline;
% param=[-A -W -U -S -R -Cp -Cn]

while(not(isempty(R)))
   [T,R]=strtok(R);
   switch T
       case '-A'
         [T,R]=strtok(R); param(1)=str2num(T);
       case '-W'
         [T,R]=strtok(R); param(2)=str2num(T);
       case '-U'
         [T,R]=strtok(R); param(3)=str2num(T);
       case '-S'
         [T,R]=strtok(R); param(4)=str2num(T);
       case '-R'
         [T,R]=strtok(R); param(5)=str2num(T);
       case '-Cp' 
         [T,R]=strtok(R); param(6)=str2num(T); 
       case '-Cn'
         [T,R]=strtok(R); param(7)=str2num(T);
       otherwise
           error('Unrecognized option.');
   end
end
   if(size(X,1)~=length(Y))
       error('number of examples and length of Y do not match.');
   end
else
    if(length(Y)~=0 & size(X,1)~=length(Y))
       error('number of examples and length of Y do not match.');
    end 
   if (length(w)~=size(X,2)+1)
       error('number of features and length of w do not match.');
    end 
end  
start=cputime;
if nargin<=3 % training mode
    if (param(1)<2) % supervised
        L=find(Y);
        X=[X(L,:) ones(length(L),1)];
        Y=Y(L);
        C=ones(length(Y),1);
        C(find(Y>0))=param(6);
        C(find(Y<0))=param(7);
    else
        X=[X ones(size(X,1),1)];  
        C=ones(length(Y),1);
    end 
       [w,o]=svmlin_mex(param,X',Y,C);
else % test mode
       X=[X ones(size(X,1),1)]; 
       o=svmlin_mex(param,X',Y,[],w);
end
time_secs=cputime-start;
%fprintf(1,'time taken: %d secs\n', time_secs);
