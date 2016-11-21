classdef (Abstract) Classifier
    
   methods (Abstract)
       obj = trainOn(obj,trainFeatures,trainLabels);
       labels = classifyOn(obj,evalFeatures);
   end
    
end

