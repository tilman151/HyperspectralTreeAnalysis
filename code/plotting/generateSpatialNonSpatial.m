paths = cell(1,14);
names = cell(1,14);
curIndex = 1;
names{curIndex} = 'Spatial        \nIndices';
paths{curIndex} = '..\..\results\RandomForest_100\Indices\SpatialFeatureExtractor_5_Max_Mean_Min_Var\20170316_0054';
curIndex = curIndex + 1;
names{curIndex} = 'Non Spatial\nIndices';
paths{curIndex} = '..\..\results\\RandomForest_100\Indices\20170329_2215';
curIndex = curIndex + 1;
names{curIndex} = 'Spatial        \nMCLDA 5';
paths{curIndex} = '..\..\results\RandomForest_100\MulticlassLda_5\SpatialFeatureExtractor_5_Max_Mean_Min_Var\20170219_1427';
curIndex = curIndex + 1;
names{curIndex} = 'Non Spatial\nMCLDA 5';
paths{curIndex} = '..\..\results\RandomForest_100\MulticlassLda_5\20170301_1251';
curIndex = curIndex + 1;
names{curIndex} = 'Spatial        \nMCLDA 14';
paths{curIndex} = '..\..\results\RandomForest_100\MulticlassLda_14\SpatialFeatureExtractor_5_Max_Mean_Min_Var\20170217_1502';
curIndex = curIndex + 1;
names{curIndex} = 'Non Spatial\nMCLDA 14';
paths{curIndex} = '..\..\results\RandomForest_100\MulticlassLda_14\20170301_0858';
curIndex = curIndex + 1;
names{curIndex} = 'Spatial        \nPCA 5';
paths{curIndex} = '..\..\results\RandomForest_100\PCA_5\SpatialFeatureExtractor_5_Max_Mean_Min_Var\20170219_1524';
curIndex = curIndex + 1;
names{curIndex} = 'Non Spatial\nPCA 5';
paths{curIndex} = '..\..\results\RandomForest_100\PCA_5\20170301_1621';
curIndex = curIndex + 1;
names{curIndex} = 'Spatial        \nPCA 14';
paths{curIndex} = '..\..\results\RandomForest_100\PCA_14\SpatialFeatureExtractor_5_Max_Mean_Min_Var\20170218_1738';
curIndex = curIndex + 1;
names{curIndex} = 'Non Spatial\nPCA 14';
paths{curIndex} = '..\..\results\RandomForest_100\PCA_14\20170307_1111';
curIndex = curIndex + 1;
names{curIndex} = 'Spatial        \nSELD 5';
paths{curIndex} = '..\..\results\RandomForest_100\SELD_20_5\SpatialFeatureExtractor_5_Max_Mean_Min_Var\20170219_1322';
curIndex = curIndex + 1;
names{curIndex} = 'Non Spatial\nSELD 5';
paths{curIndex} = '..\..\results\RandomForest_100\SELD_20_5\20170307_1307';
curIndex = curIndex + 1;
names{curIndex} = 'Spatial        \nSELD 14';
paths{curIndex} = '..\..\results\RandomForest_100\SELD_20_14\SpatialFeatureExtractor_5_Max_Mean_Min_Var\20170219_1212';
curIndex = curIndex + 1;
names{curIndex} = 'Non Spatial\nSELD 14';
paths{curIndex} = '..\..\results\RandomForest_100\SELD_20_14\20170307_1449';
curIndex = curIndex + 1;

accuracies = getAccuraciesForEachFold( paths );

plotBoxPlot(accuracies, names, 'Accuracy', [], 'export/SpatialVsNonSpatial', [1000, 500]);