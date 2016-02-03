load('labels_101415.mat')
% positiveInstances = 'D:\Soysal D\Kaggle Competitions\Whale Recognition\imgs_subset_neg
negativeFolder = 'D:\Soysal D\Kaggle Competitions\Whale Recognition\imgs_subset_neg_v5';
trainCascadeObjectDetector('detectorFile_v17.xml', ...
    positiveInstances, negativeFolder, ... 
    'NumCascadeStages',15,'FalseAlarmRate',0.0005,...
    'FeatureType','HOG');