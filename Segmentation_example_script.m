% Implementation of Semantic Segmentation
%
% written by Oguzhan Ulucan, Izmir University of Economics, January 2020
% e-mail: oguzhan.ulucan.iz@gmail.com
%
% This work was published in
%   "A Large-Scale Dataset for Fish Segmentation and Classification"
%   Oguzhan Ulucan, Diclehan Karakaya and Mehmet Turkan
%   ASYU 2020.

% If you use this dataset in your work, please consider to cite:

% @inproceedings{ulucan2020large,
%   title={A Large-Scale Dataset for Fish Segmentation and Classification},
%   author={Ulucan, Oguzhan and Karakaya, Diclehan and Turkan, Mehmet},
%   booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
%   pages={1--5},
%   year={2020},
%   organization={IEEE}
% }

% O.Ulucan , D.Karakaya and M.Turkan.(2020) 
% A large-scale dataset for fish segmentation and classification.
% In Conf. Innovations Intell. Syst. Appli. (ASYU)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% By following the steps below, segmentation of any class in the dataset is possible via semantic segmentation. 
% The code was written for MATLAB enviroment.
% After downloading the "Fish_Dataset", tasks such as, segmentation, feature extraction or classification can be carried out.
% In this text, the explanation of the segmentation process is provided.

clear all, clc, close all

imageDir = ' '; % The directory of the images which would like be used to train the segmention network should be in here.
labelDir = ' '; % The directory of the pair-waise ground-truths (GT) of images should be in here.
imds = imageDatastore(imageDir);

%%

classNames = [" Fish type ", "background"];  % Fish type should be specified in here to label the segmented images. 
labelIDs   = [1 0];
pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);

%% Create a semantic segmentation network
% This part can be modified according to your own work.

numFilters = 64; 
filterSize = 3;
numClasses = 2;
layers = [
    imageInputLayer([445 590 3])
    convolution2dLayer(filterSize, numFilters, 'Padding',[1 1])
    reluLayer()
    maxPooling2dLayer(2,'Stride',1)
    convolution2dLayer(filterSize, numFilters, 'Padding',[1 1])
    reluLayer()
    convolution2dLayer(filterSize, numFilters, 'Padding',[1 1])
    reluLayer()
    convolution2dLayer(filterSize, numFilters, 'Padding',[1 1])
    reluLayer()
    transposedConv2dLayer(4, numFilters, 'Stride', 1, 'Cropping', 1);
    convolution2dLayer(1, numClasses);
    softmaxLayer()
    pixelClassificationLayer()
    ]


%% Train the Network
% This part can be modified according to your own work.

opts = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',10, ...
    'MiniBatchSize',8);

%%
trainingData = pixelLabelImageDatastore(imds,pxds);
network = trainNetwork(trainingData,layers,opts);


%% Improvement

tbl = countEachLabel(trainingData);
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
classWeights = 1./frequency;
layers(end)= pixelClassificationLayer('ClassNames',tbl.Name,'ClassWeights',classWeights);
improved_network = trainNetwork(trainingData,layers,opts);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% After training process is completed, the segmentation of the fish can be carried out.

Fish_Dir = ' '; 	  % The directory of the images which would like to be segmented should be in here.
Segmented_Fish_Dir = ' '; % The segmented images will be saved in this directory.

im_Set = imageSet(Fish_Dir);

for k = 1 : im_Set.Count %number of images
  testImage = imread(im_Set.ImageLocation{k}); 
  C = semanticseg(testImage,improved_network);
  B = labeloverlay(testImage,C);
  baseFileName = sprintf('%04d.png', k); 	%label every image in order
  fullFileName = fullfile(Segmented_Fish_Dir, baseFileName);
  imwrite(B, fullFileName);
end
