clear all;
close all;

%Get training images
digits_ds = imageDatastore('training4','IncludeSubfolders',true,'LabelSource','foldernames');

[trainImgs,testImgs] = splitEachLabel(digits_ds,0.6);
numClasses = numel(categories(digits_ds.Labels));

%Create a network by modifying AlexNet
net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(numClasses);
layers(end) = classificationLayer;

%Set training algorithm options
options = trainingOptions('sgdm','InitialLearnRate', 0.001);

t=cputime;
%Perform training
[digitnet,info] = trainNetwork(trainImgs, layers, options);
test4=digitnet;
save test4;
e=cputime-t;
disp(e);
%Use trained network to classify test images
testpreds = classify(digitnet,testImgs);
accuracy = mean(testpreds == testImgs.Labels);
disp(accuracy);