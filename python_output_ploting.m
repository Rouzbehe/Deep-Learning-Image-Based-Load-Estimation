clear all, clc, close all

load arrdata.mat
plotfname='python-SVM-FR_13,20,21,24,26';
testfeatNo=43;
tvp=arr;
plotCorrelation_CI(tvp,plotfname,testfeatNo)
