function [m_graphFunction] = readGeneNetworkDataset
pth = which('readGeneNetworkDataset');
folder=[pth(1:end-(length('readGeneNetworkDataset')+2))  'GeneNetworkDataset\'];
file=strcat(folder,'giExpGtimpute_nooutliers.mat');
load(file)
m_graphFunction=X;
