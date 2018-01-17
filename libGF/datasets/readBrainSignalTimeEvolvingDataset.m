function [m_spatialAdjacency,m_timeAdjacency,t_brainSignalTimeSeries] = readBrainSignalTimeEvolvingDataset
%%
%This function returns the brain data in a tensor so that every slice of
%the tensor correspond to different measured signal
%The adjacency is constructed using partial correlation

%% initialize
pth = which('readBrainSignalTimeEvolvingDataset');
folder=[pth(1:end-(length('readBrainSignalTimeEvolvingDataset')+2))  'BrainDataSet\'];

%%
% Eight sets of measurements were taken on this patient, corresponding to 
% eight different epileptic seizures.  An epileptologist, examining the 
% data for each seizure, identified two key periods relating to the seizure
% i.e., the so-called pre-ictal and ictal periods, and time series data 
% for those periods were extracted.


% ictal period
file=strcat(folder,'sz1_ict.dat');
[t_brainSignalTimeSeries(:,:,1)] = importdata(file);
file=strcat(folder,'sz2_ict.dat');
[t_brainSignalTimeSeries(:,:,2)] = importdata(file);
file=strcat(folder,'sz3_ict.dat');
[t_brainSignalTimeSeries(:,:,3)] = importdata(file);
file=strcat(folder,'sz4_ict.dat');
[t_brainSignalTimeSeries(:,:,4)] = importdata(file);
file=strcat(folder,'sz5_ict.dat');
[t_brainSignalTimeSeries(:,:,5)] = importdata(file);
file=strcat(folder,'sz6_ict.dat');
[t_brainSignalTimeSeries(:,:,6)] = importdata(file);
file=strcat(folder,'sz7_ict.dat');
[t_brainSignalTimeSeries(:,:,7)] = importdata(file);
file=strcat(folder,'sz8_ict.dat');
[t_brainSignalTimeSeries(:,:,8)] = importdata(file);

% pre-ictal period
file=strcat(folder,'sz1_pre.dat');
[t_brainSignalTimeSeries(:,:,9)] = importdata(file);
file=strcat(folder,'sz2_pre.dat');
[t_brainSignalTimeSeries(:,:,10)] = importdata(file);
file=strcat(folder,'sz3_pre.dat');
[t_brainSignalTimeSeries(:,:,11)] = importdata(file);
file=strcat(folder,'sz4_pre.dat');
[t_brainSignalTimeSeries(:,:,12)] = importdata(file);
file=strcat(folder,'sz5_pre.dat');
[t_brainSignalTimeSeries(:,:,13)] = importdata(file);
file=strcat(folder,'sz6_pre.dat');
[t_brainSignalTimeSeries(:,:,14)] = importdata(file);
file=strcat(folder,'sz7_pre.dat');
[t_brainSignalTimeSeries(:,:,15)] = importdata(file);
file=strcat(folder,'sz8_pre.dat');
[t_brainSignalTimeSeries(:,:,16)] = importdata(file);

%the matrices were constructed using sz1_pre.dat time series
%and Yanning linear Svars approch
%import cell array for time 0
file=strcat(folder,'A0_pre.mat');
A0St=importdata(file);
%import cell array for time 0
file=strcat(folder,'A1_pre.mat');
A1St=importdata(file);

m_spatialAdjacency=A0St{1,1};
m_spatialAdjacency=m_spatialAdjacency-diag(diag(m_spatialAdjacency));
m_timeAdjacency=A1St{1,1};
m_spatialAdjacency=m_spatialAdjacency-diag(diag(m_spatialAdjacency));
m_spatialAdjacency(m_spatialAdjacency<0)=0;
m_spatialAdjacency=m_spatialAdjacency+m_spatialAdjacency';

m_timeAdjacency=m_timeAdjacency-diag(diag(m_timeAdjacency));
m_timeAdjacency=m_timeAdjacency+m_timeAdjacency';
%m_spatialAdjacency(m_spatialAdjacency>0)=1;
m_timeAdjacency(m_timeAdjacency>0)=1;
%m_spatialAdjacency=m_adjacency;
 m_spatialAdjacency = t_brainSignalTimeSeries(:,:,9)*t_brainSignalTimeSeries(:,:,9)';
 m_spatialAdjacency=m_spatialAdjacency-diag(diag(m_spatialAdjacency));
 m_spatialAdjacency=m_spatialAdjacency/max(max(m_spatialAdjacency));
 m_spatialAdjacency(m_spatialAdjacency<0)=0;
  m_spatialAdjacency(m_spatialAdjacency<10^-3)=0;
 % m_spatialAdjacency(m_spatialAdjacency<0)=0;

%file=strcat(folder,'m_spatialAdjacencywithpartialCorMethod.mat');
%file=strcat(folder,'m_spatialAdjacencywithCovarianceAndYanningMethod.mat');

%m_spatialAdjacency=importdata(file);
end

