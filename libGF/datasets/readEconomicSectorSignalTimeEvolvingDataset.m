function [m_spatialAdjacency,m_economicSectorsSignals] = readEconomicSectorSignalTimeEvolvingDataset
%%
%This function returns the economiSectorSignals

%% initialize
pth = which('readEconomicSectorSignalTimeEvolvingDataset');
folder=[pth(1:end-(length('readEconomicSectorSignalTimeEvolvingDataset')+2))  'EconomicSectorDataset\'];

file=strcat(folder,'theUseofCommoditiesbyIndustries2014.csv');

[m_spatialAdjacency] = csvread(file);
file=strcat(folder,'grossOutputQtredited.csv');
m_spatialAdjacency=m_spatialAdjacency+m_spatialAdjacency';
m_spatialAdjacency=m_spatialAdjacency-diag(diag(m_spatialAdjacency));
m_spatialAdjacency=m_spatialAdjacency/1000; %trilions of dollars
m_spatialAdjacency(m_spatialAdjacency<1)=0;
[m_economicSectorsSignals] = csvread(file);
m_economicSectorsSignals=m_economicSectorsSignals/1000; %trillions of dollars
end

