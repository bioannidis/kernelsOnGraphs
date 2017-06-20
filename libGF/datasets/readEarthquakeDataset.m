function [m_spatialAdjacency,m_economicSectorsSignals] = readEarthquakeDataset
%%
%This function returns the economiSectorSignals

%% initialize
pth = which('readEarthquakeDataset');
folder=[pth(1:end-(length('readEarthquakeDataset')+2))  'EarthquakeDataset\'];

file=strcat(folder,'IOUse_Before_Redefinitions_PRO_1997-2015_Summary.xlsx');
m_economicSectorsSignals=zeros(71,19);
t_inputOutput=zeros(71,71,19);
for s_year=1997:2015
	m_aux = xlsread(file,sprintf('%g',s_year));
	m_aux(isnan(m_aux))=0;
	m_aux(m_aux<0)=0;
	t_inputOutput(:,:,s_year-1996) =m_aux(1:71,1:71);
	t_adjacency(:,:,s_year-1996)=t_inputOutput(:,:,s_year-1996) -diag(diag(t_inputOutput(:,:,s_year-1996) ));
	t_adjacency(:,:,s_year-1996)=t_adjacency(:,:,s_year-1996)'+t_adjacency(:,:,s_year-1996);
	t_adjacency(:,:,s_year-1996)=t_adjacency(:,:,s_year-1996)/1000000;
	m_economicSectorsSignals(:,s_year-1996)=sum(m_aux(1:71,1:71),1);
end
t_inputOutput((t_inputOutput)<0)=0;
m_spatialAdjacency=mean(t_inputOutput,3);
m_spatialAdjacency=m_spatialAdjacency-diag(diag(m_spatialAdjacency));
m_spatialAdjacency=m_spatialAdjacency'+m_spatialAdjacency;
m_spatialAdjacency=m_spatialAdjacency/1000000;
m_economicSectorsSignals=m_economicSectorsSignals/1000000; %trillions of dollars
m_spatialAdjacency(m_spatialAdjacency<0.01)=0;
ind_non_zero = setdiff(1:71, find(sum(m_spatialAdjacency)==0)); % Delete nodes that are not connected
m_spatialAdjacency=m_spatialAdjacency(ind_non_zero,ind_non_zero);
t_adjacency=t_adjacency(ind_non_zero,ind_non_zero,:);
m_economicSectorsSignals=m_economicSectorsSignals(ind_non_zero,:);
% discard first economicSectorSignal and last adjacency so that the data are not
% used for reconstruction as well as topology generation.
t_adjacency=t_adjacency(:,:,1:end-1);
m_economicSectorsSignals=m_economicSectorsSignals(:,2:end);

end

