function [m_spatialAdjacency,m_economicSectorsSignals] = readEconomicSectorSignalTimeEvolvingExtendedDataset
%%
%This function returns the economiSectorSignals

%% initialize
pth = which('readEconomicSectorSignalTimeEvolvingExtendedDataset');
folder=[pth(1:end-(length('readEconomicSectorSignalTimeEvolvingExtendedDataset')+2))  'EconomicSectorDataset\ExtendedGraphDataset\'];

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
% % % Santiago
% % % Load Data
% % 
% % load('io_data_2008_2011.mat');
% % 
% % % Generate some useful global variables
% % 
% % Generate the shift
% % 
% % S = (A2008+A2009+A2010)/3; % Average of three years
% % 
% % S = (S + S')/2; % Symmetrize
% % 
% % S(66,67) = 0;S(67,66) = 0; % Delete edge between market and added value
% % 
% % S = S/1000000; x2011 = x2011/1000000; % Work in trillions
% % 
% % S(S<0.01)=0; % Threshold
% % 
% % ind_non_zero = setdiff(1:67, find(sum(S)==0)); % Delete nodes that are not connected
% % 
% % S = S(ind_non_zero, ind_non_zero); % Update shift
% % 
% % x2011 = x2011(ind_non_zero); % Update signal
%% Columns picked from the corresponding table

% % Farms
% % Forestry, fishing, and related activities
% % Oil and gas extraction
% % Mining, except oil and gas
% % Support activities for mining
% % Utilities
% % Construction
% % Wood products
% % Nonmetallic mineral products
% % Primary metals
% % Fabricated metal products
% % Machinery
% % Computer and electronic products
% % Electrical equipment, appliances, and components
% % Motor vehicles, bodies and trailers, and parts
% % Other transportation equipment
% % Furniture and related products
% % Miscellaneous manufacturing
% % Food and beverage and tobacco products
% % Textile mills and textile product mills
% % Apparel and leather and allied products
% % Paper products
% % Printing and related support activities
% % Petroleum and coal products
% % Chemical products
% % Plastics and rubber products
% % Wholesale trade
% % Retail trade
% % Air transportation
% % Rail transportation
% % Water transportation
% % Truck transportation
% % Transit and ground passenger transportation
% % Pipeline transportation
% % Other transportation and support activities
% % Warehousing and storage
% % Publishing industries (includes software)
% % Motion picture and sound recording industries
% % Broadcasting and telecommunications
% % Information and data processing services
% % Federal Reserve banks, credit intermediation, and related activities
% % Securities, commodity contracts, and investments
% % Insurance carriers and related activities
% % Funds, trusts, and other financial vehicles
% % Real estate
% % Rental and leasing services and lessors of intangible assets
% % Legal services                                                                  
% % Computer systems design and related services                                    
% % Miscellaneous professional, scientific, and technical services
% % Management of companies and enterprises
% % Administrative and support services
% % Waste management and remediation services
% % Educational services
% % Ambulatory health care services
% % Hospitals and nursing and residential care facilities
% % Social assistance
% % Performing arts, spectator sports, museums, and related activities
% % Amusements, gambling, and recreation industries
% % Accommodation
% % Food services and drinking places
% % Other services, except government
% % Federal general government
% % Federal government enterprises
% % State and local general government
% % State and local government enterprises
