function [m_spatialAdjacency,m_economicSectorsSignals] = readEarthquakeDataset
%%
%This function returns the economiSectorSignals

%% initialize
pth = which('readEarthquakeDataset');
folder=[pth(1:end-(length('readEarthquakeDataset')+2))  'EarthquakeDataset\'];

file=strcat(folder,'databaseR.xlsx');
[num,txt,raw]=xlsread(file);
m_aux=readtable(file);
m_aux.Date=datetime(m_aux.Date,'InputFormat','MM/dd/yyyy');
m_aux.Date=datenum(m_aux.Date);
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

function s_value=f(d)
s_value=vpa(exp(-d.^2));
end

function [d1km d2km]=lldistkm(latlon1,latlon2)
% format: [d1km d2km]=lldistkm(latlon1,latlon2)
% Distance:
% d1km: distance in km based on Haversine formula
% (Haversine: http://en.wikipedia.org/wiki/Haversine_formula)
% d2km: distance in km based on Pythagoras’ theorem
% (see: http://en.wikipedia.org/wiki/Pythagorean_theorem)
% After:
% http://www.movable-type.co.uk/scripts/latlong.html
%
% --Inputs:
%   latlon1: latlon of origin point [lat lon]
%   latlon2: latlon of destination point [lat lon]
%
% --Outputs:
%   d1km: distance calculated by Haversine formula
%   d2km: distance calculated based on Pythagoran theorem
%
% --Example 1, short distance:
%   latlon1=[-43 172];
%   latlon2=[-44  171];
%   [d1km d2km]=distance(latlon1,latlon2)
%   d1km =
%           137.365669065197 (km)
%   d2km =
%           137.368179013869 (km)
%   %d1km approximately equal to d2km
%
% --Example 2, longer distance:
%   latlon1=[-43 172];
%   latlon2=[20  -108];
%   [d1km d2km]=distance(latlon1,latlon2)
%   d1km =
%           10734.8931427602 (km)
%   d2km =
%           31303.4535270825 (km)
%   d1km is significantly different from d2km (d2km is not able to work
%   for longer distances).
%
% First version: 15 Jan 2012
% Updated: 17 June 2012
%--------------------------------------------------------------------------

radius=6371;
lat1=latlon1(1)*pi/180;
lat2=latlon2(1)*pi/180;
lon1=latlon1(2)*pi/180;
lon2=latlon2(2)*pi/180;
deltaLat=lat2-lat1;
deltaLon=lon2-lon1;
a=sin((deltaLat)/2)^2 + cos(lat1)*cos(lat2) * sin(deltaLon/2)^2;
c=2*atan2(sqrt(a),sqrt(1-a));
d1km=radius*c;    %Haversine distance

x=deltaLon*cos((lat1+lat2)/2);
y=deltaLat;
d2km=radius*sqrt(x*x + y*y); %Pythagoran distance

end
