function [m_adjacency,v_meanTemperature,v_altitudes] = readMeanOverTimeTemperatureUSWithAltitudeDataset
%%
%This function returns the temperature data per hour
%The subscript n and o denotes the old and new measurments

%% initialize
pth = which('readTemperatureTimeEvolvingDataset');
folder=[pth(1:end-(length('readTemperatureTimeEvolvingDataset')+2))  'TemperatureTimeEvolvingDataset\'];
file=strcat(folder,'HourlyTemperatureFor2010onlyNumericValues.csv');
%the format of m_X is elevation, lattitude, longtitude, date, time and mean temperature
[m_X] = csvread(file);



s_kNearestNeighbors=7;
m_differentLattitudesLongtitudes=unique(m_X(:,2:3),'rows');
s_differentLocations=size(m_differentLattitudesLongtitudes,1);

%% create time series on vertices
%m_temperaturetimeSeries=zeros(s_totalTime,s_differentLocations);
s_indFullData=1;
for s_indLoc=1:s_differentLocations
	[indecesRow,~]=find(m_X(:,2)==(m_differentLattitudesLongtitudes(s_indLoc,1)));
	%exlude data that are not complete
	if(size(indecesRow,1)==8759)
		m_differentLattitudesLongtitudesWithFullData(s_indFullData,:) =...
		m_differentLattitudesLongtitudes(s_indLoc,:);
		m_temperaturetimeSeries(s_indFullData,:) = m_X(indecesRow,6);
        v_altitudes(s_indFullData)=m_X(indecesRow(1),1);
		s_indFullData=s_indFullData+1;
       
	end
end

v_altitudes=v_altitudes';
%% create adjacency matrix using method from Jordan paper
s_differentLocationsWithFullData=size(m_differentLattitudesLongtitudesWithFullData,1);
m_distances=zeros(s_differentLocationsWithFullData,s_differentLocationsWithFullData);
for s_indLoc1=1:s_differentLocationsWithFullData
	for s_indLoc2=1:s_differentLocationsWithFullData
		[m_distances(s_indLoc1,s_indLoc2),~]=...
			lldistkm(m_differentLattitudesLongtitudesWithFullData(s_indLoc1,:)...
			,m_differentLattitudesLongtitudesWithFullData(s_indLoc2,:));
	end
end

m_kNearestNeighbors=zeros(s_differentLocationsWithFullData,s_kNearestNeighbors);
m_distances=m_distances/(600);
for s_indLoc=1:s_differentLocationsWithFullData
	[~,v_sortNeighbors]=sort(m_distances(s_indLoc,:));
	m_kNearestNeighbors(s_indLoc,:)=v_sortNeighbors(2:s_kNearestNeighbors+1);
end
m_adjacency=zeros(size(m_distances));

for s_indLoc1=1:s_differentLocationsWithFullData
	for s_indLoc2=1:s_differentLocationsWithFullData
		if(s_indLoc1~=s_indLoc2)
			m_adjacency(s_indLoc1,s_indLoc2)=f(m_distances(s_indLoc1,s_indLoc2))/...
				sqrt(sum(f(m_distances(s_indLoc2,m_kNearestNeighbors(s_indLoc2,:)')))...
				*sum(f(m_distances(s_indLoc1,m_kNearestNeighbors(s_indLoc1,:)'))));
			
		end
	end
end
v_meanTemperature=mean(m_temperaturetimeSeries,2);

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
