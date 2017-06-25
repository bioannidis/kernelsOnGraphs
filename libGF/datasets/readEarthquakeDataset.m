function [m_spatialAdjacency,m_magnitudesignals] = readEarthquakeDataset
%%
%This function returns the economiSectorSignals

%% initialize
pth = which('readEarthquakeDataset');
folder=[pth(1:end-(length('readEarthquakeDataset')+2))  'EarthquakeDataset\'];

file=strcat(folder,'databaseR.xlsx');
%[num,txt,raw]=xlsread(file);
m_aux=readtable(file);
m_aux.Date=datetime(m_aux.Date,'InputFormat','MM/dd/yyyy');
v_time=datenum(m_aux.Date);
v_long=m_aux.Longitude;
v_lat=m_aux.Latitude;
m_longlat=[v_long,v_lat];
[~,idx]=unique(m_longlat,'rows');
m_longlatRed=m_longlat(idx,:);
v_magnitude=m_aux.Magnitude;
v_long=m_longlat(:,1);
v_lat=m_longlat(:,2);
s_rangelat=max(v_lat)-min(v_lat);
s_rangelong=max(v_long)-min(v_long);
s_finegridscale=0.05;
s_gridSize=s_finegridscale*s_rangelong;
s_steplat=s_rangelat/s_gridSize;
s_steplong=s_rangelong/s_gridSize;
c_grid=cell(round(s_gridSize)+1);
s_countRepeating=1;
m_longlatGrid=m_longlat;
for s_ind=1:size(m_longlat,1)
    s_indlong=round((m_longlat(s_ind,1)-min(v_long))/s_steplong)+1;
    s_indlat=round((m_longlat(s_ind,2)-min(v_lat))/s_steplat)+1;
    v_aux1=c_grid(s_indlong,s_indlat);
    v_aux1=cell2mat(v_aux1);
    if ~isempty(v_aux1)
        s_countRepeating=s_countRepeating+1;
    end
    v_aux1=[v_aux1...
        m_longlat(s_ind,:)];
    c_grid(s_indlong,s_indlat)={v_aux1};
    m_longlatGrid(s_ind,:)=[(s_indlong-1)*s_steplong+min(v_long),(s_indlat-1)*s_steplat+min(v_lat)];
end
%[v_indlong,v_indLat]=find(~cellfun('isempty',c_grid));
[~,idx]=unique(m_longlatGrid,'rows','stable');
m_longlatRed=m_longlatGrid(idx,:);
v_longRed=m_longlatRed(:,1);
v_latRed=m_longlatRed(:,2);

s_mintime=v_time(1);
s_timeperiod=720; %yearly
c_magnitudesignals=cell(size(m_longlatRed,1),round((v_time(end)-s_mintime)/s_timeperiod)+1);

%remove nan values in time
v_ind=find(isnan(v_time));
v_time(v_ind)=[];
m_longlatGrid(v_ind,:)=[];

for s_indtime=1:size(v_time)
 
    s_vertInd=find(ismember(m_longlatRed,m_longlatGrid(s_indtime,:),'rows'));
    s_periodind=floor((v_time(s_indtime)-s_mintime)/s_timeperiod)+1;
    v_aux=c_magnitudesignals(s_vertInd,s_periodind);
    v_aux=cell2mat(v_aux);
    v_aux=[v_aux...
        v_magnitude(s_indtime)];
    c_magnitudesignals(s_vertInd,s_periodind)={v_aux};
end
m_magnitudesignals=cellfun(@mean,c_magnitudesignals);
m_magnitudesignals(find(isnan(m_magnitudesignals)))=0;



s_kNearestNeighbors=7;

s_differentLocationsWithFullData=size(m_longlatRed,1);
m_distances=zeros(s_differentLocationsWithFullData,s_differentLocationsWithFullData);
for s_indLoc1=1:s_differentLocationsWithFullData
	for s_indLoc2=1:s_differentLocationsWithFullData
		[m_distances(s_indLoc1,s_indLoc2),~]=...
			lldistkm(m_longlatRed(s_indLoc1,:)...
			,m_longlatRed(s_indLoc2,:));
	end
end
m_kNearestNeighbors=zeros(s_differentLocationsWithFullData,s_kNearestNeighbors);
m_distances=(m_distances)/5000;
for s_indLoc=1:s_differentLocationsWithFullData
	[~,v_sortNeighbors]=sort(m_distances(s_indLoc,:));
	m_kNearestNeighbors(s_indLoc,:)=v_sortNeighbors(2:s_kNearestNeighbors+1);
end
m_spatialAdjacency=zeros(size(m_distances));

for s_indLoc1=1:s_differentLocationsWithFullData
	for s_indLoc2=1:s_differentLocationsWithFullData
		if(s_indLoc1~=s_indLoc2)
			val=f(m_distances(s_indLoc1,s_indLoc2))/...
				sqrt(sum(f(m_distances(s_indLoc2,m_kNearestNeighbors(s_indLoc2,:)')))...
				*sum(f(m_distances(s_indLoc1,m_kNearestNeighbors(s_indLoc1,:)'))));
			m_spatialAdjacency(s_indLoc1,s_indLoc2)=val;
		end
    end
            s_indLoc1

end

end

function s_value=f(d)
s_value=vpa(exp(-d.^2));
end

function [d1km,d2km]=lldistkm(latlon1,latlon2)
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
