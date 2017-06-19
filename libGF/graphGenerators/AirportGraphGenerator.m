classdef AirportGraphGenerator < GraphGenerator
    
    properties % required by parent classes
        c_parsToPrint  = {'ch_name','s_numberOfVertices'};
        c_stringToPrint  = {'',''};
        c_patternToPrint = {'%s%s','%s%d vertices'};
    end
    
    
    properties(Constant)
        ch_name = 'Airport';
        THRESHOLD = 80;       % delay more than 80min is outlier
    end
    
    properties
        ch_filepath = {'./libGF/datasets/AirportsDataset/July2015.csv','./libGF/datasets/AirportsDataset/August2015.csv','./libGF/datasets/AirportsDataset/September2015.csv'};   % path to data file
        airportTable = [];       % table that contains raw data
        delayTable;         % a table that each entry contains the average 
                            % delay of one day and one airport                           
        v_listOfAirport;    % valid airports ID
        v_listOfDay;        % days involved
        v_numberOfFlightsPerAirport;
        c_relevantVariableName = {'ORIGIN_AIRPORT_ID', ...
            'DEST_AIRPORT_ID', 'DEP_DELAY', 'ARR_DELAY'};   % variables
                            % that we care about
    end
    
    methods
        % constructor
        function obj = AirportGraphGenerator(varargin)
            obj@GraphGenerator(varargin{:});
        end
        
        % realization: create graph
        function obj = processData(obj) 
            for iPath = 1 : length(obj.ch_filepath)
                % read data into a table
                obj.airportTable = AirportGraphGenerator.importfile(obj.ch_filepath{iPath});
                %% remove all NaN entries
                obj = obj.removeNanRows();
                
                %% average all fights in the same day and the same date
                [depDelayMatrix, arrDelayMatrix, listOfAirport, adjacency, listOfNumberOfFlights] = obj.getDelayTable();
                %% create adjacency matrix
                %listOfDay = obj.v_listOfDay;
                %adjacency = obj.createAdjacencyMatrix(listOfAirport, listOfDay);
                %%
                depDelay{iPath} = depDelayMatrix;
                arrDelay{iPath} = arrDelayMatrix;
                airportList{iPath} = listOfAirport;
                adj{iPath} = adjacency;
				listOfNumOfFlights{iPath}=listOfNumberOfFlights;
            end
            %%
            delayData.year = '2015';
            delayData.month = {'Jul','Aug','Sep'};
            delayData.depDelay = depDelay;
            delayData.arrDelay = arrDelay;
            delayData.airportList = airportList;
            delayData.adjacency = adj;
			delayData.listOfNumOfFlights=listOfNumOfFlights;
            save('delaydata2015SP.mat', 'delayData')
            
        end
        
        function graph = realization(obj)
            graph = Graph('m_adjacency', obj.adjacency);
        end
    end
    
    methods  % getter and setter
        function listOfAirport = get.v_listOfAirport(obj)
            if isempty(obj.v_listOfAirport)
                airportIDList = obj.airportTable.ORIGIN_AIRPORT_ID;
                listOfAirport = sort(unique(airportIDList));  % ascend
            else
                listOfAirport = obj.v_listOfAirport;
			end
        end
        
        function listOfDay = get.v_listOfDay(obj)
            if isempty(obj.v_listOfDay)
                dayList = obj.airportTable.FL_DATE;
                listOfDay = sort(unique(dayList));            % ascend
            else
                listOfDay = obj.v_listOfDay;
            end
        end
    end
    
    methods
        function [depDelayMatrix, arrDelayMatrix, listOfAirport, adjacency,listOfNumberOfFlights] = getDelayTable(obj)
            % get the two tables from raw data
            % depDelayTable:
            %       each entry contains the average departure delay of 
            %       that day and that airport
            % depDelayTable:
            %       each entry contains the average arrival delay of 
            %       that day and that airport
            listOfDay = obj.v_listOfDay;
            listOfAirport = obj.v_listOfAirport;
            
            depDelayMatrix = zeros(length(listOfAirport), length(listOfDay));
            arrDelayMatrix = zeros(length(listOfAirport), length(listOfDay));
            adjacency = zeros(length(listOfAirport), length(listOfAirport), length(listOfDay));
            
            %% Online algorithm
%             depFlight = zeros(length(listOfAirport), length(listOfDay));
%             arrFlight = zeros(length(listOfAirport), length(listOfDay));
%             
%             for i = 1:size(obj.airportTable,1)
%                 entry = obj.airportTable(i,:);
%                 day = find( entry.FL_DATE == listOfDay );
%                 origin = find( entry.ORIGIN_AIRPORT_ID == listOfAirport );
%                 dest = find( entry.DEST_AIRPORT_ID ==  listOfAirport );
%                 depD = entry.DEP_DELAY;
%                 arrD = entry.ARR_DELAY;
%                 
%                 if depD > obj.THRESHOLD || arrD > obj.THRESHOLD
%                     continue;
%                 end
%                 
%                 depDelayMatrix(origin, day) = depDelayMatrix(origin, day) + depD;
%                 arrDelayMatrix(dest, day) = arrDelayMatrix(dest, day) + arrD;
%                 
%                 depFlight(origin, day) = depFlight(origin, day) + 1;
%                 arrFlight(dest, day) = arrFlight(dest, day) + 1;
%                 
%                 adjacency(origin, dest, day) = adjacency(origin, dest, day) + 1;
%                 
%                 if mod(i, 10000) == 0
%                     fprintf('Iteration %d\n',i)
%                 end
%             end
%             
%             depFlight( depFlight == 0 ) = eps;
%             arrFlight( arrFlight == 0 ) = eps;
%             
%             depDelayMatrix = depDelayMatrix ./ depFlight;
%             arrDelayMatrix = arrDelayMatrix ./ arrFlight;
            %
            %% vector form code
			obj.v_numberOfFlightsPerAirport=zeros(size(listOfAirport));
            for day = 1:length(listOfDay)
                % create a sub-table that contains rows of that day
                dayTable = obj.airportTable( obj.airportTable.FL_DATE == listOfDay(day), : );
                for airport = 1:length(listOfAirport)
                    % create a sub-table that contains rows of that airport
                    orgDayTable = dayTable( dayTable.ORIGIN_AIRPORT_ID == ...
                        listOfAirport(airport), :);
                    dstDayTable = dayTable( dayTable.DEST_AIRPORT_ID == ...
                        listOfAirport(airport), :);
                    obj.v_numberOfFlightsPerAirport(airport)=obj.v_numberOfFlightsPerAirport(airport)+size(orgDayTable,1);
                    % find all the outliers
                    depOutlierIndex = abs(orgDayTable.DEP_DELAY) > obj.THRESHOLD;
                    arrOutlierIndex = abs(dstDayTable.DEP_DELAY) > obj.THRESHOLD;
                    
                    % update delay matrix
                    depDelayMatrix(airport, day) = mean(orgDayTable.DEP_DELAY(~depOutlierIndex));
                    arrDelayMatrix(airport, day) = mean(dstDayTable.ARR_DELAY(~arrOutlierIndex));
                    
                    % update adjacency matrix
                    for airport2 = 1:length(listOfAirport)
                        if airport2 == airport
                            continue;
                        end
                        
                        adjacency(airport, airport2, day) = sum( orgDayTable.DEST_AIRPORT_ID == listOfAirport(airport2) );
                    end
                end
                fprintf('Day %d finished\n', day);
            end
            
            % remove airports that contains NaN
            mask = isnan(depDelayMatrix) | isnan(arrDelayMatrix);
            airportIndex = ~(sum(mask, 2) >= 1);
            depDelayMatrix = depDelayMatrix(airportIndex,:);
            arrDelayMatrix = arrDelayMatrix(airportIndex,:);
            listOfAirport = listOfAirport(airportIndex);
			listOfNumberOfFlights=obj.v_numberOfFlightsPerAirport(airportIndex);
            adjacency = adjacency(airportIndex, airportIndex, :);
        end
        
        function A = createAdjacencyMatrix(obj, listOfAirport, listOfDay)
            A = zeros(length(listOfAirport), length(listOfAirport), length(listOfDay));
            for slice = 1 : length(listOfDay)
                day = listOfDay(slice);
                subDayTab = obj.airportTable( obj.airportTable.FL_DATE == day,: );
                for row = 1 : length(listOfAirport)
                    airportOrgID = listOfAirport(row);
                    subOrgTab = subDayTab( subDayTab.ORIGIN_AIRPORT_ID == airportOrgID,: );
                    for col = 1 : length(listOfAirport)
                        airportDstID = listOfAirport(col);
                        %subOrgDstTab = subOrgTab(obj.airportTable.DEST_AIRPORT_ID == airportOrgID);
                        A(row,col,slice) = sum(subOrgTab.DEST_AIRPORT_ID == airportDstID);
                    end
                end
                %A(:,:,slice) = A(:,:,slice) + A(:,:,slice)';
            end
        end
        
        function obj = removeNanRows(obj)
            % remove all nan rows that contains in obj.airportTable
            rows = false( size(obj.airportTable,1),1 );
            for col = 1:length(obj.c_relevantVariableName)
                varCol = obj.airportTable.(obj.c_relevantVariableName{col});
                rows = rows | isnan(varCol);
            end
            obj.airportTable(rows, :) = [];     % delete nan rows
        end
        
        function T = readDataFile(obj)
            % READDATAFILE read airport data from csv/xlsx file
            %   Data file should be in the format specified by readtable function.
            % Since data can contain invalid items, NaN rows must be removed.
            filename = obj.ch_filepath;
            airportTable = readtable(filename);
            nanIndex = zeros(size(airportTable,1),1);
            for column = 1 : length(obj.relevantVariableName)
                nanIndex = nanIndex | ...
                    isnan(airportTable.(obj.relevantVariableName{column}));
            end
            airportTable(nanIndex,:) = [];  % delete relevant nan rows
            T = airportTable;
        end
    end
    
    methods(Static)
       
        function A = createAdjacencyMatrixDepreciated(T)
            % create Laplacian matrix from table T
            % T is a table with each row denoting flight and delays
            % Node of graph is all the aiports
            % edges are # of flights between any two airports
            vertex = AirportGraphGenerator.getVertex(T);
            A = zeros(length(vertex));
            
            ROWS = size(T,1);
            for r = 1 : ROWS
                origin = T.ORIGIN_AIRPORT_ID(r);
                dest = T.DEST_AIRPORT_ID(r);
                
                orgIndex = (vertex == origin);
                desIndex = (vertex == dest);
                A(orgIndex,desIndex) = A(orgIndex,desIndex) + 1;
            end
            A = A + A';
            A = double(A>0);
        end
        
        function VERTEX = getVertex(table)
            originAirport = unique(table.ORIGIN_AIRPORT_ID); % distinct
            destAirport = unique(table.DEST_AIRPORT_ID);
            VERTEX = union(originAirport, destAirport);
		end
		
		function [ m_training_delay, m_test_delay, m_training_adj , m_test_adj ] = getTwoMonths(s_selectedAirportNum,s_departureDelay,str_datasetName1,str_datasetName2)
            % s_selectedAirportNum: number of the most crowded airports
            % that will be returned.
            %
            % s_departure_delay:   1: departure delay
            %                      0: arrival delay
            
            folder = 'libGF/datasets/AirportsDataset/';
            %% load dataset
            load([folder str_datasetName1])
            depDelaySep2014 = delayData.depDelay{3};
            arrDelaySep2014 = delayData.arrDelay{3};
            airportListSep2014 = delayData.airportList{3};
            adjSep2014 = delayData.adjacency{3};                           
            
            %%            
            load([folder str_datasetName2])
            depDelaySep2015 = delayData.depDelay{3};
            arrDelaySep2015 = delayData.arrDelay{3};
            airportListSep2015 = delayData.airportList{3};
            adjSep2015 = delayData.adjacency{3};
            
            %% Sort data in the same order
            % 1. Find common airports
            [v_commonAirports,v_inds2014,v_inds2015] = intersect( airportListSep2014 , airportListSep2015 );
            %nn = norm(  airportListSep2014(v_inds2014) - airportListSep2015(v_inds2015))
                                 
            % 2. Sort data
            depDelaySep2014 = depDelaySep2014(v_inds2014,:);
            arrDelaySep2014 = arrDelaySep2014(v_inds2014,:);
            adjSep2014 = adjSep2014(v_inds2014,v_inds2014,:);
            
            depDelaySep2015 = depDelaySep2015(v_inds2015,:);
            arrDelaySep2015 = arrDelaySep2015(v_inds2015,:);
            adjSep2015 = adjSep2015(v_inds2015,v_inds2015,:);
            
            % check
            %s = [sum(adjSep2014(1:100,1:10,:),3) sum(adjSep2015(1:100,1:10,:),3)]
            
            
           
            
			%% select busiest airports
			A = sum(adjSep2014,3);			
			[~,v_inds] = sort(sum(A,2),'descend');
			v_inds = v_inds(1:s_selectedAirportNum); % indices of the most crowded airports
			
            if s_departureDelay            
                m_training_delay = depDelaySep2014(v_inds,:);
                m_test_delay = depDelaySep2015(v_inds,:);              
            else                
                m_training_delay = arrDelaySep2014(v_inds,:);
                m_test_delay = arrDelaySep2015(v_inds,:);                
            end
            m_training_adj = adjSep2014(v_inds,v_inds,:);
            m_test_adj = adjSep2015(v_inds,v_inds,:);   
            
            %m_cov = cov([m_training_delay m_test_delay]')
            
            
		end
		
		function [ m_training_delay, m_test_delay, m_training_adj , m_test_adj ] = getSixMonths(s_selectedAirportNum,s_departureDelay,str_datasetName1,str_datasetName2)
            % s_selectedAirportNum: number of the most crowded airports
            % that will be returned.
            %
            % s_departure_delay:   1: departure delay
            %                      0: arrival delay
            	
			[comDepCell, comArrCell, comAdjCell] = AirportGraphGenerator.get6monthData(str_datasetName1,str_datasetName2);
			
			
			if s_departureDelay
				comCell = comDepCell;
			else
				comCell = comArrCell;
			end
			m_training_delay = [];
			m_training_adj = zeros(size(comAdjCell{1},1),size(comAdjCell{1},2));
			for k = 1:length(comCell)-1
				m_training_delay = [m_training_delay comCell{k}];
				m_training_adj = m_training_adj + sum(comAdjCell{k},3);
			end
			m_test_delay = comCell{end};
			m_test_adj = sum(comAdjCell{end},3);
			
            
            
			%% select busiest airports
			A = m_training_adj;
			[~,v_inds] = sort(sum(A,2),'descend');
			v_inds = v_inds(1:s_selectedAirportNum); % indices of the most crowded airports
			
            
			m_training_delay = m_training_delay(v_inds,:);
			m_test_delay = m_test_delay(v_inds,:);
            
            m_training_adj = m_training_adj(v_inds,v_inds,:);
            m_test_adj = m_test_adj(v_inds,v_inds,:);   
            
            %m_cov = cov([m_training_delay m_test_delay]')
            
            
		end
		
        function [comDepCell, comArrCell, comAdjCell] = get6monthData(str_datasetName1,str_datasetName2)
			folder = 'libGF/datasets/AirportsDataset/';
			
             %% load dataset
            load([folder str_datasetName1])
            depDelay2014 = delayData.depDelay;
            arrDelay2014 = delayData.arrDelay;
            airportList2014 = delayData.airportList;
            adj2014 = delayData.adjacency;          
            %%            
            load([folder str_datasetName2])
            depDelay2015 = delayData.depDelay;
            arrDelay2015 = delayData.arrDelay;
            airportList2015 = delayData.airportList;
            adj2015 = delayData.adjacency;
            
            
            %%
            airportListCell = [airportList2014, airportList2015];
            depDelayCell = [depDelay2014, depDelay2015];
            arrDelayCell = [arrDelay2014, arrDelay2015];
            adjCell = [adj2014, adj2015];
            
            %% find common airports and their index
            [~, indexCell]  = AirportGraphGenerator.getCommonIndex( airportListCell );
            
            %% get data
            for i = 1 : length(indexCell)
                comDepCell{i} = depDelayCell{i}(indexCell{i},:);
                comArrCell{i} = arrDelayCell{i}(indexCell{i},:);
                comAdjCell{i} = adjCell{i}(indexCell{i}, indexCell{i}, :);
            end
        end
        
         function [commonList, indexCell]  = getCommonIndex( listCell )
            commonList = listCell{1};
            for i = 2 : length(listCell)
                commonList = intersect(commonList, listCell{i});
            end
            
            for i = 1 : length(listCell)
                for k = 1 : length(commonList)
                    index(k) = find( listCell{i} == commonList(k) );
                end
                indexCell{i} = index;
            end
        end
        
	
	
		

		
        function Jul2014 = importfile(filename, startRow, endRow)
            %IMPORTFILE Import numeric data from a text file as a matrix.
            %   JUL2014 = IMPORTFILE(FILENAME) Reads data from text file FILENAME for
            %   the default selection.
            %
            %   JUL2014 = IMPORTFILE(FILENAME, STARTROW, ENDROW) Reads data from rows
            %   STARTROW through ENDROW of text file FILENAME.
            %
            % Example:
            %   Jul2014 = importfile('Jul2014.csv', 2, 520881);
            %
            %    See also TEXTSCAN.
            
            % Auto-generated by MATLAB on 2016/05/20 12:29:23
            
            %% Initialize variables.
            delimiter = ',';
            if nargin<=2
                startRow = 2;
                endRow = inf;
            end
            
            %% Format string for each line of text:
            %   column1: datetimes (%{MM/dd/yyyy}D)
            %	column2: double (%f)
            %   column3: double (%f)
            %	column4: double (%f)
            %   column5: text (%s)
            %	column6: double (%f)
            %   column7: text (%s)
            %	column8: double (%f)
            %   column9: double (%f)
            %	column10: text (%s)
            %   column11: text (%s)
            %	column12: text (%s)
            %   column13: text (%s)
            %	column14: text (%s)
            % For more information, see the TEXTSCAN documentation.
            formatSpec = '%{MM/dd/yyyy}D%f%f%f%s%f%s%f%f%s%s%s%s%s%[^\n\r]';
            
            %% Open the text file.
			
            fileID = fopen(filename,'r');
            
            %% Read columns of data according to format string.
            % This call is based on the structure of the file used to generate this
            % code. If an error occurs for a different file, try regenerating the code
            % from the Import Tool.
            dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
            for block=2:length(startRow)
                frewind(fileID);
                dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
                for col=1:length(dataArray)
                    dataArray{col} = [dataArray{col};dataArrayBlock{col}];
                end
            end
            
            %% Close the text file.
            fclose(fileID);
            
            %% Post processing for unimportable data.
            % No unimportable data rules were applied during the import, so no post
            % processing code is included. To generate code which works for
            % unimportable data, select unimportable cells in a file and regenerate the
            % script.
            
            %% Create output variable
            Jul2014 = table(dataArray{1:end-1}, 'VariableNames', {'FL_DATE','AIRLINE_ID','FL_NUM','ORIGIN_AIRPORT_ID','ORIGIN','DEST_AIRPORT_ID','DEST','DEP_DELAY','ARR_DELAY','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY'});
            
            % For code requiring serial dates (datenum) instead of datetime, uncomment
            % the following line(s) below to return the imported dates as datenum(s).
            
            % Jul2014.FL_DATE=datenum(Jul2014.FL_DATE);


        
        end
    end
    
end