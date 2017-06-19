
classdef ReadCrimeDataset < ReadDataset
	
	
	properties(Constant)
		ch_TxtName = './libGF/datasets/CrimeDataset/CommViolPredUnnormalizedData.txt';
		
	end
	
	
	
	methods(Static)
		
		function [ m_test, v_range ] = getTestTables			
			%
			%
			% m_test        : 
			%               
			%               
		    %               
			% 
			%
			% v_range       : 2x1 vector contains the min and max rating 
			%               available in the dataset
			%
				D = textread(ReadCrimeDataset.ch_TxtName, '%s', 'delimiter', ',');
				FID = fopen(FileName, 'TISdirs');

				D1= textscan(FID,'%s','delimiter','\n');
				s_SetNum = 5;
			s_UsersNum = 943; 
			s_MoviesNum = 1682;			
			m_test=zeros(s_UsersNum,s_MoviesNum,s_SetNum);
			v_range=zeros(2,1);
			
			for s_setInd = 1:s_SetNum
				
				ch_file = sprintf('%s%d%s',[ReadMovieLensDataset.ch_folderName 'u'],s_setInd,'_test.txt');
				
				
				TestData=table2array(D); % columns contain user id | item id | rating | timestamp.
				%                          The time stamps are unix seconds since 1/1/1970 UTC
				
				for icount=1:size(TestData,1)
					m_test(TestData(icount,1),TestData(icount,2),s_setInd)=TestData(icount,3);
				end
			
			end		
			%m_test=(m_test-1)/4;
			m_test(m_test==0)=NaN;
			v_range(1)=min(min(min(m_test)));
			v_range(2)=max(max(max(m_test)));
			% NaN and range [0,1]
			
			%save([ReadMovieLensDataset.ch_folderName 'm_test.mat'],'m_test','v_range');
		end
		
	end
end
