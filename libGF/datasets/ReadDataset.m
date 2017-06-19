classdef ReadDataset
	
	methods(Static)
		
		function m_merged = mergeDataMatrices(t_input)
			% t_input:     N x M x P tensor
			%
			% m_merged:    N x M x 1 matrix where m_merged(n,m) is the only
			%              entry of t_input(n,m,:) different from NaN
		
			m_merged = NaN(size(t_input,1),size(t_input,2));
			for rowInd = 1:size(t_input,1)
				for colInd = 1:size(t_input,2)
					slabInd = find( ~isnan( t_input(rowInd,colInd,:) )  );
					if ~isempty(slabInd)
						if length(slabInd)~=1
							keyboard;
						end
						m_merged(rowInd,colInd) = t_input(rowInd,colInd,slabInd);
					end
					
				end				
			end
			
			
		end
		
		
	end
	
	
end