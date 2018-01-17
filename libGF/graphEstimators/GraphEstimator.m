classdef GraphEstimator < Parameter
	% This class is cool
	properties(Constant)
	end
	
	properties
		s_regularizationParameter
		s_numFoldValidation  = 10;
	end
		
	methods
		
		function obj = GraphEstimator(varargin)
			obj@Parameter(varargin{:});
        end
		
   end
	
		
	methods(Abstract)
				
		estimate = estimate(obj,m_graphFunction,sideInfo);			
		%
		% Input:
		% c_SAMPLES                 
		%                           N x L  cell array containing the 
        %                           different graph functions 
        %                           for each lth sample
		%                           
		% Output:                   
		% estimate                  NxN the adjacency
		
	end
	
end

