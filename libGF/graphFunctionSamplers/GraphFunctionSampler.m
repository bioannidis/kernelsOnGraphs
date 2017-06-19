classdef GraphFunctionSampler < Parameter
	% 
	properties(Constant)
	end
	
	properties
	end
		
	methods
		
		function obj = GraphFunctionSampler(varargin)
			obj@Parameter(varargin{:});
		end
		
	end
	
	methods(Abstract)
				
		[m_samples,m_positions] = sample(obj,m_graphFunction);			
		%
		% Input:
		% M_GRAPHFUNCTION           N x S_NUMBEROFREALIZATIONS  matrix,
		%                           where N is the number of vertices of
		%                           OBJ.graph. Each realization (column) is
		%                           power-normalized (in expectation)
		%
		% Output:
		% M_SAMPLES                 S x S_NUMBEROFREALIZATIONS  matrix with
		%                           samples of the graph function in
		%                           M_GRAPHFUNCTION 
		% M_POSITIONS               S x S_NUMBEROFREALIZATIONS matrix
		%                           containing the indices of the vertices
		%                           where the samples were taken
		%
		
		
		
	end
	
end

