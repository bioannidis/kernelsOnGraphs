classdef EvolvingGraphFunctionGenerator < Parameter
	% Subclasses generate graphs either from real data or randomly
	
	properties(Constant)
	end
	
	properties
		t_adjacency   % adjacency at each time
	end
		
	methods
		
		function obj = EvolvingGraphFunctionGenerator(varargin) 
			obj@Parameter(varargin{:});
		end
		
	end
	
	methods(Abstract)
				
		M_graphFunction = realization(obj,s_numberOfRealizations);
		%
		% Input:
		% S_NUMBEROFREALIZATIONS     
		%
		% Output:
		% M_GRAPHFUNCTION           N x S_NUMBEROFREALIZATIONS  matrix,
		%                           where N is the number of vertices of
		%                           OBJ.graph. Each realization (column) is
		%                           power-normalized (in expectation)
		
	end
	
end

