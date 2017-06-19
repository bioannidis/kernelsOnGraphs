classdef GraphFunctionGenerator < Parameter
	% Subclasses generate graphs either from real data or randomly
	
	properties(Constant)
	end
	
	properties
		graph   % object of class Graph
	end
		
	methods
		
		function obj = GraphFunctionGenerator(varargin) 
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

