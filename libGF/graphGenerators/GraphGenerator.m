classdef GraphGenerator < Parameter
	% Subclasses generate graphs either from real data or randomly
	
	properties(Constant)
	end
	
	properties
	end
		
	methods
		
		function obj = GraphGenerator(varargin)
			obj@Parameter(varargin{:});
		end
		
	end
	
	methods(Abstract)
				
		graph = realization(obj);			
		% GRAPH      Object of class Graph
		
	end
	
end

