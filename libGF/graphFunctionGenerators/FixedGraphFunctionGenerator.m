classdef FixedGraphFunctionGenerator  < GraphFunctionGenerator
	
	% This class returns the same function every time it is invoked
	
	properties % Required by superclass Parameter
		c_parsToPrint    = {};
		c_stringToPrint  = {};
		c_patternToPrint = {};
	end 
	
	properties
		graphFunction
	end
	
	methods
		
		function obj = FixedGraphFunctionGenerator(varargin)
			% constructor
			obj@GraphFunctionGenerator(varargin{:});
		end
		
		
		function M_graphFunction = realization(obj,s_numberOfRealizations)
			% M_GRAPHFUNCTION   N x S_NUMBEROFREALIZATIONS matrix where N is
			%                   the number of vertices. Each column is a
			%                   replica of the function OBJ.graphFunction
			
			assert(~isempty(obj.graph));
			assert(~isempty(obj.graphFunction));
			
			if nargin < 2
				s_numberOfRealizations = 1;
			end
			
			
			M_graphFunction = repmat(obj.graphFunction(:),1,s_numberOfRealizations);
			
		end
		
		
	end
	
end

