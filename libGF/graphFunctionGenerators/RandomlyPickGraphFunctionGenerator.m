classdef RandomlyPickGraphFunctionGenerator  < GraphFunctionGenerator
	
	% This class returns the same function every time it is invoked
	
	properties % Required by superclass Parameter
		c_parsToPrint    = {};
		c_stringToPrint  = {};
		c_patternToPrint = {};
	end 
	
	properties
		m_graphFunction % N x n_signals matrix
	end
	
	methods
		
		function obj = RandomlyPickGraphFunctionGenerator(varargin)
			% constructor
			obj@GraphFunctionGenerator(varargin{:});
		end
		
		
		function M_graphFunction = realization(obj,s_numberOfRealizations)
			% M_GRAPHFUNCTION   N x S_NUMBEROFREALIZATIONS matrix where N is
			%                   the number of vertices. Each column is a
			%                   randomly chosen column of OBJ.m_graphFunction
						
			assert(~isempty(obj.m_graphFunction));
			
			if nargin < 2
				s_numberOfRealizations = 1;
			end
						
			n_columns = size(obj.m_graphFunction,2);
			M_graphFunction = obj.m_graphFunction(:,randi(n_columns,1,s_numberOfRealizations));
			
		end
		
		
	end
	
end

