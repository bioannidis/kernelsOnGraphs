classdef SEMGraphFunctionGenerator  < GraphFunctionGenerator
	
	
	properties % Required by superclass Parameter
		c_parsToPrint    = {'ch_name','s_bandwidth','ch_distribution'};
		c_stringToPrint  = {'',    'B',             'distribution'};
		c_patternToPrint = {'%s%s signal','%s = %d','%s = %s'};
	end 
	
	properties
		ch_name = 'SEMGraphFunctionGenerator';
        s_var;
	end
	
	methods
		
		function obj = SEMGraphFunctionGenerator(varargin)
			% constructor
			obj@GraphFunctionGenerator(varargin{:});
		end
		
		
        function m_graphFunction = realization(obj,s_numberOfRealizations)
			% M_GRAPHFUNCTION   N x S_NUMBEROFREALIZATIONS matrix where N is
			%                   the number of vertices. Each column is a
			%                   signal whose graph fourier transform is 
			%                   i.i.d. standard Gaussian distributed for
			%                   the first OBJ.s_bandwidth entries and zero
			%                   for the remaining ones
			
			assert(~isempty(obj.graph));
			
			if nargin < 2
				s_numberOfRealizations = 1;
            end
            
			m_graphFunction=...
                inv(eye(obj.graph.getNumberOfVertices)-obj.graph.m_adjacency)*obj.s_var*randn(obj.graph.getNumberOfVertices, s_numberOfRealizations);
			
		end
		
		
		
	end
	
end

