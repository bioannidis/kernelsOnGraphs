classdef SemiParametricGraphFunctionGenerator  < GraphFunctionGenerator
	
	
	properties % Required by superclass Parameter
		c_parsToPrint    = {};
		c_stringToPrint  = {};
		c_patternToPrint = {};
	end 
	
	properties
		name = 'SemiParametric';
		m_parametricBasis; % basis functions
        graphFunctionGenerator; %object of abstract class GraphFunctionGenerator
    end
	
	methods
		
		function obj = SemiParametricGraphFunctionGenerator(varargin)
			% constructor
			obj@GraphFunctionGenerator(varargin{:});
		end
		
		
		function M_graphFunction = realization(obj,s_numberOfRealizations)
			% M_GRAPHFUNCTION   N x S_NUMBEROFREALIZATIONS matrix where N is
			%                   the number of vertices. Each column is a
			%                   signal
			
			assert(~isempty(obj.graph));
			
			if nargin < 2
				s_numberOfRealizations = 1;
            end			
			m_B = obj.m_parametricBasis;
			M_graphFunction =obj.graphFunctionGenerator.realization(s_numberOfRealizations) +m_B*randn(size(m_B,2),s_numberOfRealizations);
			v_powerVect=sqrt(sum(M_graphFunction.^2,1));
			%a=repmat(v_powerVect,size(M_graphFunction,1),1);
			M_graphFunction=M_graphFunction./repmat(v_powerVect,size(M_graphFunction,1),1);
			
		end
	
		
	end
	
end
