classdef ErdosRenyiGraphGenerator < GraphGenerator
	
	
	properties % required by parent classes
		c_parsToPrint  = {'ch_name','s_edgeProbability','s_numberOfVertices'};
		c_stringToPrint  = {'','Edge prob.',''};
		c_patternToPrint = {'%s%s random graph','%s = %g','%s%d vertices'};
	end 
	
	properties(Constant)
		ch_name = 'Erdos-Renyi';
	end
	
	properties		
		s_edgeProbability;
		s_numberOfVertices;
	end
	
	methods
		
		function obj = ErdosRenyiGraphGenerator(varargin)
			% Constructor
			obj@GraphGenerator(varargin{:});
			
		end
			
		function graph = realization(obj)
			% Output:
			% GRAPH       Object of class Graph which contains a
			%             realization of Erdos-Renyi random graph
			%
			
			assert(~isempty(obj.s_edgeProbability));
			assert(~isempty(obj.s_numberOfVertices));
			
			m_adjacency = rand(obj.s_numberOfVertices) < obj.s_edgeProbability;
			m_adjacency = m_adjacency - diag(diag(m_adjacency)); % replace with something more efficient
			m_adjacency = triu(m_adjacency) + triu(m_adjacency)';% replace with something more efficient
			
			graph = Graph('m_adjacency',m_adjacency);
		end
		
	end
	
end

