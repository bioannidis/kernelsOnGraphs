classdef ErdosRenyiSimpleExtendedGraphGenerator < GraphGenerator
	
	
	properties % required by parent classes
		c_parsToPrint  = {'ch_name','s_edgeProbability','s_numberOfVertices'};
		c_stringToPrint  = {'','Edge prob.',''};
		c_patternToPrint = {'%s%s random graph','%s = %g','%s%d vertices'};
	end
	
	properties(Constant)
		ch_name = 'Erdos-Renyi';
	end
	
	properties
		v_edgeProbability;
		s_numberOfVertices;
		s_maximumTime;
		v_propagationWeight;
	end
	
	methods
		
		function obj = ErdosRenyiSimpleExtendedGraphGenerator(varargin)
			% Constructor
			obj@GraphGenerator(varargin{:});
			
		end
		
		function graph = realization(obj)
			% Output:
			% GRAPH       Object of class Graph which contains a
			%             realization of Erdos-Renyi random extended
			%             graph. The size of v_edgeProbability can be
			%             1 or s_maximumTime. If it is 1 the same
			%             probability will be used for each graph.
			%             propagation weight correspond to the weight
			%             of the edge between same vertices in
			%             consecutive time steps NOW ONLY SCALAR SUPORTED
			%             The signal is only affected by the previous
			%             value in this context
			
			assert(~isempty(obj.v_edgeProbability));
			assert(~isempty(obj.s_numberOfVertices));
			m_extendedAdjacency=zeros(obj.s_numberOfVertices*obj.s_maximumTime);
			%connects the graphs in consecutive times SHOULD EXTEND
			s_propagationWeight=obj.v_propagationWeight(1);
			m_connectorGraph=s_propagationWeight*diag(ones(obj.s_numberOfVertices,1));
			if (length(obj.v_edgeProbability)==1)
				s_edgeProbability=obj.v_edgeProbability(1);
				for s_ind=1:obj.s_maximumTime
					m_adjacency = rand(obj.s_numberOfVertices) < s_edgeProbability;
					m_adjacency = m_adjacency - diag(diag(m_adjacency)); % replace with something more efficient
					m_adjacency = triu(m_adjacency) + triu(m_adjacency)';% replace with something more efficient
					m_extendedAdjacency((s_ind-1)*(obj.s_numberOfVertices)+1:(s_ind)*(obj.s_numberOfVertices),...
						(s_ind-1)*(obj.s_numberOfVertices)+1:(s_ind)*(obj.s_numberOfVertices))=...
						m_adjacency;
					if s_ind<obj.s_maximumTime
						m_extendedAdjacency((s_ind)*(obj.s_numberOfVertices)+1:(s_ind+1)*(obj.s_numberOfVertices),...
							(s_ind-1)*(obj.s_numberOfVertices)+1:(s_ind)*(obj.s_numberOfVertices))=...
							m_connectorGraph;
						m_extendedAdjacency((s_ind-1)*(obj.s_numberOfVertices)+1:(s_ind)*(obj.s_numberOfVertices),...
							(s_ind)*(obj.s_numberOfVertices)+1:(s_ind+1)*(obj.s_numberOfVertices))=...
							m_connectorGraph';
					end
				end
			else if (length(obj.v_edgeProbability)==obj.s_maximumTime)
					assert('Not supported yet');
					
				else
					assert('Not supported yet');
				end
			end
				
				graph = Graph('m_adjacency',m_extendedAdjacency);
			end
			
		end
		
	end
