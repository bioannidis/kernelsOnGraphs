classdef ExtendedGraphGenerator < GraphGenerator
	
	
	properties % required by parent classes
		c_parsToPrint  = {};
		c_stringToPrint  = {};
		c_patternToPrint = {};
	end
	
	properties(Constant)
		ch_name = 'ExtendedGraphGenerator';
	end
	
	properties
		t_spatialAdjacency;  % NxNxt tensor containing the adjancency at each
		% time t
		t_timeAdjacency; % NxNx(t-1) tensor containing the effect of the previous
		% time
	end
	
	methods
		
		function obj = ExtendedGraphGenerator(varargin)
			% Constructor
			obj@GraphGenerator(varargin{:});
			
		end
		
		function graph = realization(obj)
			% Output:
			% GRAPH       Object of class Graph which contains a
			%             the extended adjacency
			
			
			assert(~isempty(obj.t_spatialAdjacency));
			assert(~isempty(obj.t_timeAdjacency));
			t_timeAdjacency=obj.t_timeAdjacency;
			t_spatialAdjacency=obj.t_spatialAdjacency;
			s_maximumTime=size(t_spatialAdjacency,3);
			s_numberOfVertices=size(t_timeAdjacency,1);
			m_extendedAdjacency=zeros(s_numberOfVertices*s_maximumTime);
			for s_ind=1:s_maximumTime
				m_extendedAdjacency((s_ind-1)*(s_numberOfVertices)+1:(s_ind)*(s_numberOfVertices),...
					(s_ind-1)*(s_numberOfVertices)+1:(s_ind)*(s_numberOfVertices))=...
					t_spatialAdjacency(:,:,s_ind);
				if s_ind<s_maximumTime
					m_extendedAdjacency((s_ind)*(s_numberOfVertices)+1:(s_ind+1)*(s_numberOfVertices),...
						(s_ind-1)*(s_numberOfVertices)+1:(s_ind)*(s_numberOfVertices))=...
					t_timeAdjacency(:,:,s_ind);
					m_extendedAdjacency((s_ind-1)*(s_numberOfVertices)+1:(s_ind)*(s_numberOfVertices),...
						(s_ind)*(s_numberOfVertices)+1:(s_ind+1)*(s_numberOfVertices))=...
					t_timeAdjacency(:,:,s_ind)';
								
				end
			end
			graph = Graph('m_adjacency',m_extendedAdjacency);
			
		end
	end
end
