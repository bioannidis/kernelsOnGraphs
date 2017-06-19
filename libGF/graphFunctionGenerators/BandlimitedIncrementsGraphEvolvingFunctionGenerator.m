classdef BandlimitedIncrementsGraphEvolvingFunctionGenerator  < GraphFunctionGenerator
	
	
	properties % Required by superclass Parameter
		c_parsToPrint    = {'ch_name','s_bandwidth','ch_distribution'};
		c_stringToPrint  = {'',    'B',             'distribution'};
		c_patternToPrint = {'%s%s signal','%s = %d','%s = %s'};
	end
	
	properties
		ch_name = 'Bandlimited';
		ch_distribution = 'normal';
		%    can be 'normal' or 'uniform'
		v_bandwidth ; % bandwidth at each time step for now scalar(same bandwidth)
		s_maximumTime; % maximum Time step
		
		b_sortedSpectrum = 0;  % if 1, the entries of the Fourier transform
		% are generated and then sorted
		
		b_generateSameFunction = 0; % generate the same function if set to 1
		s_mean = 0;
		s_weightDecay;
	end
	
	methods
		
		function obj = BandlimitedIncrementsGraphEvolvingFunctionGenerator(varargin)
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
			assert(~isempty(obj.v_bandwidth));
			s_numberOfVertices=obj.graph.getNumberOfVertices()/obj.s_maximumTime;
			if nargin < 2
				s_numberOfRealizations = 1;
			end
			
			if obj.b_generateSameFunction
				rng(1);
            end
            
			m_graphFunction=zeros(s_numberOfVertices*obj.s_maximumTime,s_numberOfRealizations);
			%multiple bandwidth not supported yet
            if(size(obj.v_bandwidth,1)==1)
				for s_time=1:obj.s_maximumTime
					s_bandwidth=obj.v_bandwidth(1);
					m_B = obj.basis(s_bandwidth,s_time,s_numberOfVertices);
					%freq = randn(obj.s_bandwidth,s_numberOfRealizations);
					if obj.b_sortedSpectrum %not supported
						%[~, ind] = sort(abs(freq), 'descend');
						%M_graphFunction = sqrt(size(m_B,1)/obj.s_bandwidth) * m_B*freq(ind);
						% m_graphFunction = sqrt(size(m_B,1)/obj.s_bandwidth) * ...
						%	m_B*sort(randn(obj.s_bandwidth,s_numberOfRealizations),1,'descend');
					else
						switch obj.ch_distribution
							case 'normal'
								if s_time==1
								m_graphFunction(s_numberOfVertices*(s_time-1)+1:s_numberOfVertices*s_time,:)...
									= sqrt(size(m_B,1)/obj.v_bandwidth) * ...
									m_B*(randn(obj.v_bandwidth,s_numberOfRealizations));
								else
										m_graphFunction(s_numberOfVertices*(s_time-1)+1:...
                                            s_numberOfVertices*s_time,:)...
									=obj.s_weightDecay*m_graphFunction(s_numberOfVertices*(s_time-2)+1:...
                                    s_numberOfVertices*(s_time-1),:)+...
									randn*sqrt(size(m_B,1)/obj.v_bandwidth) * ...
									m_B*(randn(obj.v_bandwidth,s_numberOfRealizations));
								end
							case 'uniform'
								if s_time==1
								m_graphFunction(s_numberOfVertices*(s_time-1)+1:s_numberOfVertices*s_time,:)...
									= sqrt(size(m_B,1)/obj.v_bandwidth) * ...
									m_B*(rand(obj.v_bandwidth,s_numberOfRealizations));
								else
										m_graphFunction(s_numberOfVertices*(s_time-1)+1:s_numberOfVertices*s_time,:)...
									=obj.s_weightDecay*m_graphFunction(s_numberOfVertices*(s_time-2)+1:...
                                    s_numberOfVertices*(s_time-1),:)+...
									rand*sqrt(size(m_B,1)/obj.v_bandwidth) * ...
									m_B*(rand(obj.v_bandwidth,s_numberOfRealizations));
								end
							otherwise
								error('unrecognized distribution');
						end
						% normalize
						%M_graphFunction = M_graphFunction - mean(M_graphFunction);
						%M_graphFunction = M_graphFunction / norm(M_graphFunction);
						%M_graphFunction = randn(obj.s_bandwidth,s_numberOfRealizations);
					end
					
					%             N = obj.graph.getNumberOfVertices();
					%             atilde = (1:N)';
					%             alpha = exp( - atilde / 50 );
					%             V = obj.graph.getLaplacianEigenvectors();
					%             M_graphFunction = 10*V*alpha * ones(s_numberOfRealizations,1);
				end
			end
		end
		
		function m_basis = basis(obj,s_otherBandwidth,s_time,s_numberOfVertices)
			%  M_BASIS            N x S_OTHERBANDWIDTH matrix containing the
			%                     first OBJ.s_bandwidth eigenvectors of the
			%                     Laplacian
			%  S_OTHERBANDWIDTH   optional parameter to specify the number of
			%                     desired columns. (Default: =
			%                     obj.s_bandwidth)
			
			if nargin<2 % default option
				s_otherBandwidth = obj.v_bandwidth;
			end
			m_adjacency = obj.graph.m_adjacency((s_time-1)*s_numberOfVertices+1:(s_time)*s_numberOfVertices,...
				(s_time-1)*s_numberOfVertices+1:(s_time)*s_numberOfVertices);
			v_degrees = sum(m_adjacency,2);
			m_L = diag(v_degrees) - m_adjacency;
			[m_V,~,~] = eig(m_L);
			m_basis = m_V(:,1:s_otherBandwidth);
			
		end
		
		
	end
	
end

