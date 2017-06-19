classdef UniformWithOutliersGraphFunctionSampler < GraphFunctionSampler 
	% 
	properties(Constant)
	end
	
	properties
		c_parsToPrint   = {'ch_name','s_numberOfSamples','s_SNR'};
		c_stringToPrint = {'',       'S',                'SNR'};
		c_patternToPrint= {'%s%s',   '%s = %d',          '%s = %g dB'};
	end 
	
	properties
		ch_name = 'Uniform sampler';
		s_numberOfSamples
		s_outlyingPercentage;% [0,1] if a random variable greater than this value
		                     % an outlyier will be added
		s_SNR = Inf;  % signal to noise ratio in dB (Inf <-> no noise)
		% Gaussian noise of variance sigma2 is added to
		% guarantee that
		%                   || graphFunction ||^2 / length(graphFunction)
		% s_SNR = 10*log10(-----------------------------------------------)
		%                                  sigma2
	end
		
	methods
		
		function obj = UniformWithOutliersGraphFunctionSampler(varargin)
			obj@GraphFunctionSampler(varargin{:});
		end
		
	end
	
	methods
		
		function [m_samples,m_positions] = sample(obj,m_graphFunction)
			%
			% Input:
			% M_GRAPHFUNCTION           N x S_NUMBEROFREALIZATIONS  matrix,
			%                           where N is the number of vertices of
			%                           OBJ.graph. Each realization (column) is
			%                           power-normalized (in expectation)
			%
			% Output:
			% M_SAMPLES                 obj.S_NUMBEROFSAMPLES x
			%                           S_NUMBEROFREALIZATIONS  matrix with
			%                           noisy samples of the graph function in
			%                           M_GRAPHFUNCTION.
			% M_POSITIONS               obj.S_NUMBEROFSAMPLES x
			%                           S_NUMBEROFREALIZATIONS matrix
			%                           containing the indices of the vertices
			%                           where the samples were taken. Each
			%                           column is generated drawing vertices
			%                           uniformly at random without replacement.
			
			assert(~isempty(obj.s_numberOfSamples));
			
			s_numberOfVertices = size(m_graphFunction,1);
			s_numberOfRealizations = size(m_graphFunction,2);
			
			m_positions = zeros(obj.s_numberOfSamples,s_numberOfRealizations);
			m_samples = zeros(obj.s_numberOfSamples,s_numberOfRealizations);
			
			for realizationCounter= 1:s_numberOfRealizations
				m_positions(:,realizationCounter) = randperm(s_numberOfVertices,obj.s_numberOfSamples);
				m_samples(:,realizationCounter) = m_graphFunction( m_positions(:,realizationCounter) , realizationCounter );
				
				
				if obj.s_SNR < Inf
					snr = 10^(obj.s_SNR/10);  % natural units
					v_outlyierInd=rand(size(m_samples(:,realizationCounter)))>obj.s_outlyingPercentage;
					noisePower = norm(  m_graphFunction(:,realizationCounter) )^2/(s_numberOfVertices*snr);
					m_samples(:,realizationCounter) = m_samples(:,realizationCounter)+ v_outlyierInd.*(sqrt(noisePower)*randn(obj.s_numberOfSamples,1));
					
				end
				
			end
			
		end
		
	end
	
end
