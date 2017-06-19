classdef PartiallyObservedGraphFunctionSampler < GraphFunctionSampler 
	% used when we have unknown values .(recommender systems)
	properties(Constant)
	end
	
	properties
		c_parsToPrint   = {'ch_name','s_numberOfSamples','s_SNR'};
		c_stringToPrint = {'',       'S',                'SNR'};
		c_patternToPrint= {'%s%s',   '%s = %d',          '%s = %g'};
	end 
	
	properties
		ch_name = 'Partially Observed Sampler';
		s_numberOfSamples
		s_SNR = Inf;  % signal to noise ratio in dB (Inf <-> no noise)
        v_knownValuesInd
	end
		
	methods
		
		function obj = PartiallyObservedGraphFunctionSampler(varargin)
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
			v_knownIndeces = find(obj.v_knownValuesInd);
  
			for realizationCounter= 1:s_numberOfRealizations
                v_knownIndecesPermuted=v_knownIndeces(randperm(size(v_knownIndeces,1)));
				m_positions(:,realizationCounter) = v_knownIndecesPermuted(1:obj.s_numberOfSamples);
				m_samples(:,realizationCounter) = m_graphFunction( m_positions(:,realizationCounter) , realizationCounter );
			end
			
			if obj.s_SNR < Inf
				snr = 10^(obj.s_SNR/10);  % natural units
				noisePower = 1/snr;
				m_samples = m_samples + sqrt(noisePower)*randn(obj.s_numberOfSamples,s_numberOfRealizations);
				
			end
			
		end
		
	end
	
end

