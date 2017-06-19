classdef GenKFFunctionEstimator< GraphFunctionEstimator
	% This was written by Vassilis
	%KalmanFilterOnGraphsFunctionEstimator
	%it its implemented for one step prediction
	%needs initial conditions
	%it is programmed so that it resets the new values
	%needed for next iteration
	
	properties(Constant)
	end
	
	properties % Required by superclass Parameter
		c_parsToPrint    = {};
		c_stringToPrint  = {};
		c_patternToPrint = {};
	end
	%%
	properties
		ch_name = 'KFOnGFunctionEstimator';
		
		s_maximumTime;% depth of the extended graph and current time index
		
		t_previousMinimumSquaredError; %NxNxS_NUMBEROFREALIZATIONS  tensor corresponding to the MSE
		%of the previous iteration
		
		m_previousEstimate;            %NxS_NUMBEROFREALIZATIONS
		%matrix corresponding to the previous
		%estimates S_NUMBEROFREALIZATIONS
		%is the number of
		%Monte Carlo iterations
	end
	
	
	methods
		
		function obj = GenKFFunctionEstimator(varargin)
			obj@GraphFunctionEstimator(varargin{:});
		end
		function N = getNumOfVertices(obj)
			N = size(obj.m_previousEstimate,1);
		end
	end
	
	methods
		function m_estimate=estimate(obj,m_samples,m_positions)
			%not implemented
			m_estimate=[];
		end
		function [m_estimate,t_newMSE] = oneStepKF(obj,m_samples,m_positions,m_transitions,m_correlations,s_sigma)
			%
			% Input:
			% M_SAMPLES                 S_t x S_NUMBEROFREALIZATIONS  matrix with
			%                           samples of the graph function in
			%                           M_GRAPHFUNCTION
			% M_POSITIONS               S_t x S_NUMBEROFREALIZATIONS matrix
			%                           containing the indices of the vertices
			%                           where the samples were taken
			%
			% m_trantions               NxN transition matrix at time t
			%
			%
			% m_correlations            NxN noise correlation matrix
			%                           at time t
			%
			%
			% Output:                   N x S_NUMBEROFREALIZATIONS matrix. N is
			%                           the number of nodes and each column
			%                           contains the estimate of the graph
			%                           function
			%
			s_numberOfVertices = size(obj.m_previousEstimate,1);
			s_numberOfRealizations = size(obj.m_previousEstimate,2);
			t_previousMinimumSquaredError=obj.t_previousMinimumSquaredError;
			m_previousEstimate=obj.m_previousEstimate;
			m_estimate = zeros(s_numberOfVertices,s_numberOfRealizations);
			t_newMSE= zeros(size(t_previousMinimumSquaredError,1),size(t_previousMinimumSquaredError,2),size(t_previousMinimumSquaredError,3));
			for s_realizationCounter = 1:s_numberOfRealizations
				%selection Matrix
				m_phi=zeros(size(m_positions,1),s_numberOfVertices);
				for s_ind=1:size(m_positions,1)
					m_phi(s_ind,m_positions(s_ind,s_realizationCounter))=1;
				end
				%CHECK m_phi
				%Prediction
				v_prediction=m_transitions*m_previousEstimate(:,s_realizationCounter);
				%Mimumum Prediction MSE Matrix
				m_minPredMinimumMSE=m_transitions*t_previousMinimumSquaredError(:,:,s_realizationCounter)*m_transitions' +m_correlations;
				%Kalman Gain Matrix
				m_kalmanGain=m_minPredMinimumMSE*m_phi'/(s_sigma^2*eye(size(m_phi,1))+m_phi*m_minPredMinimumMSE*m_phi');
				%Correction
				m_estimate(:,s_realizationCounter)=v_prediction+m_kalmanGain*(m_samples(:,s_realizationCounter)-m_phi*v_prediction);
				
				%Minuimum MSE Matrix
				t_newMSE(:,:,s_realizationCounter)=(eye(s_numberOfVertices)-m_kalmanGain*m_phi)*m_minPredMinimumMSE;
			end
			
			
			
		end
		
	end
	
end
