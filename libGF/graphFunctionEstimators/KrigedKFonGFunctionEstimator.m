classdef KrigedKFonGFunctionEstimator< GraphFunctionEstimator
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
		ch_name = 'KrigedKFonGFunctionEstimator';
		
		s_maximumTime;% depth of the extended graph and current time index
		
		t_previousMinimumSquaredError; %NxNxS_NUMBEROFREALIZATIONS  tensor corresponding to the MSE
		%of the previous iteration
		s_mu1=1;
        s_mu2=1;
		m_previousEstimate;            %NxS_NUMBEROFREALIZATIONS
		%matrix corresponding to the previous
		%estimates S_NUMBEROFREALIZATIONS
		%is the number of
		%Monte Carlo iterations
	end
	
	
	methods
		
		function obj = KrigedKFonGFunctionEstimator(varargin)
			obj@GraphFunctionEstimator(varargin{:});
		end
		function N = getNumOfVertices(obj)
			N = size(obj.m_previousEstimate,1);
		end
	end
	
	methods
		function [m_estimateKr,m_estimateKF,t_newMSEKF]=estimate(obj,m_samples,m_positions,m_transitions,m_stateEvolutionKernel,m_spatialCovariance,m_obsNoiseCovariace)
			m_obsNoiseCovariace=size(m_obsNoiseCovariace,1)*m_obsNoiseCovariace;
            [m_estimateKF,t_newMSEKF] =oneStepKF(obj,m_samples,m_positions,m_transitions,m_stateEvolutionKernel,m_spatialCovariance,m_obsNoiseCovariace);
            
			 m_estimateKr=kriging(obj,m_samples,m_positions,m_estimateKF,m_obsNoiseCovariace,m_spatialCovariance);
        end
    
		function [m_estimate,t_newMSE] = oneStepKF(obj,m_samples,m_positions,m_transitions,m_stateNoiseKernel,m_spatialKernel,m_noiseCovariace)
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
            %t_stateEvolutionKernel=repmat(m_stateEvolutionKernel,[1,1,size(t_previousMinimumSquaredError,3)]);
			t_newMSE= zeros(size(t_previousMinimumSquaredError,1),size(t_previousMinimumSquaredError,2),size(t_previousMinimumSquaredError,3));
			for s_realizationCounter = 1:s_numberOfRealizations
				%selection Matrix
				m_phi=zeros(size(m_positions,1),s_numberOfVertices);
				for s_ind=1:size(m_positions,1)
					m_phi(s_ind,m_positions(s_ind,s_realizationCounter))=1;
                end
                m_spatioTempCov=(1/obj.s_mu2)*m_phi*m_spatialKernel*m_phi'...
                    +m_noiseCovariace;
				%CHECK m_phi
				%Prediction
                
				v_prediction=m_transitions*m_previousEstimate(:,s_realizationCounter);
				%Mimumum Prediction MSE Matrix
				m_minPredMinimumMSE=m_transitions*t_previousMinimumSquaredError(:,:,s_realizationCounter)*m_transitions' +(1/obj.s_mu1)*m_stateNoiseKernel;
				%Kalman Gain Matrix
				m_kalmanGain=m_minPredMinimumMSE*m_phi'/(m_spatioTempCov+m_phi*m_minPredMinimumMSE*m_phi');
				%Correction
				m_estimate(:,s_realizationCounter)=v_prediction+m_kalmanGain*(m_samples(:,s_realizationCounter)-m_phi*v_prediction);
				
				%Minuimum MSE Matrix
				t_newMSE(:,:,s_realizationCounter)=(eye(s_numberOfVertices)-m_kalmanGain*m_phi)*m_minPredMinimumMSE;
			end
			
			
			
        end
		
        
        
        
        function [m_estimate] =kriging(obj,m_samples,m_positions,m_spatiotemporalComponent,m_noiseCovariace,m_spatioKernel)
			%
			% Input:
			% M_SAMPLES                 S_t x S_NUMBEROFREALIZATIONS  matrix with
			%                           samples of the graph function in
			%                           M_GRAPHFUNCTION
            % m_spatiotemporalComponent N x S_NUMBEROFREALIZATIONS matrix
            %                         
            %
			% M_POSITIONS               S_t x S_NUMBEROFREALIZATIONS matrix
			%                           containing the indices of the vertices
			%                           where the samples were taken
			%
			%
			%
			% t_spatialCovariance       NxNxS_NUMBEROFREALIZATIONS
			%                           correlation of the spatial
			%                           components in general different for
			%                           each allocation ?
			%                       
			%
			%
			% Output:                   N x S_NUMBEROFREALIZATIONS matrix. N is
			%                           the number of nodes and each column
			%                           contains the estimate of the
			%                           spatial component
			%
			s_numberOfVertices = size(m_spatiotemporalComponent,1);
			s_numberOfRealizations = size(m_spatiotemporalComponent,2);
			
			m_estimate = zeros(s_numberOfVertices,s_numberOfRealizations);
            
            
			for s_realizationCounter = 1:s_numberOfRealizations
				%selection Matrix
				m_phi=zeros(size(m_positions,1),s_numberOfVertices);
				for s_ind=1:size(m_positions,1)
					m_phi(s_ind,m_positions(s_ind,s_realizationCounter))=1;
                end
                m_estimate(:,s_realizationCounter)=(m_spatioKernel*m_phi'...
                    /((1/obj.s_mu2)*m_phi*m_spatioKernel*m_phi'+m_noiseCovariace))*...
                    (m_samples(:,s_realizationCounter)-m_phi*m_spatiotemporalComponent(:,s_realizationCounter));
			end
			
			
			
		end
	end
	
end
