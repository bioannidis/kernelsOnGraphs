classdef RTSSmootherFunctionEstimator< GraphFunctionEstimator
	% This was written by Vassilis
	
	
	properties(Constant)
	end
	
	properties % Required by superclass Parameter
		c_parsToPrint    = {};
		c_stringToPrint  = {};
		c_patternToPrint = {};
	end
	%%
	properties
		ch_name = 'RTSSmootherFunctionEstimator';
		
		s_maximumTime;
        m_initialState;    %NxS_montecarlo
        t_initialErrorcov; %NxNxS_montecarlo
        m_transMat;
        m_transCov;
        s_sigma; %1/square root of mu
	end
	
	
	methods
		
		function obj = RTSSmootherFunctionEstimator(varargin)
			obj@GraphFunctionEstimator(varargin{:});
		end
		function N = getNumOfVertices(obj)
			N = size(obj.m_previousEstimate,1);
		end
	end
	
	methods
        function [m_initialGraphSignal,t_smoothedEstimate]=estimate(obj,t_samples,t_positions)
			%not implemented
            s_numberOfVertices=size(obj.m_transMat,1);
            s_monteCarlo=size(obj.m_initialState,2);
            t_filteredEstimates=zeros(s_numberOfVertices,obj.s_maximumTime+1,s_monteCarlo); %Nxs_maximumTimexS_montecarlo
            t_predictions=zeros(s_numberOfVertices,obj.s_maximumTime+1,s_monteCarlo);        %Nxs_maximumTimexS_montecarlo
            t_errorPredCov=zeros(s_numberOfVertices,s_numberOfVertices,obj.s_maximumTime+1,s_monteCarlo);       %NxNxs_maximumTimexS_montecarlo
            t_errorCov=zeros(s_numberOfVertices,s_numberOfVertices,obj.s_maximumTime+1,s_monteCarlo);           %NxNxs_maximumTimexS_montecarlo
            t_errorCov(:,:,1,:)=obj.t_initialErrorcov;
            t_filteredEstimates(:,1,:)=obj.m_initialState;
            %carefull for the initial 1 or 2
            for s_t=2:obj.s_maximumTime+1
               [t_filteredEstimates(:,s_t,:),t_predictions(:,s_t,:),t_errorCov(:,:,s_t,:),t_errorPredCov(:,:,s_t,:)]=...
                   oneStepKF(obj,squeeze(t_samples(:,s_t-1,:)),squeeze(t_positions(:,s_t-1,:)),...
                   squeeze(t_filteredEstimates(:,s_t-1,:)),squeeze(t_errorCov(:,:,s_t-1,:)));
               
            end
            t_smoothedEstimate=zeros(s_numberOfVertices,obj.s_maximumTime+1,s_monteCarlo); %Nxs_maximumTimexS_montecarlo
            t_smoothedEstimate(:,obj.s_maximumTime+1,:)=t_filteredEstimates(:,obj.s_maximumTime+1,:);
            %check if T or T+1
            for s_t=obj.s_maximumTime:-1:1
                t_smoothedEstimate(:,s_t,:)=...
                    oneStepKS(obj,squeeze(t_predictions(:,s_t+1,:)),squeeze(t_filteredEstimates(:,s_t,:))...
                    ,squeeze(t_errorCov(:,:,s_t,:)),squeeze(t_errorPredCov(:,:,s_t+1,:)),squeeze(t_smoothedEstimate(:,s_t+1,:)));
            end
            m_initialGraphSignal=squeeze(t_smoothedEstimate(:,1,:));
            t_smoothedEstimate=t_smoothedEstimate(:,2:end,:);
            
        end
        function [m_smoothedEstimate]=oneStepKS(obj,m_prediction,m_estimate,t_newMSE,t_minPredMinimumMSE,m_nextEstimate)
            m_transitions=obj.m_transMat;
			s_numberOfRealizations = size(m_nextEstimate,2);
            m_smoothedEstimate=zeros(size(m_estimate));
            for s_realizationCounter =1: s_numberOfRealizations
                m_kalmanSmootherGain=t_newMSE(:,:,s_realizationCounter)*m_transitions'/(t_minPredMinimumMSE(:,:,s_realizationCounter));
                m_smoothedEstimate(:,s_realizationCounter)=m_estimate(:,s_realizationCounter)+m_kalmanSmootherGain*(m_prediction(:,s_realizationCounter)-m_nextEstimate(:,s_realizationCounter));
            end
        end
		function [m_estimate,m_prediction,t_newMSE,t_minPredMinimumMSE] = oneStepKF(obj,t_samples,t_positions,m_previousEstimate,t_previousMinimumSquaredError)
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
			m_transitions=obj.m_transMat;
            m_correlations=obj.m_transCov;
            s_numberOfVertices = size(m_transitions,1);
			s_numberOfRealizations = size(m_previousEstimate,2);
			m_estimate = zeros(s_numberOfVertices,s_numberOfRealizations);
            m_prediction = zeros(s_numberOfVertices,s_numberOfRealizations);

			t_newMSE= zeros(size(t_previousMinimumSquaredError,1),size(t_previousMinimumSquaredError,2),size(t_previousMinimumSquaredError,3));
            t_minPredMinimumMSE= zeros(size(t_previousMinimumSquaredError,1),size(t_previousMinimumSquaredError,2),size(t_previousMinimumSquaredError,3));
			for s_realizationCounter = 1:s_numberOfRealizations
				%selection Matrix
				m_phi=zeros(size(t_positions,1),s_numberOfVertices);
				for s_ind=1:size(t_positions,1)
					m_phi(s_ind,t_positions(s_ind,s_realizationCounter))=1;
				end
				%CHECK m_phi
				%Prediction
				v_prediction=m_transitions*m_previousEstimate(:,s_realizationCounter);
                m_prediction(:,s_realizationCounter)=v_prediction;
				%Mimumum Prediction MSE Matrix
				m_minPredMinimumMSE=m_transitions*t_previousMinimumSquaredError(:,:,s_realizationCounter)*m_transitions' +m_correlations;
				t_minPredMinimumMSE(:,:,s_realizationCounter)=m_minPredMinimumMSE;
                %Kalman Gain Matrix
				m_kalmanGain=m_minPredMinimumMSE*m_phi'/(obj.s_sigma^2*eye(size(m_phi,1))+m_phi*m_minPredMinimumMSE*m_phi');
				%Correction
				m_estimate(:,s_realizationCounter)=v_prediction+m_kalmanGain*(t_samples(:,s_realizationCounter)-m_phi*v_prediction);
				
				%Minuimum MSE Matrix
				t_newMSE(:,:,s_realizationCounter)=(eye(s_numberOfVertices)-m_kalmanGain*m_phi)*m_minPredMinimumMSE;
			end
			
			
			
		end
		
	end
	
end
