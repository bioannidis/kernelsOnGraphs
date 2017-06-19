classdef LMSFullTrackingAlgorithmEstimator< GraphFunctionEstimator
    % This was written by Vassilis
    %% method from paper adaptive least mean squares estimation of graph signals
    
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
        s_maximumTime; % NxS matrix containing the projections of the indicator vectors
        % for nodes that samples were taken to the PW
        % space
        s_bandwidth    % bandwidth of signal considered
        graph;         %graph object containing the adjacency
		s_stepLMS; %step of algo
    end
    
    
    methods
        
        function obj = LMSFullTrackingAlgorithmEstimator(varargin)
            obj@GraphFunctionEstimator(varargin{:});
        end
        function N = getNumOfVertices(obj)
            N=0;
        end
    end
    methods
		function m_totalEstimate=estimate(obj,m_samples,m_positions,m_graphFunction)
            
            
            graph=obj.graph;
            s_bandwidth=obj.s_bandwidth;
			s_numberOfRealizations=size(m_samples,2);
            s_numberOfSamples=size(m_samples,1)/obj.s_maximumTime;
            s_numberOfVertices=size(graph.getLaplacian,1);
            m_previousEstimate=randn(s_numberOfVertices,size(m_samples,2));
			m_previousEstimate=LMSFullTrackingAlgorithmEstimator.projectToPWSpaceAndBack(m_previousEstimate,s_bandwidth,graph);
		    m_totalEstimate=zeros(s_numberOfVertices*obj.s_maximumTime,size(m_samples,2));
            
            s_mu=obj.s_stepLMS;
            
            for s_timeInd=1:obj.s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                    (s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                    (s_timeInd)*s_numberOfSamples;
                
                %samples and positions at time t
                m_samplest=m_samples(v_timetIndicesForSamples,:);
                m_positionst=m_positions(v_timetIndicesForSamples,:);
                %estimate
                
             
				for s_realizationCounter = 1:s_numberOfRealizations
				%selection Matrix

				v_differenceObservedEstimates=zeros(s_numberOfVertices,1);
				v_differenceObservedEstimates(m_positionst(:,s_realizationCounter))=...
					m_graphFunction(m_positionst(:,s_realizationCounter),s_realizationCounter)...
					-m_previousEstimate(m_positionst(:,s_realizationCounter),s_realizationCounter);
				m_totalEstimate(v_timetIndicesForSignals,s_realizationCounter)...
                    =m_previousEstimate(:,s_realizationCounter)+...
					s_mu*LMSFullTrackingAlgorithmEstimator.projectToPWSpaceAndBack(v_differenceObservedEstimates,s_bandwidth,graph);
                    
				end
                % prepare for next iteration
                m_previousEstimate=m_totalEstimate(v_timetIndicesForSignals,:);
                
            end
        end
        
    end
    methods(Static)
        function m_projectedLowPassComponents=estimateProjectedLowPassComponents(graph,s_bandwidth,m_)
            m_projectedLowPassComponents=zeros(graph.getNumberOfVertices,graph.getNumberOfVertices);
            for s_vertInd=1:graph.getNumberOfVertices
                v_indVec=zeros(graph.getNumberOfVertices,1);
                v_indVec(s_vertInd)=1;
				v_f_tilde_normalized = graph.getNormalizedLaplacianEigenvectors'*v_indVec;
                v_f_tilde_normalized(s_bandwidth+1:end)=0;
                m_projectedLowPassComponents(:,s_vertInd)=graph.getNormalizedLaplacianEigenvectors*v_f_tilde_normalized;
            end
        end
		function m_projectedLowPassComponents=projectToPWSpaceAndBack(m_previousEstimate,s_bandwidth,graph)
			m_projectedLowPassComponents=zeros(size(m_previousEstimate));
            for s_realizationCounter=1:size(m_previousEstimate,2)               
				v_f_tilde_normalized = graph.getNormalizedLaplacianEigenvectors'*m_previousEstimate(:,s_realizationCounter);
                v_f_tilde_normalized(s_bandwidth+1:end)=0;
                m_projectedLowPassComponents(:,s_realizationCounter)=graph.getNormalizedLaplacianEigenvectors*v_f_tilde_normalized;
            end
		end
	end
    
end
