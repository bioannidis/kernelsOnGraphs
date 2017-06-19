classdef DistributedTrackingAlgorithmEstimator< GraphFunctionEstimator
    % This was written by Vassilis
    %% method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
    % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
    
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
        m_projectedLowPassComponents; % NxN matrix containing the projections of the indicator vectors
        % for nodes that samples were taken to the PW
        % space
        
        m_distancesBetweenVertices    % NxN matrix containing the distances between each pair of
        % nodes Dmax the maximum entry of
        % this matrix
        
        t_errorsOnSampled            % SxDmaxS_NUMBEROFREALIZATIONS tensor containing the
        % error of each sampled vertice
        % up to Dmax steps before
        
        m_previousEstimate;          %NxS_NUMBEROFREALIZATIONS
        %matrix corresponding to the previous
        %estimates
    end
    
    
    methods
        
        function obj = DistributedTrackingAlgorithmEstimator(varargin)
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
        function [m_estimate,t_errorsOnSampled] = oneStepDistributedEstimete(obj,m_samples,m_positions,s_mu,s_beta)
            %
            % Input:
            % M_SAMPLES                 S_t x S_NUMBEROFREALIZATIONS  matrix with
            %                           samples of the graph function in
            %                           M_GRAPHFUNCTION
            % M_POSITIONS               S_t x S_NUMBEROFREALIZATIONS matrix
            %                           containing the indices of the vertices
            %                           where the samples were taken
            %
            %
            % Output:                   N x S_NUMBEROFREALIZATIONS matrix. N is
            %                           the number of nodes and each column
            %                           contains the estimate of the graph
            %                           function
            %
            s_numberOfVertices = size(obj.m_previousEstimate,1);
            s_numberOfRealizations = size(obj.m_previousEstimate,2);
            m_distancesBetweenVertices=obj.m_distancesBetweenVertices;
            m_projectedLowPassComponents=obj.m_projectedLowPassComponents;
            t_errorsOnSampled=obj.t_errorsOnSampled;
            m_previousEstimate=obj.m_previousEstimate;
            m_estimate = zeros(s_numberOfVertices,s_numberOfRealizations);
            
            for s_realizationCounter = 1:s_numberOfRealizations
                %update error matrix
                v_previousSampledEstimate=m_previousEstimate...
                    (m_positions(:,s_numberOfRealizations),s_realizationCounter);
                v_currentSamples=m_samples(:,s_numberOfRealizations);
                s_numberOfSamples=size(v_currentSamples,1);
                v_previousEstimate=m_previousEstimate...
                    (:,s_realizationCounter);
                m_errorsOnSampled=t_errorsOnSampled(:,:,s_realizationCounter);
                %shift matrix
                m_errorsOnSampled = circshift(m_errorsOnSampled,1,2);
                m_errorsOnSampled(:,1)=v_currentSamples-v_previousSampledEstimate;
                % update storage
                t_errorsOnSampled(:,:,s_realizationCounter)=m_errorsOnSampled;
                % update Estimation
                for s_vertInd=1:s_numberOfVertices
                    m_estimate(s_vertInd,s_realizationCounter)=...
                        (1-s_mu*s_beta)*v_previousEstimate(s_vertInd);
                    for s_samplVertInd=1:s_numberOfSamples
                        s_samplePosition=m_positions(s_samplVertInd,s_realizationCounter);
                        
                        m_estimate(s_vertInd,s_realizationCounter)=...
                            m_estimate(s_vertInd,s_realizationCounter)...
                            +s_mu*m_errorsOnSampled(s_samplVertInd,...
                            m_distancesBetweenVertices(s_vertInd,s_samplePosition)+1)...
                            *m_projectedLowPassComponents(s_vertInd,s_samplePosition);
                    end
                end
                
            end
            
            
            
        end
        
    end
    
end
