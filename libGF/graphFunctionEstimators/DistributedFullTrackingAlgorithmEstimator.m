classdef DistributedFullTrackingAlgorithmEstimator< GraphFunctionEstimator
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
        s_maximumTime; % NxS matrix containing the projections of the indicator vectors
        % for nodes that samples were taken to the PW
        % space
        s_bandwidth    % bandwidth of signal considered
        graph;         %graph object containing the adjacency
        s_mu=1.2;
        s_beta=0.5;
    end
    
    
    methods
        
        function obj = DistributedFullTrackingAlgorithmEstimator(varargin)
            obj@GraphFunctionEstimator(varargin{:});
        end
        function N = getNumOfVertices(obj)
            N=0;
        end
    end
    methods
        function t_estimate=estimate(obj,m_samples,m_positions)
            
            
            graph=obj.graph;
            s_bandwidth=obj.s_bandwidth;
            s_numberOfSamples=size(m_samples,1)/obj.s_maximumTime;
            s_numberOfVertices=size(graph.getLaplacian,1);
            m_previousEstimate=zeros(s_numberOfVertices,size(m_samples,2));
            m_projectedLowPassComponents=...
                DistributedFullTrackingAlgorithmEstimator.estimateProjectedLowPassComponents(graph,s_bandwidth);
            m_distancesBetweenVertices=...
                DistributedFullTrackingAlgorithmEstimator.estimateDistancesBetweenVertices(graph);
            t_errorsOnSampled=zeros(s_numberOfSamples,max(max(m_distancesBetweenVertices))+1,size(m_samples,2));
            distributedTrackingAlgorithmEstimator=...
                DistributedTrackingAlgorithmEstimator(...
                'm_projectedLowPassComponents',m_projectedLowPassComponents,...
                'm_distancesBetweenVertices',m_distancesBetweenVertices,...
                't_errorsOnSampled',t_errorsOnSampled,...
                'm_previousEstimate',m_previousEstimate);
            s_mu=obj.s_mu;
            s_beta=obj.s_beta;
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
                
                [t_estimate(v_timetIndicesForSignals,:)...
                    ,t_errorsOnSampled]=...
                    distributedTrackingAlgorithmEstimator.oneStepDistributedEstimete(m_samplest,...
                    m_positionst,s_mu,s_beta);
                % prepare for next iteration
                distributedTrackingAlgorithmEstimator.t_errorsOnSampled=t_errorsOnSampled;
                distributedTrackingAlgorithmEstimator.m_previousEstimate=t_estimate...
                    (v_timetIndicesForSignals,:);
                
            end
        end
        
    end
    methods(Static)
        function m_projectedLowPassComponents=estimateProjectedLowPassComponents(graph,s_bandwidth)
            m_projectedLowPassComponents=zeros(graph.getNumberOfVertices,graph.getNumberOfVertices);
            for s_vertInd=1:graph.getNumberOfVertices
                v_indVec=zeros(graph.getNumberOfVertices,1);
                v_indVec(s_vertInd)=1;
				v_f_tilde_normalized = graph.getNormalizedLaplacianEigenvectors'*v_indVec;
                v_f_tilde_normalized(s_bandwidth+1:end)=0;
                m_projectedLowPassComponents(:,s_vertInd)=graph.getNormalizedLaplacianEigenvectors*v_f_tilde_normalized;
            end
        end
        function m_distancesBetweenVertices= estimateDistancesBetweenVertices(graph)
            % calculates the transmission delay between nodes
            % as the number of hops between them in the graph as
            % described in the paper requires unweighted graph
            %
            m_adjacency=graph.m_adjacency;
            % implement Breadth first Algorithm
            m_adjacency(m_adjacency>0.0001)=1;
            m_adjacency(m_adjacency<=0.0001)=0;
            m_distancesBetweenVertices = DistributedFullTrackingAlgorithmEstimator.computeDist(m_adjacency);
            
        end
        function dist = computeDist(adjacencyMatrix)
            matSize = size(adjacencyMatrix, 1);
            % Set all nodes as unreachable
            dist = inf(matSize);
            % Set the diagonal distances to zero (assumes every node is connected to itself)
            dist(1:(matSize + 1):end) = 0;
            for j = 1:matSize
                % Compute the number of paths bewteen nodes in the matrix at j hops
                pathCount = adjacencyMatrix ^ j;
                idxMap = pathCount > 0;
                dist(idxMap) = min(dist(idxMap), j + zeros(sum(sum(idxMap)), 1));
            end
        end
    end
    
end
