%
%
%  FIGURES FOR THE PAPER ON Kalman Filter On Graphs
%
%
classdef KFonGSimulations < simFunctionSet
    
    properties
        
    end
    
    methods
        
        %% Synthetic data simulations
        %  Data used:
        %  Goal: Compare Batch Approach up to time t
        %  with Kalman filter estimate for time t
        %  Kernel used: inverse Laplacian
        function F = compute_fig_1001(obj,niter)
            
            %% 0. define parameters
            s_maximumTime=100;
            s_mu=10^-6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            
            v_numberOfSamples=5;    % must extend to support vector case
            
            s_numberOfVertices=10;  % size of the graph
            
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            v_sigma=sqrt((1:s_maximumTime)*v_numberOfSamples*s_mu)'; %choosen so Kernel Regression
            %is equivelant with
            %Kalman filter
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_edgeProbability=0.6;   %probability of connection
            v_propagationWeight=0.8; % weight of edges between the same node
            % in consecutive time instances
            graphGenerator = ErdosRenyiSimpleExtendedGraphGenerator('v_edgeProbability',...
                v_edgeProbability,'s_numberOfVertices',s_numberOfVertices,'s_maximumTime',...
                s_maximumTime,'v_propagationWeight',v_propagationWeight);
            graph=graphGenerator.realization;
            %toc
            %% 2. choise of Kernel must be positive definite
            % inverse Laplacian
            s_beta=1; % ensures that Kernel is positive definite
            m_kernelInv=graph.getLaplacian()+s_beta*eye(graph.getNumberOfVertices);
            [~,p1]=chol(m_kernelInv);%check if PD
            [~,p2]=chol(m_kernelInv((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
                s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
                *s_numberOfVertices));
            if (p1+p2~=0)
                assert(1==0,'Not Positive Definite Kernel')
            end
            %% generate transition, correlation matrices
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (m_kernelInv,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            s_weightDecay=0.1; %shows the percentage of the previous signal that will be added
            %to the next
            v_bandwidth=5;    %bandwidth of the bandlimited signal
            
            bandlimitedIncrementsGraphEvolvingFunctionGenerator=...
                BandlimitedIncrementsGraphEvolvingFunctionGenerator('graph',graph,'v_bandwidth',v_bandwidth,...
                's_maximumTime',s_maximumTime,'s_weightDecay',s_weightDecay);
            m_graphFunction=bandlimitedIncrementsGraphEvolvingFunctionGenerator.realization...
                (s_monteCarloSimulations);
            
            %% 4. generate observations
            
            sampler = UniformGraphFunctionSampler('s_numberOfSamples',v_numberOfSamples,'s_SNR',s_SNR);
            m_samples=zeros(v_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
            m_positions=zeros(v_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
            for s_timeInd=1:s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:(s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*v_numberOfSamples+1:(s_timeInd)*v_numberOfSamples;
                
                [m_samples(v_timetIndicesForSamples,:),...
                    m_positions(v_timetIndicesForSamples,:)]...
                    = sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
            end
            
            %% 5. batch estimate
            nonParametricBatchGraphFunctionEstimator=NonParametricBatchGraphFunctionEstimator...
                ('m_kernels',inv(m_kernelInv),...
                's_mu',s_mu,'s_maximumTime',s_maximumTime);
            m_batchEstimate=nonParametricBatchGraphFunctionEstimator.estimate...
                (m_samples,m_positions,v_numberOfSamples);
            
            %% 6. kalman estimate
            m_kfEstimate=zeros(size(m_batchEstimate));
            kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                't_previousMinimumSquaredError',t_initialSigma0,...
                'm_previousEstimate',m_initialState);
            for s_timeInd=1:s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:(s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*v_numberOfSamples+1:(s_timeInd)*v_numberOfSamples;
                
                %samples and positions at time t
                m_samplest=m_samples(v_timetIndicesForSamples,:);
                m_positionst=m_positions(v_timetIndicesForSamples,:);
                %estimate
                
                [m_kfEstimate(v_timetIndicesForSignals,:),t_newMSE]=...
                    kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,t_transitions(:,:,s_timeInd),...
                    t_correlations(:,:,s_timeInd),v_sigma(s_timeInd));
                % prepare KF for next iteration
                kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                kFOnGFunctionEstimator.m_previousEstimate=m_kfEstimate(v_timetIndicesForSignals,:);
                
            end
            
            %% 7. measure difference
            v_difBatchKF=(m_batchEstimate(v_timetIndicesForSignals,:)...
                -m_kfEstimate(v_timetIndicesForSignals,:)...
                )
            toc
            s_relativeErrorBatch=norm(m_batchEstimate-m_graphFunction,'fro')/norm(m_graphFunction,'fro')
            
            s_relativeErrorKF=norm(m_kfEstimate-m_graphFunction,'fro')/norm(m_graphFunction,'fro')
            %% generate figure
            F=[];
        end
        
        
        
        %% Synthetic data simulations
        %  Data used:
        %  Goal: Compare Batch Approach up to time t
        %  with Bandlimited model at each time t
        %  Kernel used: inverse Laplacian
        function F = compute_fig_1002(obj,niter)
            
            %% 0. define parameters
            s_maximumTime=200;
            s_mu=10^-5;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_numberOfSamples=5;    % must extend to support vector case
            s_numberOfVertices=10;  % size of the graph
            
            % in consecutive time instances
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            
            %to be equivelant with
            %Kalman filter
            
            %% 1. define graph
            tic
            v_edgeProbability=0.6;   %probability of connection
            v_propagationWeight=0.8; % weight of edges between the same node
            % in consecutive time instances
            graphGenerator = ErdosRenyiSimpleExtendedGraphGenerator('v_edgeProbability',...
                v_edgeProbability,'s_numberOfVertices',s_numberOfVertices,'s_maximumTime',...
                s_maximumTime,'v_propagationWeight',v_propagationWeight);
            graphT=graphGenerator.realization;
            %toc
            %% 2. choise of Kernel must be positive definite
            % inverse Laplacian
            s_beta=1; % ensures that Kernel is positive definite
            m_kernelInv=graphT.getLaplacian()+s_beta*eye(graphT.getNumberOfVertices);
            [~,p1]=chol(m_kernelInv);%check if PD
            [~,p2]=chol(m_kernelInv((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
                s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
                *s_numberOfVertices));
            if (p1+p2~=0)
                assert(1==0,'Not Positive Definite Kernel')
            end
            
            
            %% 3. generate true signal
            s_weightDecay=0.1; %shows the percentage of the previous signal that will be added
            %to the next
            v_bandwidth=5;    %bandwidth of the bandlimited signal
            
            bandlimitedIncrementsGraphEvolvingFunctionGenerator=...
                BandlimitedIncrementsGraphEvolvingFunctionGenerator('graph',graphT,'v_bandwidth',...
                v_bandwidth,'s_maximumTime',s_maximumTime,'s_weightDecay',s_weightDecay);
            m_graphFunction=bandlimitedIncrementsGraphEvolvingFunctionGenerator.realization...
                (s_monteCarloSimulations);
            
            %% 4. generate observations
            
            sampler = UniformGraphFunctionSampler('s_numberOfSamples',v_numberOfSamples,'s_SNR',s_SNR);
            m_samples=zeros(v_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
            m_positions=zeros(v_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
            for s_timeInd=1:s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:(s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*v_numberOfSamples+1:(s_timeInd)*v_numberOfSamples;
                
                [m_samples(v_timetIndicesForSamples,:),...
                    m_positions(v_timetIndicesForSamples,:)]...
                    = sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
            end
            
            %% 5. batch estimate
            nonParametricBatchGraphFunctionEstimator=NonParametricBatchGraphFunctionEstimator...
                ('m_kernels',inv(m_kernelInv),...
                's_mu',s_mu,'s_maximumTime',s_maximumTime);
            m_batchEstimate=nonParametricBatchGraphFunctionEstimator.estimate...
                (m_samples,m_positions,v_numberOfSamples);
            
            %% 6. bandlimited estimate
            m_bandLimitedEstimate=zeros(size(m_batchEstimate));
            m_extendedAdjancency=graphT.m_adjacency;
            for s_timeInd=1:s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:(s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*v_numberOfSamples+1:(s_timeInd)*v_numberOfSamples;
                
                %samples and positions at time t
                
                m_samplest=m_samples(v_timetIndicesForSamples,:);
                m_positionst=m_positions(v_timetIndicesForSamples,:);
                %create take diagonals from extended graph
                m_adjacency=m_extendedAdjancency(v_timetIndicesForSignals,v_timetIndicesForSignals);
                grapht=Graph('m_adjacency',m_adjacency);
                
                %bandlimited estimate
                bandlimitedGraphFunctionEstimator= BandlimitedGraphFunctionEstimator('m_laplacian'...
                    ,grapht.getLaplacian,'s_bandwidth',v_bandwidth);
                m_bandLimitedEstimate(v_timetIndicesForSignals,:)=...
                    bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                
            end
            
            %% 7. measure difference
            
            s_relativeErrorBatch=norm(m_batchEstimate-m_graphFunction,'fro')/norm(m_graphFunction,'fro')
            
            s_relativeErrorbandLimitedEstimate...
                =norm(m_bandLimitedEstimate-m_graphFunction,'fro')/norm(m_graphFunction,'fro')
            %% generate figure
            F=[];
        end
        
        %% Synthetic data simulations
        %  Data used:
        %  Goal: Kalman filter how it tracks a specific node over time vs
        %  Batch approach
        %  Kernel used: inverse Laplacian
        function F = compute_fig_1003(obj,niter)
            
            %% 0. define parameters
            % maximum signal instances sampled
            s_maximumTime=20;
            s_mu=10^-4;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            
            v_numberOfSamples=3;    % must extend to support vector case
            
            s_numberOfVertices=10;  % size of the graph
            
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            v_sigma=sqrt((1:s_maximumTime)*v_numberOfSamples*s_mu)'; %choosen so Kernel Regression
            %is equivelant with
            %Kalman filter
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_edgeProbability=0.6;   %probability of connection
            v_propagationWeight=0.8; % weight of edges between the same node
            % in consecutive time instances
            graphGenerator = ErdosRenyiSimpleExtendedGraphGenerator('v_edgeProbability',...
                v_edgeProbability,'s_numberOfVertices',s_numberOfVertices,'s_maximumTime',...
                s_maximumTime,'v_propagationWeight',v_propagationWeight);
            graphT=graphGenerator.realization;
            %toc
            %% 2. choise of Kernel must be positive definite
            % inverse Laplacian
            s_beta=1; % ensures that Kernel is positive definite
            m_kernelInv=graphT.getLaplacian()+s_beta*eye(graphT.getNumberOfVertices);
            [~,p1]=chol(m_kernelInv);%check if PD
            [~,p2]=chol(m_kernelInv((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
                s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
                *s_numberOfVertices));
            if (p1+p2~=0)
                assert(1==0,'Not Positive Definite Kernel')
            end
            %% generate transition, correlation matrices
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (m_kernelInv,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            s_weightDecay=0.1; %shows the percentage of the previous signal that will be added
            %to the next
            v_bandwidth=5;    %bandwidth of the bandlimited signal
            
            bandlimitedIncrementsGraphEvolvingFunctionGenerator=...
                BandlimitedIncrementsGraphEvolvingFunctionGenerator('graph',graphT,'v_bandwidth',v_bandwidth,...
                's_maximumTime',s_maximumTime,'s_weightDecay',s_weightDecay);
            m_graphFunction=bandlimitedIncrementsGraphEvolvingFunctionGenerator.realization...
                (s_monteCarloSimulations);
            
            %% 4. generate observations
            
            sampler = UniformGraphFunctionSampler('s_numberOfSamples',v_numberOfSamples,'s_SNR',s_SNR);
            m_samples=zeros(v_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
            m_positions=zeros(v_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
            for s_timeInd=1:s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:(s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*v_numberOfSamples+1:(s_timeInd)*v_numberOfSamples;
                
                [m_samples(v_timetIndicesForSamples,:),...
                    m_positions(v_timetIndicesForSamples,:)]...
                    = sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
            end
            
            %% 5. batch estimate
            nonParametricBatchGraphFunctionEstimator=NonParametricBatchGraphFunctionEstimator...
                ('m_kernels',inv(m_kernelInv),...
                's_mu',s_mu,'s_maximumTime',s_maximumTime);
            m_batchEstimate=nonParametricBatchGraphFunctionEstimator.estimate...
                (m_samples,m_positions,v_numberOfSamples);
            
            %% 6. kalman estimate
            m_kfEstimate=zeros(size(m_batchEstimate));
            kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                't_previousMinimumSquaredError',t_initialSigma0,...
                'm_previousEstimate',m_initialState);
            for s_timeInd=1:s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:(s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*v_numberOfSamples+1:(s_timeInd)*v_numberOfSamples;
                
                %samples and positions at time t
                m_samplest=m_samples(v_timetIndicesForSamples,:);
                m_positionst=m_positions(v_timetIndicesForSamples,:);
                %estimate
                
                [m_kfEstimate(v_timetIndicesForSignals,:),t_newMSE]=...
                    kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,t_transitions(:,:,s_timeInd),...
                    t_correlations(:,:,s_timeInd),v_sigma(s_timeInd));
                % prepare KF for next iteration
                kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                kFOnGFunctionEstimator.m_previousEstimate=m_kfEstimate(v_timetIndicesForSignals,:);
                
            end
            toc
            
            %% 7. measure difference
            v_difBatchKF=(m_batchEstimate(v_timetIndicesForSignals,:)...
                -m_kfEstimate(v_timetIndicesForSignals,:)...
                )
            
            s_relativeErrorBatch=norm(m_batchEstimate-m_graphFunction,'fro')/norm(m_graphFunction,'fro')
            
            s_relativeErrorKF=norm(m_kfEstimate-m_graphFunction,'fro')/norm(m_graphFunction,'fro')
            
            % create vector to plot
            %choose vertice for plotting
            s_vertToPlot=1;
            s_chooseRealization=2;
            %take the mean across montecarlos CHECK
            v_graphFunction=m_graphFunction(:,s_chooseRealization);
            v_kfEstimate=m_kfEstimate(:,s_chooseRealization);
            v_batchEstimate=m_batchEstimate(:,s_chooseRealization);
            v_positions=m_positions(:,s_chooseRealization);
            v_vertPositions=zeros(s_maximumTime,1);
            v_vertEstimateKf=zeros(s_maximumTime,1);
            v_vertEstimateBatch=zeros(s_maximumTime,1);
            v_vertGraphFunction=zeros(s_maximumTime,1);
            for s_timeInd=1:s_maximumTime
                v_timetIndicesForSamples=(s_timeInd-1)*v_numberOfSamples+1:(s_timeInd)*v_numberOfSamples;
                v_vertPositions(s_timeInd)=any(s_vertToPlot==v_positions(v_timetIndicesForSamples));
                v_vertEstimateKf(s_timeInd)=v_kfEstimate((s_timeInd-1)*s_numberOfVertices+s_vertToPlot);
                v_vertEstimateBatch(s_timeInd)=v_batchEstimate((s_timeInd-1)*s_numberOfVertices+s_vertToPlot);
                v_vertGraphFunction(s_timeInd)=v_graphFunction((s_timeInd-1)*s_numberOfVertices+s_vertToPlot);
            end
            v_vertPositions=~v_vertPositions;
            v_vertSampledValues=(v_vertGraphFunction.*v_vertPositions)';
            [~,v_posSampl,v_vertNonZeroSampledValues]=find(v_vertSampledValues);
            plot(1:s_maximumTime,v_vertEstimateBatch','b',...
                1:s_maximumTime,v_vertEstimateKf','r',...
                1:s_maximumTime,v_vertGraphFunction','g',...
                v_posSampl',v_vertNonZeroSampledValues','p');
            legend('Batch Estimate','KF Estimate','True function');
            %% generate figure
            F=[];
        end
        
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare Batch Approach up to time t
        %  with Kalman filter estimate for time t
        %  Kernel used: inverse Laplacian
        function F = compute_fig_2001(obj,niter)
            
            %% 0. define parameters
            % maximum signal instances sampled
            s_maximumTime=10;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720
            s_samplePeriod=360;
            s_mu=10^-8;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=0.1;
            v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            
            
            v_propagationWeight=200.0; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            
            
            v_sigma=sqrt((1:s_maximumTime)*v_numberOfSamples*s_mu)'; %choosen so Kernel Regression
            %is equivelant with
            %Kalman filter
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                zeros(s_numberOfVertices,s_numberOfVertices,s_maximumTime);
            t_timeAdjacencyAtDifferentTimes...
                =zeros(s_numberOfVertices,s_numberOfVertices,s_maximumTime-1);
            for s_timeInd=1:s_maximumTime
                t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd)=m_adjacency;
                if(s_timeInd~=s_maximumTime)
                    t_timeAdjacencyAtDifferentTimes(:,:,s_timeInd)=v_propagationWeight*eye(size(m_adjacency));
                end
            end
            
            graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            graphT=graphGenerator.realization;
            
            %% 2. choise of Kernel must be positive definite
            % inverse Laplacian
            s_beta=1; % ensures that Kernel is positive definite
            m_kernelInv=graphT.getLaplacian()+s_beta*eye(graphT.getNumberOfVertices);
            [~,p1]=chol(m_kernelInv);%check if PD
            [~,p2]=chol(m_kernelInv((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
                s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
                *s_numberOfVertices));
            if (p1+p2~=0)
                assert(1==0,'Not Positive Definite Kernel')
            end
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (m_kernelInv,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            %% 4. generate observations
            
            sampler = UniformGraphFunctionSampler('s_numberOfSamples',v_numberOfSamples,'s_SNR',s_SNR);
            m_samples=zeros(v_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
            m_positions=zeros(v_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
            for s_timeInd=1:s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:(s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*v_numberOfSamples+1:(s_timeInd)*v_numberOfSamples;
                
                [m_samples(v_timetIndicesForSamples,:),...
                    m_positions(v_timetIndicesForSamples,:)]...
                    = sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
            end
            tic
            %% 5. batch estimate
            nonParametricBatchGraphFunctionEstimator=NonParametricBatchGraphFunctionEstimator...
                ('m_kernels',inv(m_kernelInv),...
                's_mu',s_mu,'s_maximumTime',s_maximumTime);
            m_batchEstimate=nonParametricBatchGraphFunctionEstimator.estimate...
                (m_samples,m_positions,v_numberOfSamples);
            batchTime=toc
            %% 6. kalman estimate
            tic
            m_kfEstimate=zeros(size(m_batchEstimate));
            kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                't_previousMinimumSquaredError',t_initialSigma0,...
                'm_previousEstimate',m_initialState);
            for s_timeInd=1:s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:(s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*v_numberOfSamples+1:(s_timeInd)*v_numberOfSamples;
                
                %samples and positions at time t
                m_samplest=m_samples(v_timetIndicesForSamples,:);
                m_positionst=m_positions(v_timetIndicesForSamples,:);
                %estimate
                
                [m_kfEstimate(v_timetIndicesForSignals,:),t_newMSE]=...
                    kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,t_transitions(:,:,s_timeInd),...
                    t_correlations(:,:,s_timeInd),v_sigma(s_timeInd));
                % prepare KF for next iteration
                kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                kFOnGFunctionEstimator.m_previousEstimate=m_kfEstimate(v_timetIndicesForSignals,:);
                
            end
            kkfTime=toc
            %% 7. measure difference
            v_difBatchKF=(m_batchEstimate(v_timetIndicesForSignals,:)...
                -m_kfEstimate(v_timetIndicesForSignals,:)...
                );
            toc
            s_relativeErrorBatch=norm(m_batchEstimate-m_graphFunction,'fro')/norm(m_graphFunction,'fro')
            
            s_relativeErrorKF=norm(m_kfEstimate-m_graphFunction,'fro')/norm(m_graphFunction,'fro')
            %% generate figure
            F=[];
        end
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare Batch Approach up to time t
        %  with Bandlimited model at each time t
        %  Kernel used: inverse Laplacian
        function F = compute_fig_2002(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=24;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=144;
            s_mu=10^-6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.4:1);
            v_bandwidthPercentage=0.01;
            s_stepLMS=0.6;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=1.0; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                zeros(s_numberOfVertices,s_numberOfVertices,s_maximumTime);
            t_timeAdjacencyAtDifferentTimes...
                =zeros(s_numberOfVertices,s_numberOfVertices,s_maximumTime-1);
            for s_timeInd=1:s_maximumTime
                t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd)=m_adjacency;
                if(s_timeInd~=s_maximumTime)
                    t_timeAdjacencyAtDifferentTimes(:,:,s_timeInd)=v_propagationWeight*eye(size(m_adjacency));
                end
            end
            
            graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            graphT=graphGenerator.realization;
            
            %% 2. choise of Kernel must be positive definite
            % inverse Laplacian
            s_beta=1; % ensures that Kernel is positive definite
            m_kernelInv=graphT.getLaplacian()+s_beta*eye(graphT.getNumberOfVertices);
            [~,p1]=chol(m_kernelInv);%check if PD
            [~,p2]=chol(m_kernelInv((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
                s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
                *s_numberOfVertices));
            if (p1+p2~=0)
                assert(1==0,'Not Positive Definite Kernel')
            end
            
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            t_batchEstimate=zeros(size(m_graphFunction,1),size(m_graphFunction,2)...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(size(m_graphFunction,1),size(m_graphFunction,2)...
                ,size(v_numberOfSamples,2));
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:(s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:(s_timeInd)*s_numberOfSamples;
                    
                    [m_samples(v_timetIndicesForSamples,:),...
                        m_positions(v_timetIndicesForSamples,:)]...
                        = sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
                end
                
                %% 5. batch estimate
                nonParametricBatchGraphFunctionEstimator=NonParametricBatchGraphFunctionEstimator...
                    ('m_kernels',inv(m_kernelInv),...
                    's_mu',s_mu,'s_maximumTime',s_maximumTime);
                t_batchEstimate(:,:,s_sampleInd)=nonParametricBatchGraphFunctionEstimator.estimate...
                    (m_samples,m_positions,s_numberOfSamples);
                
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
                m_extendedAdjancency=graphT.m_adjacency;
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:(s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:(s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %create take diagonals from extended graph
                    m_adjacency=m_extendedAdjancency(v_timetIndicesForSignals,v_timetIndicesForSignals);
                    grapht=Graph('m_adjacency',m_adjacency);
                    
                    %bandlimited estimate
                    bandlimitedGraphFunctionEstimator= BandlimitedGraphFunctionEstimator('m_laplacian'...
                        ,grapht.getLaplacian,'s_bandwidth',v_bandwidth);
                    t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                    
                end
                v_relativeErrorBatch(s_sampleInd)=norm(t_batchEstimate(:,:,s_sampleInd)...
                    -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                
                v_relativeErrorbandLimitedEstimate(s_sampleInd)...
                    =norm(t_bandLimitedEstimate(:,:,s_sampleInd)-m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                
            end
            %% 7. measure difference
            
            
            %% generate figure
            F=[];
        end
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare Batch Approach up to time t
        %  with Bandlimited model at each time t
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_2003(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=40;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=2;
            s_mu=10^-6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.2:1);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=1; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(v_propagationWeight*eye(size(m_adjacency)),[1,1,s_maximumTime-1]);
            
            graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            graphT=graphGenerator.realization;
            
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            s_sigma=0.4;
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigma,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime)
            extendedGraphKernel = ExtendedGraphKernel...
                ('t_invSpatialKernel',t_invDiffusionKernel,...
                't_invTemporalKernel',-t_timeAdjacencyAtDifferentTimes);
            
            m_extendedGraphKernel=extendedGraphKernel.generateKernelMatrix;
            % make kernel great again
            s_beta=1;
            m_extendedGraphKernel=m_extendedGraphKernel+s_beta*eye(size(m_extendedGraphKernel));
            [~,p1]=chol(m_extendedGraphKernel);%check if PD
            [~,p2]=chol(m_extendedGraphKernel((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
                s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
                *s_numberOfVertices));
            if (p1+p2~=0)
                assert(1==0,'Not Positive Definite Kernel')
            end
            
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    [m_samples(v_timetIndicesForSamples,:),...
                        m_positions(v_timetIndicesForSamples,:)]...
                        = sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
                end
                
                %% 5. batch estimate
                nonParametricBatchGraphFunctionEstimator=NonParametricBatchGraphFunctionEstimator...
                    ('m_kernels',m_extendedGraphKernel,...
                    's_mu',s_mu,'s_maximumTime',s_maximumTime);
                t_batchEstimate(:,:,s_sampleInd)=nonParametricBatchGraphFunctionEstimator.estimate...
                    (m_samples,m_positions,s_numberOfSamples);
                
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                myLegend={};
                v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_extendedAdjancency=graphT.m_adjacency;
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=m_extendedAdjancency(v_timetIndicesForSignals,...
                            v_timetIndicesForSignals);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    v_relativeErrorBatch(s_sampleInd)=norm(t_batchEstimate(:,:,s_sampleInd)...
                        -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                    
                    m_relativeErrorbandLimitedEstimate(s_sampleInd,s_bandInd)...
                        =norm(t_bandLimitedEstimate(:,:,s_sampleInd,s_bandInd)...
                        -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                    
                    myLegend{s_bandInd}=strcat('Bandlimited  ',...
                        sprintf(' W=%g',s_bandwidth));
                end
            end
            %% 7. measure difference
            
            myLegend{s_bandInd+1}='Batch approach';
            F = F_figure('X',v_numberOfSamples,'Y',[m_relativeErrorbandLimitedEstimate,...
                v_relativeErrorBatch']',...
                'xlab','Number of observed vertices (S)','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
        end
        function F = compute_fig_2203(obj,niter)
            F = obj.load_F_structure(2003);
            F.ylimit=[0 2];
            %F.xlimit=[10 100];
            F.styles = {'-','--','-o','-x','--^','--','--'};
            F.pos=[680 729 509 249];
            F.tit='Temperature across contiguous USA'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare Kalman filter up to time t
        %  with Bandlimited model at each time t
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_2004(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=4;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=2;
            s_mu=10^-6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.2:1);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=1; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(v_propagationWeight*eye(size(m_adjacency)),[1,1,s_maximumTime-1]);
            
            graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            graphT=graphGenerator.realization;
            
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            s_sigma=0.4;
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigma,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime)
            m_invExtendedKernel=KFonGSimulations.createInvExtendedGraphKernel...
                (t_invDiffusionKernel,-t_timeAdjacencyAtDifferentTimes);
            
            % make kernel great again
            s_beta=0;
            m_invExtendedKernel=m_invExtendedKernel+s_beta*eye(size(m_invExtendedKernel));
            [~,p1]=chol(m_invExtendedKernel);%check if PD
            [~,p2]=chol(m_invExtendedKernel((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
                s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
                *s_numberOfVertices));
            if (p1+p2~=0)
                assert(1==0,'Not Positive Definite Kernel')
            end
            t_eyeTens=repmat(eye(size(m_diffusionKernel)),[1,1,s_maximumTime]);
            t_invDiffusionKernel=t_invDiffusionKernel+s_beta*t_eyeTens;
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invDiffusionKernel,-t_timeAdjacencyAtDifferentTimes,m_invExtendedKernel...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    [m_samples(v_timetIndicesForSamples,:),...
                        m_positions(v_timetIndicesForSamples,:)]...
                        = sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                
                v_relativeErrorKf(s_sampleInd)=norm(t_kfEstimate(:,:,s_sampleInd)...
                    -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                myLegend={};
                v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_extendedAdjancency=graphT.m_adjacency;
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=m_extendedAdjancency(v_timetIndicesForSignals,...
                            v_timetIndicesForSignals);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                    m_relativeErrorbandLimitedEstimate(s_sampleInd,s_bandInd)...
                        =norm(t_bandLimitedEstimate(:,:,s_sampleInd,s_bandInd)...
                        -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                    
                    myLegend{s_bandInd}=strcat('Bandlimited  ',...
                        sprintf(' W=%g',s_bandwidth));
                end
            end
            %% 7. measure difference
            
            myLegend{s_bandInd+1}='Batch approach';
            F = F_figure('X',v_numberOfSamples,'Y',[m_relativeErrorbandLimitedEstimate,...
                v_relativeErrorKf']',...
                'xlab','Number of observed vertices (S)','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
        end
        function F = compute_fig_2204(obj,niter)
            F = obj.load_F_structure(2004);
            F.ylimit=[0 2];
            %F.xlimit=[10 100];
            F.styles = {'-','--','-o','-x','--^','--','--'};
            F.pos=[680 729 509 249];
            F.tit='Temperature across contiguous USA'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare Kalman filter up to time t Plot reconstruction
        %  error as increase the sampling set
        %  with Bandlimited model at each time t and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        %TODO why error increases
        function F = compute_fig_2005(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=1000;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-14;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.2:1);
            v_bandwidthPercentage=(0.01:0.01:0.01);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=1; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(v_propagationWeight*eye(size(m_adjacency)),[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            s_sigma=0.4;
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigma,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime)
            % 			m_invExtendedKernel=KFonGSimulations.createInvExtendedGraphKernel...
            % 				(t_invDiffusionKernel,-t_timeAdjacencyAtDifferentTimes);
            %
            % make kernel great again
            % 			s_beta=0;
            % 			m_invExtendedKernel=m_invExtendedKernel+s_beta*eye(size(m_invExtendedKernel));
            % 			[~,p1]=chol(m_invExtendedKernel);%check if PD
            % 			[~,p2]=chol(m_invExtendedKernel((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
            % 				s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
            % 				*s_numberOfVertices));
            % 			if (p1+p2~=0)
            % 				assert(1==0,'Not Positive Definite Kernel')
            % 			end
            % 			t_eyeTens=repmat(eye(size(m_diffusionKernel)),[1,1,s_maximumTime]);
            % 			t_invDiffusionKernel=t_invDiffusionKernel+s_beta*t_eyeTens;
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invDiffusionKernel,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            m_relativeErrorDistr=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            v_relativeErrorKf=zeros(size(v_numberOfSamples,2),1);
            m_relativeErrorbandLimitedEstimate=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samples(v_timetIndicesForSamples,:)=m_samples(1:s_numberOfSamples,:);
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                
                v_relativeErrorKf(s_sampleInd)=norm(t_kfEstimate(:,:,s_sampleInd)...
                    -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    m_relativeErrorbandLimitedEstimate(s_sampleInd,s_bandInd)...
                        =norm(t_bandLimitedEstimate(:,:,s_sampleInd,s_bandInd)...
                        -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                    
                    myLegend{s_bandInd}=strcat('Bandlimited  ',...
                        sprintf(' W=%g',s_bandwidth));
                end
                
                %% 7.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    m_relativeErrorDistr(s_sampleInd,s_bandInd)=...
                        norm( t_distrEstimate(:,:,s_sampleInd,s_bandInd)...
                        -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                    myLegend{s_bandInd+size(v_bandwidth,2)}=strcat('DLSR, ',...
                        sprintf(' W=%g',s_bandwidth));
                    
                end
            end
            
            
            %% 8. measure difference
            
            
            myLegend{s_bandInd+size(v_bandwidth,2)+1}='Kernel Kalman Filter';
            F = F_figure('X',v_numberOfSamples,'Y',[m_relativeErrorbandLimitedEstimate...
                ,m_relativeErrorDistr,...
                v_relativeErrorKf]',...
                'xlab','Number of observed vertices (S)','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
        end
        function F = compute_fig_2205(obj,niter)
            F = obj.load_F_structure(2005);
            F.ylimit=[0 2];
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Temperature across contiguous USA'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare Kalman filter up to time t reconstruction error
        %  measured seperately at each time, summed and averaged.
        %  Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model at each time t and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        %TODO why error increases
        function F = compute_fig_2006(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=8000;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-8;
            s_sigmaForDiffusion=1.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.1:0.1);
            v_bandwidthPercentage=(0.01:0.01:0.01);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=1; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_timeAuxMatrix=...
                repmat( diag(sum(m_timeAdjacency+m_timeAdjacency')),[1,1,s_maximumTime]);
            t_timeAuxMatrix(:,:,1)=t_timeAuxMatrix(:,:,1)-diag(sum(m_timeAdjacency));
            t_timeAuxMatrix(:,:,s_maximumTime)=t_timeAuxMatrix(:,:,s_maximumTime)-diag(sum(m_timeAdjacency'));
            t_invSpatialDiffusionKernel=repmat(inv(m_diffusionKernel),[1,1,s_maximumTime]);
            t_invSpatialDiffusionKernel=t_invSpatialDiffusionKernel+t_timeAuxMatrix;
            % 			m_invExtendedKernel=KFonGSimulations.createInvExtendedGraphKernel...
            % 				(t_invDiffusionKernel,-t_timeAdjacencyAtDifferentTimes);
            %
            % make kernel great again
            % 			s_beta=0;
            % 			m_invExtendedKernel=m_invExtendedKernel+s_beta*eye(size(m_invExtendedKernel));
            % 			[~,p1]=chol(m_invExtendedKernel);%check if PD
            % 			[~,p2]=chol(m_invExtendedKernel((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
            % 				s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
            % 				*s_numberOfVertices));
            % 			if (p1+p2~=0)
            % 				assert(1==0,'Not Positive Definite Kernel')
            % 			end
            % 			t_eyeTens=repmat(eye(size(m_diffusionKernel)),[1,1,s_maximumTime]);
            % 			t_invDiffusionKernel=t_invDiffusionKernel+s_beta*t_eyeTens;
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samples(v_timetIndicesForSamples,:)=m_samples(1:s_numberOfSamples,:);
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 7.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
            end
            
            
            %% 8. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            for s_sampleInd=1:size(v_numberOfSamples,2)
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    if s_timeInd>1
                        s_prevTimeInd=s_timeInd-1;
                    else
                        s_prevTimeInd=s_timeInd;
                    end
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =m_relativeErrorKf(s_prevTimeInd, s_sampleInd)+...
                        norm(t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd)...
                        -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                        norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                    for s_bandInd=1:size(v_bandwidth,2)
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            m_relativeErrorDistr(s_prevTimeInd,(s_sampleInd-1)*...
                            size(v_bandwidth,2)+s_bandInd)+...
                            norm( t_distrEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)...
                            -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                            norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            =m_relativeErrorbandLimitedEstimate(s_prevTimeInd,...
                            (s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            +norm(t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)...
                            -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                            norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                        
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('Bandlimited, ',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                    end
                    myLegendKF{s_sampleInd}=strcat('Kernel Kalman Filter, ',...
                        sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                    
                end
            end
            %normalize errors
            m_normalizer=repmat((1:s_maximumTime)',1,size(m_relativeErrorKf,2));
            m_relativeErrorKf=m_relativeErrorKf./m_normalizer;
            m_normalizer=repmat((1:s_maximumTime)',1,size(m_relativeErrorbandLimitedEstimate,2));
            m_relativeErrorbandLimitedEstimate=m_relativeErrorbandLimitedEstimate./m_normalizer;
            m_normalizer=repmat((1:s_maximumTime)',1,size(m_relativeErrorDistr,2));
            m_relativeErrorDistr=m_relativeErrorDistr./m_normalizer;
            myLegend=[myLegendDLSR myLegendBan myLegendKF];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorbandLimitedEstimate,...
                m_relativeErrorKf]',...
                'xlab','Time evolution','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
        end
        function F = compute_fig_2206(obj,niter)
            F = obj.load_F_structure(2006);
            F.ylimit=[0 2];
            F.logy = 1;
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Temperature across contiguous USA'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare Kalman filter up to time t Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model at each time t and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        %TODO why error increases
        function F = compute_fig_2007(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=500;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-8;
            s_sigmaForDiffusion=1.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.2:0.1:0.3);
            v_bandwidthPercentage=(0.01:0.02:0.05);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=1; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_timeAuxMatrix=...
                repmat( diag(sum(m_timeAdjacency+m_timeAdjacency')),[1,1,s_maximumTime]);
            t_timeAuxMatrix(:,:,1)=t_timeAuxMatrix(:,:,1)-diag(sum(m_timeAdjacency));
            t_timeAuxMatrix(:,:,s_maximumTime)=t_timeAuxMatrix(:,:,s_maximumTime)-diag(sum(m_timeAdjacency'));
            t_invSpatialDiffusionKernel=repmat(inv(m_diffusionKernel),[1,1,s_maximumTime]);
            t_invSpatialDiffusionKernel=t_invSpatialDiffusionKernel+t_timeAuxMatrix;
            % 			m_invExtendedKernel=KFonGSimulations.createInvExtendedGraphKernel...
            % 				(t_invDiffusionKernel,-t_timeAdjacencyAtDifferentTimes);
            %
            % make kernel great again
            % 			s_beta=0;
            % 			m_invExtendedKernel=m_invExtendedKernel+s_beta*eye(size(m_invExtendedKernel));
            % 			[~,p1]=chol(m_invExtendedKernel);%check if PD
            % 			[~,p2]=chol(m_invExtendedKernel((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
            % 				s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
            % 				*s_numberOfVertices));
            % 			if (p1+p2~=0)
            % 				assert(1==0,'Not Positive Definite Kernel')
            % 			end
            % 			t_eyeTens=repmat(eye(size(m_diffusionKernel)),[1,1,s_maximumTime]);
            % 			t_invDiffusionKernel=t_invDiffusionKernel+s_beta*t_eyeTens;
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samples(v_timetIndicesForSamples,:)=m_samples(1:s_numberOfSamples,:);
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 7.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
            end
            
            
            %% 8. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            for s_sampleInd=1:size(v_numberOfSamples,2)
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =norm(t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd)...
                        -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                        norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                    for s_bandInd=1:size(v_bandwidth,2)
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            norm( t_distrEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)...
                            -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                            norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            =norm(t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)...
                            -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                            norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                        
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('Bandlimited, ',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                    end
                    myLegendKF{s_sampleInd}=strcat('Kernel Kalman Filter, ',...
                        sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                    
                end
            end
            myLegend=[myLegendDLSR myLegendBan myLegendKF];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorbandLimitedEstimate,...
                m_relativeErrorKf]',...
                'xlab','Time evolution','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
        end
        function F = compute_fig_2207(obj,niter)
            F = obj.load_F_structure(2007);
            F.ylimit=[0 2];
            F.logy = 1;
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Temperature across contiguous USA'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare Kalman filter up to time t Plot  reconstruct error
        %  as time evolves
        %  with Bandlimited model at each time t and WangWangGuo paper and
        %  batch approach
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_2008(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=10;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=2;
            s_mu=10^-5;
            s_sigma=1;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.1:0.1);
            v_bandwidthPercentage=(0.01:0.01:0.01);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=1; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigma,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_timeAuxMatrix=...
                repmat( diag(sum(m_timeAdjacency+m_timeAdjacency')),[1,1,s_maximumTime]);
            t_timeAuxMatrix(:,:,1)=t_timeAuxMatrix(:,:,1)-diag(sum(m_timeAdjacency));
            t_timeAuxMatrix(:,:,s_maximumTime)=t_timeAuxMatrix(:,:,s_maximumTime)-diag(sum(m_timeAdjacency'));
            t_invSpatialDiffusionKernel=repmat(inv(m_diffusionKernel),[1,1,s_maximumTime]);
            t_invSpatialDiffusionKernel=t_invSpatialDiffusionKernel+t_timeAuxMatrix;
            m_invExtendedKernel=KFonGSimulations.createInvExtendedGraphKernel...
                (t_invSpatialDiffusionKernel,-t_timeAdjacencyAtDifferentTimes);
            
            %make kernel great again
            s_beta=0;
            m_invExtendedKernel=m_invExtendedKernel+s_beta*eye(size(m_invExtendedKernel));
            [~,p1]=chol(m_invExtendedKernel);%check if PD
            [~,p2]=chol(m_invExtendedKernel((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
                s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
                *s_numberOfVertices));
            if (p1+p2~=0)
                assert(1==0,'Not Positive Definite Kernel')
            end
            t_eyeTens=repmat(eye(size(m_diffusionKernel)),[1,1,s_maximumTime]);
            t_invSpatialDiffusionKernel=t_invSpatialDiffusionKernel+s_beta*t_eyeTens;
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_batchEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samples(v_timetIndicesForSamples,:)=m_samples(1:s_numberOfSamples,:);
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 7.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                end
                
                %% 8. batch estimate
                nonParametricBatchGraphFunctionEstimator=NonParametricBatchGraphFunctionEstimator...
                    ('m_kernels',inv(m_invExtendedKernel),...
                    's_mu',s_mu,'s_maximumTime',s_maximumTime);
                t_batchEstimate(:,:,s_sampleInd)=nonParametricBatchGraphFunctionEstimator.estimate...
                    (m_samples,m_positions,v_numberOfSamples);
            end
            
            
            %% 9. measure difference
            
            t_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2),size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeBatch=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            t_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2),size(v_bandwidth,2));
            for s_sampleInd=1:size(v_numberOfSamples,2)
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =norm(t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd)...
                        -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                        norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                    m_relativeBatch(s_timeInd, s_sampleInd)...
                        =norm(t_batchEstimate(v_timetIndicesForSignals,:,s_sampleInd)...
                        -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                        norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                    for s_bandInd=1:size(v_bandwidth,2)
                        s_bandwidth=v_bandwidth(s_bandInd);
                        t_relativeErrorDistr(s_timeInd,s_sampleInd,s_bandInd)=...
                            norm( t_distrEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)...
                            -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                            norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                        t_relativeErrorbandLimitedEstimate(s_timeInd,s_sampleInd,s_bandInd)...
                            =norm(t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)...
                            -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                            norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                        
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('Bandlimited, ',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                    end
                    myLegendKF{s_sampleInd}=strcat('Kernel Kalman Filter, ',...
                        sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                    myLegendBatch{s_sampleInd}=strcat('Kernel Ridge Regression, ',...
                        sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                    
                end
            end
            myLegend=[myLegendDLSR myLegendBan myLegendKF myLegendBatch];
            F = F_figure('X',(1:s_maximumTime),'Y',[t_relativeErrorDistr(:,:,1)...
                ,t_relativeErrorbandLimitedEstimate(:,:,1),...
                m_relativeErrorKf,m_relativeBatch]',...
                'xlab','Time evolution','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
        end
        function F = compute_fig_2208(obj,niter)
            F = obj.load_F_structure(2008);
            F.ylimit=[0 2];
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Temperature across contiguous USA'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare perfomance of Kalman filter up to time t as I
        %  change the parameters diffusion parameter reg parameter and
        %  weight of scaled Identity as time evolves
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_2009(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=360 ;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=24;
            v_mu=[10^-7]; %opt
            v_sigmaForDiffusion=[1.4]; %opt
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.2:0.2:0.2);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=[0,0.001,0.01,10,20,30]; % weight of edges between the same node opt
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            %
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            t_kfEstimate=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,size(v_numberOfSamples,2),s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_relativeErrorKf=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,s_maximumTime,size(v_numberOfSamples,2));
            v_allPositions=(1:s_numberOfVertices)';
            
            for s_scalingInd=1:size(v_propagationWeight,2)
                s_propagationWeight=v_propagationWeight(s_scalingInd);
                m_timeAdjacency=s_propagationWeight*(eye(size(m_adjacency)));
                
                t_timeAdjacencyAtDifferentTimes=...
                    repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
                
                % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
                % 			graphT=graphGenerator.realization;
                %
                %% 2. choise of Kernel must be positive definite
                % diffusion kernel
                graph=Graph('m_adjacency',m_adjacency);
                for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                    diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_sigmaInd),...
                        'm_laplacian',graph.getLaplacian);
                    m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
                    %m_diffusionKernel=m_diffusionKernel/max(
                    %check expression again
                    t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
                    
                    %% generate transition, correlation matrices
                    m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
                    m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
                    t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
                    for s_ind=1:s_monteCarloSimulations
                        t_sigma0(:,:,s_ind)=m_sigma0;
                    end
                    [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                        (t_invSpatialDiffusionKernel...
                        ,-t_timeAdjacencyAtDifferentTimes...
                        ,s_maximumTime,s_numberOfVertices,m_sigma0);
                    
                    %% 3. generate true signal
                    
                    m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
                    
                    m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
                    
                    
                    %%
                    
                    
                    
                    for s_sampleInd=1:size(v_numberOfSamples,2)
                        %% 4. generate observations
                        s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                        sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                        m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        %Same sample locations needed for distributed algo
                        [m_samples(1:s_numberOfSamples,:),...
                            m_positions(1:s_numberOfSamples,:)]...
                            = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                        for s_timeInd=2:s_maximumTime
                            %time t indices
                            v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                (s_timeInd)*s_numberOfVertices;
                            v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                (s_timeInd)*s_numberOfSamples;
                            m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                            for s_mtId=1:s_monteCarloSimulations
                                m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                                    ((s_timeInd-1)*s_numberOfVertices+...
                                    m_positions(v_timetIndicesForSamples,s_mtId));
                            end
                            
                        end
                        
                        %% 5. KF estimate
                        kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                            't_previousMinimumSquaredError',t_initialSigma0,...
                            'm_previousEstimate',m_initialState);
                        for s_muInd=1:size(v_mu,2)
                            v_sigmaForKalman=sqrt((1:s_maximumTime)'*v_numberOfSamples*v_mu(s_muInd))';
                            v_normOFKFErrors=zeros(s_maximumTime,1);
                            v_normOFNotSampled=zeros(s_maximumTime,1);
                            for s_timeInd=1:s_maximumTime
                                %time t indices
                                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                    (s_timeInd)*s_numberOfVertices;
                                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                    (s_timeInd)*s_numberOfSamples;
                                
                                %samples and positions at time t
                                m_samplest=m_samples(v_timetIndicesForSamples,:);
                                m_positionst=m_positions(v_timetIndicesForSamples,:);
                                %estimate
                                
                                [m_prevEstimate,t_newMSE]=...
                                    kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                                    t_transitions(:,:,s_timeInd),...
                                    t_correlations(:,:,s_timeInd),v_sigmaForKalman(s_sampleInd,s_timeInd));
                                t_kfEstimate(s_scalingInd,s_sigmaInd,...
                                    s_muInd,s_sampleInd,v_timetIndicesForSignals,:)=m_prevEstimate;
                                % prepare KF for next iteration
                                kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                                kFOnGFunctionEstimator.m_previousEstimate=m_prevEstimate;
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                for s_mtind=1:s_monteCarloSimulations
                                    v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                                    v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                                        norm(squeeze(t_kfEstimate(s_scalingInd,s_sigmaInd,s_muInd,s_sampleInd,v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                        ,s_mtind))...
                                        -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                    v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                                        norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                end
                                
                                s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                                t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_timeInd, s_sampleInd)...
                                    =sum(v_normOFKFErrors(1:s_timeInd))/...
                                    s_summedNorm;
                                
                                
                                
                                myLegendKF{s_scalingInd}...
                                    =...
                                    sprintf('KKF scaling=%g',v_propagationWeight(s_scalingInd));
                                
                                
                            end
                        end
                        
                    end
                end
            end
            
            
            
            plot(squeeze(t_kfEstimate(1,1,1,1,(0:s_maximumTime-1)*s_numberOfVertices+1,1)))
            m_relativeErrorKf=squeeze(t_relativeErrorKf);
            m_relativeErrorKf=reshape(t_relativeErrorKf,[size(v_propagationWeight,2)*...
                size(v_sigmaForDiffusion,2)*size(v_mu,2)*size(v_numberOfSamples,2),s_maximumTime,]);
            myLegendKF=squeeze(myLegendKF);
            myLegendKF=reshape(myLegendKF,[1,size(v_propagationWeight,2)*...
                size(v_sigmaForDiffusion,2)*size(v_mu,2)*size(v_numberOfSamples,2)]);
            myLegend=[myLegendKF];
            F = F_figure('X',(1:s_maximumTime),'Y',m_relativeErrorKf,...
                'xlab','Time evolution','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
            %F.ylimit=[0 0.5];
            F.caption=[	sprintf('regularization parameter mu=%g\n',v_mu),...
                sprintf(' diffusion parameter sigma=%g\n',v_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)];
        end
        function F = compute_fig_2209(obj,niter)
            F = obj.load_F_structure(2009);
            F.ylimit=[0.1 0.4];
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Parameter selection'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare Kalman filter up to time t reconstruction error
        %  measured seperately at each time, summed and normalized.
        %  Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model, Kernel Rigde regression
        %  at each time t and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        
        function F = compute_fig_2010(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=2;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-8;
            s_sigmaForDiffusion=1.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.1:0.1);
            v_bandwidthPercentage=(0.01:0.01:0.01);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=1;
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            
            m_timeAdjacency=m_adjacency+v_propagationWeight*eye(size(m_adjacency));
            
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialKernel=repmat(inv(m_diffusionKernel)+...
                diag(sum(m_timeAdjacency)),[1,1,s_maximumTime]);
            % 			t_invTemporalKernel=repmat(diag(sum(m_timeAdjacency))-m_timeAdjacency...
            % 				,[1,1,s_maximumTime-1]);
            m_invExtendedKernel=KFonGSimulations.createInvExtendedGraphKernel...
                (t_invSpatialKernel,-t_timeAdjacencyAtDifferentTimes);
            
            %  % % make kernel great again
            s_beta=100;
            m_invExtendedKernel=m_invExtendedKernel+s_beta*eye(size(m_invExtendedKernel));
            [~,p1]=chol(m_invExtendedKernel);%check if PD
            [~,p2]=chol(m_invExtendedKernel((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
                s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
                *s_numberOfVertices));
            if (p1+p2~=0)
                assert(1==0,'Not Positive Definite Kernel')
            end
            t_eyeTens=repmat(eye(size(m_diffusionKernel)),[1,1,s_maximumTime]);
            t_invSpatialKernel=t_invSpatialKernel+s_beta*t_eyeTens;
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samples(v_timetIndicesForSamples,:)=m_samples(1:s_numberOfSamples,:);
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 7.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
            end
            
            
            %% 8. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            for s_sampleInd=1:size(v_numberOfSamples,2)
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    if s_timeInd>1
                        s_prevTimeInd=s_timeInd-1;
                    else
                        s_prevTimeInd=s_timeInd;
                    end
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =m_relativeErrorKf(s_prevTimeInd, s_sampleInd)+...
                        norm(t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd)...
                        -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                        norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                    for s_bandInd=1:size(v_bandwidth,2)
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            m_relativeErrorDistr(s_prevTimeInd,(s_sampleInd-1)*...
                            size(v_bandwidth,2)+s_bandInd)+...
                            norm( t_distrEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)...
                            -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                            norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            =m_relativeErrorbandLimitedEstimate(s_prevTimeInd,...
                            (s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            +norm(t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)...
                            -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                            norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                        
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('Bandlimited, ',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                    end
                    myLegendKF{s_sampleInd}=strcat('Kernel Kalman Filter, ',...
                        sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                    
                end
            end
            %normalize errors
            m_normalizer=repmat((1:s_maximumTime)',1,size(m_relativeErrorKf,2));
            m_relativeErrorKf=m_relativeErrorKf./m_normalizer;
            m_normalizer=repmat((1:s_maximumTime)',1,size(m_relativeErrorbandLimitedEstimate,2));
            m_relativeErrorbandLimitedEstimate=m_relativeErrorbandLimitedEstimate./m_normalizer;
            m_normalizer=repmat((1:s_maximumTime)',1,size(m_relativeErrorDistr,2));
            m_relativeErrorDistr=m_relativeErrorDistr./m_normalizer;
            myLegend=[myLegendDLSR myLegendBan myLegendKF];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorbandLimitedEstimate,...
                m_relativeErrorKf]',...
                'xlab','Time evolution','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
        end
        function F = compute_fig_2210(obj,niter)
            F = obj.load_F_structure(2010);
            F.ylimit=[0 2];
            F.logy = 1;
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Temperature across contiguous USA'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare Kalman filter up to time t reconstruction error
        %  measured seperately at each time, summed and averaged.
        %  Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model at each time t and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        %TODO why error increases
        function F = compute_fig_2011(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=100;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=10;
            s_mu=10^-8;
            s_sigmaForDiffusion=1.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.1:0.1);
            v_bandwidthPercentage=(0.01:0.01:0.01);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=1; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            % data normalization
            v_mean = mean(m_temperatureTimeSeries,2);
            v_std = std(m_temperatureTimeSeries')';
            % 			m_temperatureTimeSeries = diag(1./v_std)*(m_temperatureTimeSeries...
            %                 - v_mean*ones(1,size(m_temperatureTimeSeries,2)));
            
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samples(v_timetIndicesForSamples,:)=m_samples(1:s_numberOfSamples,:);
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
            end
            
            
            %% 9. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            for s_sampleInd=1:size(v_numberOfSamples,2)
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    if s_timeInd>1
                        s_prevTimeInd=s_timeInd-1;
                    else
                        s_prevTimeInd=s_timeInd;
                    end
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =...
                        norm(t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd)...
                        -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                        norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                        =...
                        norm(t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)...
                        -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                        norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidth,2)
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            norm( t_distrEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)...
                            -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                            norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');%s_timeInd*s_numberOfVertices;
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            =norm(t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)...
                            -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                            norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');%s_timeInd*s_numberOfVertices;
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                        
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('Bandlimited, ',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                    end
                    myLegendKF{s_sampleInd}=strcat('Kernel Kalman Filter, ',...
                        sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                    myLegendKRR{s_sampleInd}=strcat('KRR time agnostic, ',...
                        sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendBan myLegendKF myLegendKRR];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorbandLimitedEstimate,...
                m_relativeErrorKf, m_relativeErrorKRR]',...
                'xlab','Time evolution','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
        end
        function F = compute_fig_2211(obj,niter)
            F = obj.load_F_structure(2011);
            F.ylimit=[0 2];
            F.logy = 1;
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Temperature across contiguous USA'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare Kalman filter up to time t reconstruction error
        %  measured seperately on the unobserved at each time, summed and normalize.
        %  Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model at each time t and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_2012(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=100;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=24;
            s_mu=10^-7;
            s_sigmaForDiffusion=1.4;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.2:0.2:0.2);
            %v_bandwidthPercentage=[0.01,0.1];
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=0.01; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2,4];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            % data normalization
            v_mean = mean(m_temperatureTimeSeries,2);
            v_std = std(m_temperatureTimeSeries')';
            % 			m_temperatureTimeSeries = diag(1./v_std)*(m_temperatureTimeSeries...
            %                 - v_mean*ones(1,size(m_temperatureTimeSeries,2)));
            
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
            end
            
            
            %% 9. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            v_normOFKFErrors=zeros(s_maximumTime,1);
            v_normOFNotSampled=zeros(s_maximumTime,1);
            v_normOfKrrErrors=zeros(s_maximumTime,1);
            m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
            m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOFKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrrErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' W=%g',s_bandwidth));
                        
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('BL-TA, ',...
                            sprintf(' W=%g',s_bandwidth));
                    end
                    myLegendKF{s_sampleInd}='KKF ';
                    myLegendKRR{s_sampleInd}='KRR-TA';
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendBan myLegendKF myLegendKRR];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorbandLimitedEstimate,...
                m_relativeErrorKf, m_relativeErrorKRR]',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F.ylimit=[0 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)];
        end
        function F = compute_fig_2212(obj,niter)
            F = obj.load_F_structure(2012);
            F.ylimit=[0 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Temperature across contiguous USA'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare perfomance of Kalman filter DLSR Bandlimited and
        %  KRR agnostic up to time t as I
        %  on tracking the signal.
        function F = compute_fig_2013(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=100;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            s_sigmaForDiffusion=1.4;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.1:0.1);
            v_bandwidthPercentage=[0.01];
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=0.1; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*( eye(size(m_adjacency)));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            % data normalization
            v_mean = mean(m_temperatureTimeSeries,2);
            v_std = std(m_temperatureTimeSeries')';
            % 			m_temperatureTimeSeries = diag(1./v_std)*(m_temperatureTimeSeries...
            %                 - v_mean*ones(1,size(m_temperatureTimeSeries,2)));
            
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %m_diffusionKernel=graph.getLaplacian();
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                    
                end
                myLegendKF{s_sampleInd}='KKF';
                
                %% 6. Kernel Ridge Regression
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                    
                end
                myLegendKRR{s_sampleInd}='KRR-TA';
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                        
                    end
                    
                    
                    myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                        'BL-TA'
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}='DLSR';
                end
                
            end
            
            for s_vertInd=1:s_numberOfVertices
                
                
                m_meanEstKF(s_vertInd,:)=mean(t_kfEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,1,:),2)';
                m_meanEstKRR(s_vertInd,:)=mean(t_kRRestimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,1,:),2)';
                m_meanEstDLSR(s_vertInd,:)=mean(t_distrEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,:,1,1),2)';
                m_meanEstBan(s_vertInd,:)=mean(t_bandLimitedEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,:,1,1),2)';
                
            end
            %% 9. measure difference
            %normalize errors
            myLegandTrueSignal{1}='True Signal';
            s_vertexToPlot=1;
            myLegend=[myLegandTrueSignal myLegendKF  myLegendKRR  myLegendDLSR myLegendBan];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_temperatureTimeSeriesSampled(s_vertexToPlot,:);...
                m_meanEstKF(s_vertexToPlot,:);m_meanEstKRR(s_vertexToPlot,:);m_meanEstDLSR(s_vertexToPlot,:);...
                m_meanEstBan(s_vertexToPlot,:)],...
                'xlab','Time evolution','ylab','function value','leg',myLegend);
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('number of samples =%g\n',v_numberOfSamples)];
            
        end
        function F = compute_fig_2213(obj,niter)
            F = obj.load_F_structure(2013);
            %F.ylimit=[0 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Temperature tracking'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare perfomance of Kalman filter up to time t as I
        %  on tracking the signal.
        function F = compute_fig_2014(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=100;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            s_sigmaForDiffusion=1.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.2:0.2:0.2);
            v_bandwidthPercentage=[0.01];
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=1.0; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*( eye(size(m_adjacency)));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            % data normalization
            v_mean = mean(m_temperatureTimeSeries,2);
            v_std = std(m_temperatureTimeSeries')';
            % 			m_temperatureTimeSeries = diag(1./v_std)*(m_temperatureTimeSeries...
            %                 - v_mean*ones(1,size(m_temperatureTimeSeries,2)));
            
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %m_diffusionKernel=graph.getLaplacian();
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    [m_samples(v_timetIndicesForSamples,:),...
                        m_positions(v_timetIndicesForSamples,:)]...
                        = sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                    
                end
                myLegendKF{s_sampleInd}=strcat('Kernel Kalman Filter, ',...
                    sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                
                %% 6. Kernel Ridge Regression
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                    
                end
                myLegendKRR{s_sampleInd}=strcat('KRR time agnostic, ',...
                    sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                
                
                
            end
            for s_vertInd=1:s_numberOfVertices
                
                
                m_meanEstKF(s_vertInd,:)=mean(t_kfEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,1,:),2)';
                m_meanEstKRR(s_vertInd,:)=mean(t_kRRestimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,1,:),2)';
                
            end
            
            
            %% 9. measure difference
            %normalize errors
            myLegandTrueSignal{1}='True Signal';
            myLegend=[myLegandTrueSignal myLegendKF myLegendKRR];
            F = F_figure('X',(1:s_maximumTime),'Y',[mean(m_temperatureTimeSeriesSampled,1);...
                mean(m_meanEstKF,1);mean(m_meanEstKRR,1)],...
                'xlab','Time evolution','ylab','function value','tit',...
                sprintf('Temperature tracking'),'leg',myLegend);
            
        end
        function F = compute_fig_2214(obj,niter)
            F = obj.load_F_structure(2014);
            F.ylimit=[0 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Temperature across contiguous USA'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare Kalman filter up to time t Plot reconstruction
        %  error as increase the sampling set
        %  with Bandlimited model at each time t and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        %TODO why error increases
        function F = compute_fig_2015(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=100;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=24;
            s_mu=10^-7;
            s_diffusionSigma=1.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.1:0.9);
            v_bandwidthPercentage=[0.01,0.05];
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=10; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            m_timeAdjacency=v_propagationWeight*(eye(size(m_adjacency)));
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_diffusionSigma,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            % 			m_invExtendedKernel=KFonGSimulations.createInvExtendedGraphKernel...
            % 				(t_invDiffusionKernel,-t_timeAdjacencyAtDifferentTimes);
            %
            % make kernel great again
            % 			s_beta=0;
            % 			m_invExtendedKernel=m_invExtendedKernel+s_beta*eye(size(m_invExtendedKernel));
            % 			[~,p1]=chol(m_invExtendedKernel);%check if PD
            % 			[~,p2]=chol(m_invExtendedKernel((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
            % 				s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
            % 				*s_numberOfVertices));
            % 			if (p1+p2~=0)
            % 				assert(1==0,'Not Positive Definite Kernel')
            % 			end
            % 			t_eyeTens=repmat(eye(size(m_diffusionKernel)),[1,1,s_maximumTime]);
            % 			t_invDiffusionKernel=t_invDiffusionKernel+s_beta*t_eyeTens;
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_bandwidth,2));
            v_relativeErrorFinalKf=zeros(size(v_numberOfSamples,2),1);
            v_relativeErrorFinalKRR=zeros(size(v_numberOfSamples,2),1);
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_bandwidth,2));
            m_relativeFinalErrorDistr=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            m_relativeFinalErrorBL=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                v_relativeErrorKRR(s_sampleInd)=norm(t_kRRestimate(:,:,s_sampleInd)...
                    -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                v_relativeErrorKf(s_sampleInd)=norm(t_kfEstimate(:,:,s_sampleInd)...
                    -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    m_relativeErrorbandLimitedEstimate(s_sampleInd,s_bandInd)...
                        =norm(t_bandLimitedEstimate(:,:,s_sampleInd,s_bandInd)...
                        -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                    
                    myLegend{s_bandInd}=strcat('Bandlimited  ',...
                        sprintf(' W=%g',s_bandwidth));
                end
                
                %% 7.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    m_relativeErrorDistr(s_sampleInd,s_bandInd)=...
                        norm( t_distrEstimate(:,:,s_sampleInd,s_bandInd)...
                        -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                    myLegend{s_bandInd+size(v_bandwidth,2)}=strcat('DLSR, ',...
                        sprintf(' W=%g',s_bandwidth));
                    
                end
                
                
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd));
                        m_relativeErrorbandLimitedEstimate(s_timeInd,s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd));
                        
                    end
                    
                    
                end
                
                s_summedNorm=sum(v_normOFNotSampled(1:s_maximumTime));
                v_relativeErrorFinalKf(s_sampleInd)...
                    =sum(v_normOFKFErrors(1:s_maximumTime))/...
                    s_summedNorm;%s_timeInd*s_numberOfVertices;
                v_relativeErrorFinalKRR(s_sampleInd)...
                    =sum(v_normOfKrrErrors(1:s_maximumTime))/...
                    s_summedNorm;%s_timeInd*s_numberOfVertices;
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_relativeFinalErrorDistr(s_sampleInd,s_bandInd)...
                        =m_relativeErrorDistr(s_maximumTime,s_bandInd)/...
                        s_summedNorm;
                    
                    m_relativeFinalErrorBL(s_sampleInd,s_bandInd)...
                        =m_relativeErrorbandLimitedEstimate(s_maximumTime,s_bandInd)/...
                        s_summedNorm;
                    myLegendDLSR{s_bandInd}=strcat('DLSR',...
                        sprintf(' W=%g',s_bandwidth ));
                    myLegendBan{s_bandInd}=...
                        strcat('BL-TA, ',...
                        sprintf(' W=%g',s_bandwidth));
                end
                
            end
            
            
            
            
            myLegendKF{1}='KKF';
            myLegendKRR{1}='KRR-TA';
            
            %% 8. measure difference
            
            myLegend=[myLegendDLSR myLegendBan myLegendKF myLegendKRR];
            F = F_figure('X',v_numberOfSamples,'Y',[m_relativeFinalErrorDistr...
                ,m_relativeFinalErrorBL,...
                v_relativeErrorFinalKf,v_relativeErrorFinalKRR]',...
                'xlab','Measuring Stations','ylab','NMSE','leg',myLegend);
            F.ylimit=[0 1];
            F.caption=[	  'NMSE vs sampling size Temperature\n',...
                sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_diffusionSigma)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)];
        end
        function F = compute_fig_2215(obj,niter)
            F = obj.load_F_structure(2015);
            F.ylimit=[0 2];
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Temperature across contiguous USA'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        %% Real data simulations
        %  Data used: Temperature Time Series in stations across continental
        %  USA
        %  Goal: Compare KKF error at T with DLSR LMS BL-IE and KRR-IE while
        %  increasing the sampling set
        function F = compute_fig_2016(obj,niter)
            %% 0. define parameters
            
            % maximum signal instances sampled
            s_maximumTime=360;
            
            % sample period: we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=24;
            
            % KKF parameters
            %regularization parameter
            s_mu=10^-7;
            s_diffusionSigma=1.6;
            
            s_monteCarloSimulations=niter;
            s_SNR=Inf; % no noise real data
            
            %sample size percentage
            v_samplePercentage=(0.1:0.1:1);
            % LMS step size
            s_stepLMS=2;
            
            %DLSR parameters
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            
            %bandwidth of bandlimited approaches
            v_bandwidth=[8,12];
            
            %% 1. define graph
            tic
            
            
            
            % load the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));
            % size of the graph
            s_numberOfVertices=size(m_adjacency,1);
            
            v_numberOfSamples=...
                round(s_numberOfVertices*v_samplePercentage);
            
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            % define extended graph for KKF
            
            % weight of edges between the same node
            % in consecutive time instances for extended graph
            v_propagationWeight=0.01;
            m_timeAdjacency=v_propagationWeight*(eye(size(m_adjacency)));
            % define adjacency in space and time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            
            %% 2. choise of Kernel
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_diffusionSigma,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            
            % create extended kernel
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            % generate transition, correlation matrices of KKF
            
            % observation variance for KKF
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            
            % initial state error covariance for KKF
            m_sigma0=zeros(s_numberOfVertices);
            % mean of initial state for KKF
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations);
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %% 4. Estimate signal
            
            % initialize tensors for estimates
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_bandwidth,2));
            m_relativeErrorLms=zeros(s_maximumTime,size(v_bandwidth,2));
            % initialize tensors for errors
            v_relativeErrorFinalKf=zeros(size(v_numberOfSamples,2),1);
            v_relativeErrorFinalKRR=zeros(size(v_numberOfSamples,2),1);
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_bandwidth,2));
            m_relativeFinalErrorDistr=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            m_relativeFinalErrorLms=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            m_relativeFinalErrorBL=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            v_allPositions=(1:s_numberOfVertices)';
            % iterate over the sampling size
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4.0 generate samples
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                %Same sample locations across time
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                end
                
                %% 4.1 KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 4.2 Kernel Ridge Regression estimate
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                % 				v_relativeErrorKRR(s_sampleInd)=norm(t_kRRestimate(:,:,s_sampleInd)...
                % 					-m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                % 				v_relativeErrorKf(s_sampleInd)=norm(t_kfEstimate(:,:,s_sampleInd)...
                % 					-m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                %% 4.3 bandlimited estimate
                
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    m_relativeErrorbandLimitedEstimate(s_sampleInd,s_bandInd)...
                        =norm(t_bandLimitedEstimate(:,:,s_sampleInd,s_bandInd)...
                        -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                    
                end
                
                %% 4.4 DLSR estimate
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_adjacency);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
                %% 4.5 LMS
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_adjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
                    t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    
                end
                
                %% 4.6 measure error
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfLMSErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            
                            if(norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                    ,s_mtind,s_sampleInd,s_bandInd)...
                                    -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro')...
                                    >norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind)))
                                %sampling set has caused huge error
                                %disregard it
                                m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                    norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind));
                            else
                                m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                    norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                    ,s_mtind,s_sampleInd,s_bandInd)...
                                    -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            end
                            if(norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                    ,s_mtind,s_sampleInd,s_bandInd)...
                                    -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro')...
                                    >norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind)))
                                %sampling set has caused huge error
                                %disregard it
                                m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                    norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind));
                            else
                                m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                    norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                    ,s_mtind,s_sampleInd,s_bandInd)...
                                    -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            end
                            
                            %                              if(m_normOfDLSRErrors(s_timeInd,s_bandInd)>norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind)))
                            %                                 m_normOfDLSRErrors(s_timeInd,s_bandInd)=norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind));
                            %                             end
                            m_normOfLMSErrors(s_timeInd,s_bandInd)=m_normOfLMSErrors(s_timeInd,s_bandInd)+...
                                norm(t_lmsEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd));
                        m_relativeErrorLms(s_timeInd,s_bandInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd));
                        m_relativeErrorbandLimitedEstimate(s_timeInd,s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd));
                        
                    end
                    
                    
                end
                
                s_summedNorm=sum(v_normOFNotSampled(1:s_maximumTime));
                v_relativeErrorFinalKf(s_sampleInd)...
                    =sum(v_normOFKFErrors(1:s_maximumTime))/...
                    s_summedNorm;%s_timeInd*s_numberOfVertices;
                v_relativeErrorFinalKRR(s_sampleInd)...
                    =sum(v_normOfKrrErrors(1:s_maximumTime))/...
                    s_summedNorm;%s_timeInd*s_numberOfVertices;
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_relativeFinalErrorDistr(s_sampleInd,s_bandInd)...
                        =m_relativeErrorDistr(s_maximumTime,s_bandInd)/...
                        s_summedNorm;
                    m_relativeFinalErrorLms(s_sampleInd,s_bandInd)...
                        =m_relativeErrorLms(s_maximumTime,s_bandInd)/...
                        s_summedNorm;
                    
                    m_relativeFinalErrorBL(s_sampleInd,s_bandInd)...
                        =m_relativeErrorbandLimitedEstimate(s_maximumTime,s_bandInd)/...
                        s_summedNorm;
                    myLegendDLSR{s_bandInd}=strcat('DLSR',...
                        sprintf(' B=%g',s_bandwidth ));
                    myLegendLMS{s_bandInd}=strcat('LMS',...
                        sprintf(' B=%g',s_bandwidth ));
                    myLegendBan{s_bandInd}=...
                        strcat('BL-IE, ',...
                        sprintf(' B=%g',s_bandwidth));
                end
                
            end
            
            
            
            
            myLegendKF{1}='KKF';
            myLegendKRR{1}='KRR-IE';
            
            %% 5. return F_Figure object containing the simulation results
            myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKRR myLegendKF];
            F = F_figure('X',v_numberOfSamples,'Y',[m_relativeFinalErrorDistr...
                ,m_relativeFinalErrorLms,m_relativeFinalErrorBL,...
                v_relativeErrorFinalKRR,v_relativeErrorFinalKf]',...
                'xlab','Measuring Stations','ylab','NMSE','leg',myLegend);
            %F.ylimit=[0 1];
            F.caption=[	  'NMSE vs sampling size Temperature\n',...
                sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_diffusionSigma)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_numberOfSamples)];
            F.styles = {'-s','-^','--s','--^',':s',':^','-.*','-.d'};
            F.colorset=[0 0 0;0 .7 0;  1 .5 1;.5 .5 0; 1 1 0; .5 0 1;1 .5 0; 1 0 0];

        end
        function F = compute_fig_2216(obj,niter)
            F = obj.load_F_structure(2016);
            F.ylimit=[0 1];
            %F.logy = 1;
            F.xlimit=[11 87];
            F.styles = {'-s','-^','--s','--^',':s',':^','-.*','-.d'};
            F.colorset=[0 0 0;0 .7 0;  1 .5 1;.5 .5 0; 1 1 0; .5 0 1;1 .5 0; 1 0 0];
            s_chunk=20;
            % 			s_intSize=size(F.Y,2)-1;
            % 			s_ind=1;
            % 			s_auxind=1;
            % 			auxY(:,1)=F.Y(:,1);
            % 			auxX(:,1)=F.X(:,1);
            % 			while s_ind<s_intSize
            % 				s_ind=s_ind+1;
            % 			if mod(s_ind,s_chunk)==0
            % 				s_auxind=s_auxind+1;
            % 			   auxY(:,s_auxind)=F.Y(:,s_ind);
            % 			   auxX(:,s_auxind)=F.X(:,s_ind);
            % 			   s_ind=s_ind-1;
            % 			end
            % 			end
            % 			s_auxind=s_auxind+1;
            % 			auxY(:,s_auxind)=F.Y(:,end);
            % 			auxX(:,s_auxind)=F.X(:,end);
            % 			F.Y=auxY;
            % 			F.X=auxX;
            F.leg_pos='northeast';
            
            F.ylab='NMSE';
            F.xlab='Sampling size S';
            
            %F.pos=[680 729 509 249];
            F.tit='';
        end
        function F = compute_fig_2316(obj,niter)
            F = obj.load_F_structure(2016);
            F.ylimit=[0 1];
            %F.logy = 1;
            F.xlimit=[11 87];
            F.styles = {'-s','-^','--s','--^',':s',':^','-.o','-.d'};
            F.colorset=[0 0 0;0 .7 0;0 0 .9 ;.5 .5 0; .9 0 .9 ;.5 0 1;0 .7 .7;1 0 0; 1 .5 0;  1 0 .5; 0 1 .5;0 1 0];
            F.leg{1}='DLSR B=2';
            F.leg{2}='DLSR B=4';
            F.leg{3}='LMS B=2';
            F.leg{4}='LMS B=4';
            F.leg{5}='BL-IE B=2';
            F.leg{6}='BL-IE B=4';
            aux=F.leg{7};
            F.leg{7}=F.leg{8};
            F.leg{8}=aux;
            F.leg{7}='KRR-IE';
            F.leg{8}='KRR-DE';
            aux=F.Y(7,:);
            F.Y(7,:)=F.Y(8,:);
            F.Y(8,:)=aux;
            s_chunk=20;
            % 			s_intSize=size(F.Y,2)-1;
            % 			s_ind=1;
            % 			s_auxind=1;
            % 			auxY(:,1)=F.Y(:,1);
            % 			auxX(:,1)=F.X(:,1);
            % 			while s_ind<s_intSize
            % 				s_ind=s_ind+1;
            % 			if mod(s_ind,s_chunk)==0
            % 				s_auxind=s_auxind+1;
            % 			   auxY(:,s_auxind)=F.Y(:,s_ind);
            % 			   auxX(:,s_auxind)=F.X(:,s_ind);
            % 			   s_ind=s_ind-1;
            % 			end
            % 			end
            % 			s_auxind=s_auxind+1;
            % 			auxY(:,s_auxind)=F.Y(:,end);
            % 			auxX(:,s_auxind)=F.X(:,end);
            % 			F.Y=auxY;
            % 			F.X=auxX;
            F.leg_pos='northeast';
            
            F.ylab='NMSE';
            F.xlab='Sampling size S';
            
            %F.pos=[680 729 509 249];
            F.tit='';
        end
        
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare Kalman filter up to time t Plot reconstruction
        %  error as increase the sampling set  and WangWangGuo paper and
        %  LMS Lorenzo
        %  Kernel used: Diffusion Kernel in space
        %TODO why error increases
        function F = compute_fig_2017(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=10;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            s_diffusionSigma=1.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.1:0.9);
            v_bandwidthPercentage=[0.01,0.05];
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=10; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            m_timeAdjacency=v_propagationWeight*(eye(size(m_adjacency)));
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_diffusionSigma,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            % 			m_invExtendedKernel=KFonGSimulations.createInvExtendedGraphKernel...
            % 				(t_invDiffusionKernel,-t_timeAdjacencyAtDifferentTimes);
            %
            % make kernel great again
            % 			s_beta=0;
            % 			m_invExtendedKernel=m_invExtendedKernel+s_beta*eye(size(m_invExtendedKernel));
            % 			[~,p1]=chol(m_invExtendedKernel);%check if PD
            % 			[~,p2]=chol(m_invExtendedKernel((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
            % 				s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
            % 				*s_numberOfVertices));
            % 			if (p1+p2~=0)
            % 				assert(1==0,'Not Positive Definite Kernel')
            % 			end
            % 			t_eyeTens=repmat(eye(size(m_diffusionKernel)),[1,1,s_maximumTime]);
            % 			t_invDiffusionKernel=t_invDiffusionKernel+s_beta*t_eyeTens;
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_bandwidth,2));
            m_relativeErrorLms=zeros(s_maximumTime,size(v_bandwidth,2));
            v_relativeErrorFinalKf=zeros(size(v_numberOfSamples,2),1);
            v_relativeErrorFinalKRR=zeros(size(v_numberOfSamples,2),1);
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_bandwidth,2));
            m_relativeFinalErrorDistr=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            m_relativeFinalErrorLms=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            m_relativeFinalErrorBL=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                v_relativeErrorKRR(s_sampleInd)=norm(t_kRRestimate(:,:,s_sampleInd)...
                    -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                v_relativeErrorKf(s_sampleInd)=norm(t_kfEstimate(:,:,s_sampleInd)...
                    -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    m_relativeErrorbandLimitedEstimate(s_sampleInd,s_bandInd)...
                        =norm(t_bandLimitedEstimate(:,:,s_sampleInd,s_bandInd)...
                        -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                    
                    myLegend{s_bandInd}=strcat('Bandlimited  ',...
                        sprintf(' W=%g',s_bandwidth));
                end
                
                %% 7.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_adjacency);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
                %% 8. LMS
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_adjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht);
                    t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    
                end
                
                
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfLMSErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfLMSErrors(s_timeInd,s_bandInd)=m_normOfLMSErrors(s_timeInd,s_bandInd)+...
                                norm(t_lmsEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd));
                        m_relativeErrorLms(s_timeInd,s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd));
                        m_relativeErrorbandLimitedEstimate(s_timeInd,s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd));
                        
                    end
                    
                    
                end
                
                s_summedNorm=sum(v_normOFNotSampled(1:s_maximumTime));
                v_relativeErrorFinalKf(s_sampleInd)...
                    =sum(v_normOFKFErrors(1:s_maximumTime))/...
                    s_summedNorm;%s_timeInd*s_numberOfVertices;
                v_relativeErrorFinalKRR(s_sampleInd)...
                    =sum(v_normOfKrrErrors(1:s_maximumTime))/...
                    s_summedNorm;%s_timeInd*s_numberOfVertices;
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_relativeFinalErrorDistr(s_sampleInd,s_bandInd)...
                        =m_relativeErrorDistr(s_maximumTime,s_bandInd)/...
                        s_summedNorm;
                    m_relativeFinalErrorLms(s_sampleInd,s_bandInd)...
                        =m_relativeErrorDistr(s_maximumTime,s_bandInd)/...
                        s_summedNorm;
                    
                    m_relativeFinalErrorBL(s_sampleInd,s_bandInd)...
                        =m_relativeErrorbandLimitedEstimate(s_maximumTime,s_bandInd)/...
                        s_summedNorm;
                    myLegendDLSR{s_bandInd}=strcat('DLSR',...
                        sprintf(' W=%g',s_bandwidth ));
                    myLegendLMS{s_bandInd}=strcat('LMS',...
                        sprintf(' W=%g',s_bandwidth ));
                    myLegendBan{s_bandInd}=...
                        strcat('BL-TA, ',...
                        sprintf(' W=%g',s_bandwidth));
                end
                
            end
            
            
            
            
            myLegendKF{1}='KKF';
            myLegendKRR{1}='KRR-TA';
            
            %% 8. measure difference
            
            myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKF myLegendKRR];
            F = F_figure('X',v_numberOfSamples,'Y',[m_relativeFinalErrorDistr...
                ,m_relativeFinalErrorLms,m_relativeFinalErrorBL,...
                v_relativeErrorFinalKf,v_relativeErrorFinalKRR]',...
                'xlab','Measuring Stations','ylab','NMSE','leg',myLegend);
            F.ylimit=[0 1];
            F.caption=[	  'NMSE vs sampling size Temperature\n',...
                sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_diffusionSigma)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)];
        end
        function F = compute_fig_2217(obj,niter)
            F = obj.load_F_structure(2017);
            F.ylimit=[0 2];
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Temperature across contiguous USA'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare perfomance of Kalman filter DLSR LMS Bandlimited and
        %  KRR agnostic up to time t as I
        %  on tracking the signal.
        function F = compute_fig_2018(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
              %% 0. define parameters
            
             % maximum signal instances sampled
            s_maximumTime=300;
            
            % sample period: we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            
            % KKF parameters
            %regularization parameter
            s_mu=10^-7;
            s_diffusionSigma=1.8;
            
            s_monteCarloSimulations=niter;
            s_SNR=Inf; % no noise real data
            
            %sample size percentage
            v_samplePercentage=(0.4:0.4:0.4);
            % LMS step size
            s_stepLMS=2;
            
            %DLSR parameters
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            
            %bandwidth of bandlimited approaches
            v_bandwidthBL=5;
            v_bandwidthLMS=8;
            v_bandwidthDLSR=8;
            if s_monteCarloSimulations>1
                    error('This function does not support multiple sampling realizations');
            end
            
            %% 1. define graph
           tic
            
            
            
            % load the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));
            % size of the graph
            s_numberOfVertices=size(m_adjacency,1);
            
            v_numberOfSamples=...
                round(s_numberOfVertices*v_samplePercentage);
            
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);   
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define extended graph for KKF
            
            % weight of edges between the same node
            % in consecutive time instances for extended graph
            v_propagationWeight=0.01;
            m_timeAdjacency=v_propagationWeight*(eye(size(m_adjacency)));
            % define adjacency in space and time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            
            %% 2. choise of Kernel
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_diffusionSigma,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            
            % create extended kernel
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            % generate transition, correlation matrices of KKF
            
            % observation variance for KKF
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            
            % initial state error covariance for KKF
            m_sigma0=zeros(s_numberOfVertices);
            % mean of initial state for KKF
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations);
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            % define tensors for estimates
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_bandwidthBL,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_bandwidthBL,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            
            
            %% 4.0 generate observations
            s_numberOfSamples=v_numberOfSamples;
            sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
            m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
            m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
            %Same sample locations needed for distributed algo
            [m_samples(1:s_numberOfSamples,:),...
                m_positions(1:s_numberOfSamples,:)]...
                = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
            for s_timeInd=2:s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                    (s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                    (s_timeInd)*s_numberOfSamples;
                m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                for s_mtId=1:s_monteCarloSimulations
                    m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                        ((s_timeInd-1)*s_numberOfVertices+...
                        m_positions(v_timetIndicesForSamples,s_mtId));
                end
                
            end
            
            %% 4.1 KKF estimate
            kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                't_previousMinimumSquaredError',t_initialSigma0,...
                'm_previousEstimate',m_initialState);
            for s_timeInd=1:s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                    (s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                    (s_timeInd)*s_numberOfSamples;
                
                %samples and positions at time t
                m_samplest=m_samples(v_timetIndicesForSamples,:);
                m_positionst=m_positions(v_timetIndicesForSamples,:);
                %estimate
                
                [t_kfEstimate(v_timetIndicesForSignals,:),t_newMSE]=...
                    kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                    t_transitions(:,:,s_timeInd),...
                    t_correlations(:,:,s_timeInd),m_sigma(s_timeInd));
                % prepare KF for next iteration
                kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:);
                
            end
            %% 4.2 Kernel Ridge Regression estimate
            
            nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
            for s_timeInd=1:s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                    (s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                    (s_timeInd)*s_numberOfSamples;
                
                %samples and positions at time t
                m_samplest=m_samples(v_timetIndicesForSamples,:);
                m_positionst=m_positions(v_timetIndicesForSamples,:);
                %estimate
                
                [t_kRRestimate(v_timetIndicesForSignals,:)]=...
                    nonParametricGraphFunctionEstimator.estimate...
                    (m_samplest,m_positionst,s_mu);
                
            end
            %% 4.3 bandlimited estimate
                       
                       
            for s_bandInd=1:size(v_bandwidthBL,2)
                s_bandwidth=v_bandwidthBL(s_bandInd);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %create take diagonals from extended graph
                    m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_adjacency);
                    
                    %bandlimited estimate
                    bandlimitedGraphFunctionEstimator= ...
                        BandlimitedGraphFunctionEstimator('m_laplacian'...
                        ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                    t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_bandInd)=...
                        bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                    
                end
                 myLegendBan{s_bandInd}=...
                        strcat('BL-IE ',...
                        sprintf(' B=%g',s_bandwidth));
                
            end
            
            %% 4.4 DLSR            
            
            for s_bandInd=1:size(v_bandwidthBL,2)
                s_bandwidth=v_bandwidthDLSR(s_bandInd);
                distributedFullTrackingAlgorithmEstimator=...
                    DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                    's_bandwidth',s_bandwidth,'graph',graph);
                t_distrEstimate(:,:,s_bandInd)=...
                    distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                myLegendDLSR{s_bandInd}=strcat('DLSR',...
                        sprintf(' B=%g',s_bandwidth));
                   
                
            end
            %% 4.5 LMS
            for s_bandInd=1:size(v_bandwidthBL,2)
                s_bandwidth=v_bandwidthLMS(s_bandInd);
                m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                grapht=Graph('m_adjacency',m_adjacency);
                lMSFullTrackingAlgorithmEstimator=...
                    LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                    's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
                t_lmsEstimate(:,:,s_bandInd)=...
                    lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                 myLegendLMS{s_bandInd}=strcat('LMS',...
                        sprintf(' B=%g',s_bandwidth));
                
            end
            
            
            
            
            
            for s_vertInd=1:s_numberOfVertices
                
                
                m_meanEstKF(s_vertInd,:)=mean(t_kfEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,:),2)';
                m_meanEstKRR(s_vertInd,:)=mean(t_kRRestimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,:),2)';
                m_meanEstDLSR(s_vertInd,:)=mean(t_distrEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,:,1),2)';
                m_meanEstBan(s_vertInd,:)=mean(t_bandLimitedEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,:,1),2)';
                m_meanEstLMS(s_vertInd,:)=mean(t_lmsEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,:,1),2)';
                
            end
            %% 5. return F_figure
            
                   
                myLegendKF{1}='KKF ';
                myLegendKRR{1}='KRR-IE';
            myLegandTrueSignal{1}='True Signal';
            v_notSamplInd=setdiff((1:s_numberOfVertices)',m_positionst);
            s_vertexToPlot=v_notSamplInd(1);
            myLegend=[myLegandTrueSignal myLegendDLSR myLegendLMS myLegendBan myLegendKRR myLegendKF   ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_temperatureTimeSeriesSampled(s_vertexToPlot,:);m_meanEstDLSR(s_vertexToPlot,:);...
                m_meanEstLMS(s_vertexToPlot,:);m_meanEstBan(s_vertexToPlot,:);m_meanEstKRR(s_vertexToPlot,:);...
                m_meanEstKF(s_vertexToPlot,:)],...
                'xlab','Time evolution','ylab','function value','leg',myLegend);
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_diffusionSigma)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_numberOfSamples)...
                sprintf('B=%g\n',v_bandwidthBL)];
             F.styles = {'-','.','--',':','--.','-.'};
            F.colorset=[0 0 0;0 .7 0;0 0 .9 ;.5 .5 0; .9 0 .9 ;1 0 0;0 .7 .7;.5 0 1; 1 .5 0;  1 0 .5; 0 1 .5;0 1 0];
            
        end
        function F = compute_fig_2218(obj,niter)
            F = obj.load_F_structure(2018);
            F.ylimit=[0 130];
            %F.logy = 1;
            %F.xlimit=[10 100];
            
            F.styles = {'-','.','--',':','--.','-.'};
            F.colorset=[0 0 0;0 .7 0;0 0 .9 ;.5 .5 0; .9 0 .9 ;1 0 0;0 .7 .7;.5 0 1; 1 .5 0;  1 0 .5; 0 1 .5;0 1 0];
            %F.pos=[680 729 509 249];
            %Initially: True signal KKF KRR-TA DLSR LMS BL-TA
            %             aux=F.leg{2};
            %             F.leg{2}=F.leg{4};
            %             F.leg{4}=F.leg{6};
            %             F.leg{6}=aux;
            %             aux=F.leg{3};
            %             F.leg{3}=F.leg{5};
            %             F.leg{5}=aux;
            %             aux=F.Y(2,:);
            %             F.Y(2,:)=F.Y(4,:);
            %             F.Y(4,:)=F.Y(6,:);
            %             F.Y(6,:)=aux;
            %             aux=F.Y(3,:);
            %             F.Y(3,:)=F.Y(5,:);
            %             F.Y(5,:)=aux;
            %
            %
            %             F.leg{4}='BL-IE';
            %             F.leg{5}='KRR-IE';
            F.leg_pos='northwest';
            F.ylab='Temperature [F]';
            F.xlab='Time [hours]';
            %F.tit='Temperature tracking';
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        
        function F = compute_fig_20181(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=30;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            s_sigmaForDiffusion=1.8;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            v_bandwidthPercentage=[];
            v_stepLMS=[0.5, 2.2, 2.6, 2.8];
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %v_bandwidthPercentage=0.01;2
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=0.01; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*( eye(size(m_adjacency)));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidthBL=5;
            v_bandwidthLMS=8;
            v_bandwidthDLSR=8;
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            % data normalization
            v_mean = mean(m_temperatureTimeSeries,2);
            v_std = std(m_temperatureTimeSeries')';
            % 			m_temperatureTimeSeries = diag(1./v_std)*(m_temperatureTimeSeries...
            %                 - v_mean*ones(1,size(m_temperatureTimeSeries,2)));
            
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %m_diffusionKernel=graph.getLaplacian();
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                
                
                
                %% 9. LMS
                for s_mulmsInd=1:size(v_stepLMS,2)
                    s_stepLMS=v_stepLMS(s_mulmsInd);
                    m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_adjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',v_bandwidthLMS,'graph',grapht,'s_stepLMS',s_stepLMS);
                    t_lmsEstimate(:,:,s_sampleInd,s_mulmsInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    myLegendLMS{(s_sampleInd-1)*size(v_stepLMS,2)+s_mulmsInd}=strcat('LMS  \mu_{LMS}',...
                        sprintf('=%g',s_stepLMS));
                end
                
            end
            
            for s_vertInd=1:s_numberOfVertices
                
                
                m_meanEstKF(s_vertInd,:)=mean(t_kfEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,1,:),2)';
                m_meanEstKRR(s_vertInd,:)=mean(t_kRRestimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,1,:),2)';
                m_meanEstDLSR(s_vertInd,:)=mean(t_distrEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,:,1,1),2)';
                m_meanEstBan(s_vertInd,:)=mean(t_bandLimitedEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,:,1,1),2)';
                m_meanEstLMS(s_vertInd,:)=mean(t_lmsEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,:,1,1),2)';
                
            end
            %% 9. measure difference
            %normalize errors
            myLegandTrueSignal{1}='True Signal';
            v_notSamplInd=setdiff((1:s_numberOfVertices)',m_positions(1:s_numberOfSamples,:));
            s_vertexToPlot=v_notSamplInd(1);
            m_estLMS=squeeze(t_lmsEstimate((0:s_maximumTime-1)*...
                s_numberOfVertices+s_vertexToPlot,1,1,:))';
            myLegend=[myLegandTrueSignal  myLegendLMS  ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_temperatureTimeSeriesSampled(s_vertexToPlot,:);...
                m_estLMS],...
                'xlab','Time evolution','ylab','function value','leg',myLegend);
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',v_stepLMS)...
                sprintf('sampling size =%g\n',v_numberOfSamples)...
                sprintf('B=%g\n',v_bandwidthLMS)];
            
        end
        function F = compute_fig_22181(obj,niter)
            F = obj.load_F_structure(20181);
            F.ylimit=[0 80];
            %F.logy = 1;
            %F.xlimit=[0 50];
            
            F.styles = {'-','--',':','--.','-.','-.','--'};
            F.colorset=[0 0 0;0 .7 0;0 0 .9 ;.5 .5 0; .9 0 .9 ;1 0 0;0 .7 .7;.5 0 1; 1 .5 0;  1 0 .5; 0 1 .5;0 1 0];
            %F.pos=[680 729 509 249];
            %Initially: True signal KKF KRR-TA DLSR LMS BL-TA
            %             aux=F.leg{2};
            %             F.leg{2}=F.leg{4};
            %             F.leg{4}=F.leg{6};
            %             F.leg{6}=aux;
            %             aux=F.leg{3};
            %             F.leg{3}=F.leg{5};
            %             F.leg{5}=aux;
            %             aux=F.Y(2,:);
            %             F.Y(2,:)=F.Y(4,:);
            %             F.Y(4,:)=F.Y(6,:);
            %             F.Y(6,:)=aux;
            %             aux=F.Y(3,:);
            %             F.Y(3,:)=F.Y(5,:);
            %             F.Y(5,:)=aux;
            %
            %
            %             F.leg{4}='BL-IE';
            %             F.leg{5}='KRR-IE';
            F.leg_pos='southeast';
            F.ylab='Temperature [F]';
            F.xlab='Time [hours]';
            %F.tit='Temperature tracking';
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        
        
        
        %% Real data simulations
        % Temperatures gft
        function F = compute_fig_20000(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=300;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            s_sigmaForDiffusion=1.8;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            v_bandwidthPercentage=[];
            s_stepLMS=2;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %v_bandwidthPercentage=0.01;2
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=0.01; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*( eye(size(m_adjacency)));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidthBL=5;
            v_bandwidthLMS=8;
            v_bandwidthDLSR=8;
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            % data normalization
            v_mean = mean(m_temperatureTimeSeries,2);
            v_std = std(m_temperatureTimeSeries')';
            % 			m_temperatureTimeSeries = diag(1./v_std)*(m_temperatureTimeSeries...
            %                 - v_mean*ones(1,size(m_temperatureTimeSeries,2)));
            
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            m_gft=abs(graph.getFourierTransform(m_temperatureTimeSeriesSampled));
            t1=1;
            t2=10;
            t3=20;
            t4=40;
            t5=50;
            F = F_figure('X',(1:s_numberOfVertices),'Y',[m_gft(:,t1),m_gft(:,t2),m_gft(:,t3),m_gft(:,t4),m_gft(:,t5);]',...
                'xlab','graph frequency','ylab','magnitude GFT','leg',{sprintf('t_1=%g\n',t1) sprintf('t_2=%g\n',t2) sprintf('t_3=%g\n',t3) sprintf('t_4=%g\n',t4) sprintf('t_5=%g\n',t5)});
            F.styles = {'-','.','--',':','--.','-.'};
            F.caption=[sprintf('t1=%g\n',t1),sprintf('t2=%g\n',t2),sprintf('t3=%g\n',t3),sprintf('t4=%g\n',t4),sprintf('t5=%g\n',t5)];
        end
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare Kalman filter up to time t reconstruction error
        %  measured seperately on the unobserved at each time, summed and normalize.
        %  Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model KRR at each time t and WangWangGuo paper LMS
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_2019(obj,niter)
            %% 0. define parameters
            
            % maximum signal instances sampled
            s_maximumTime=360;
            
            % sample period: we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=24;
            
            % KKF parameters
            %regularization parameter
            s_mu=10^-7;
            s_diffusionSigma=1.8;
            
            s_monteCarloSimulations=niter;
            s_SNR=Inf; % no noise real data
            
            %sample size percentage
            v_samplePercentage=(0.4:0.4:0.4);
            % LMS step size
            s_stepLMS=2;
            
            %DLSR parameters
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            
            %bandwidth of bandlimited approaches
            v_bandwidthBL=[5,8];
            v_bandwidthLMS=[14,18];
            v_bandwidthDLSR=[5,8];
            
            
            %% 1. define graph
            tic
            
            
            
            % load the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));
            % size of the graph
            s_numberOfVertices=size(m_adjacency,1);
            
            v_numberOfSamples=...
                round(s_numberOfVertices*v_samplePercentage);
            
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define extended graph for KKF
            
            % weight of edges between the same node
            % in consecutive time instances for extended graph
            v_propagationWeight=0.01;
            m_timeAdjacency=v_propagationWeight*(eye(size(m_adjacency)));
            % define adjacency in space and time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            
            %% 2. choise of Kernel
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_diffusionSigma,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            
            % create extended kernel
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            % generate transition, correlation matrices of KKF
            
            % observation variance for KKF
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            
            % initial state error covariance for KKF
            m_sigma0=zeros(s_numberOfVertices);
            % mean of initial state for KKF
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations);
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            % define tensors for estimates
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_bandwidthBL,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_bandwidthBL,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            
            
            %% 4.0 generate observations
            s_numberOfSamples=v_numberOfSamples;
            sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
            m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
            m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
            %Same sample locations needed for distributed algo
            [m_samples(1:s_numberOfSamples,:),...
                m_positions(1:s_numberOfSamples,:)]...
                = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
            for s_timeInd=2:s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                    (s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                    (s_timeInd)*s_numberOfSamples;
                m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                for s_mtId=1:s_monteCarloSimulations
                    m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                        ((s_timeInd-1)*s_numberOfVertices+...
                        m_positions(v_timetIndicesForSamples,s_mtId));
                end
                
            end
            
            %% 4.1 KKF estimate
            kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                't_previousMinimumSquaredError',t_initialSigma0,...
                'm_previousEstimate',m_initialState);
            for s_timeInd=1:s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                    (s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                    (s_timeInd)*s_numberOfSamples;
                
                %samples and positions at time t
                m_samplest=m_samples(v_timetIndicesForSamples,:);
                m_positionst=m_positions(v_timetIndicesForSamples,:);
                %estimate
                
                [t_kfEstimate(v_timetIndicesForSignals,:),t_newMSE]=...
                    kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                    t_transitions(:,:,s_timeInd),...
                    t_correlations(:,:,s_timeInd),m_sigma(s_timeInd));
                % prepare KF for next iteration
                kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:);
                
            end
            %% 4.2 Kernel Ridge Regression estimate
            
            nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
            for s_timeInd=1:s_maximumTime
                %time t indices
                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                    (s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                    (s_timeInd)*s_numberOfSamples;
                
                %samples and positions at time t
                m_samplest=m_samples(v_timetIndicesForSamples,:);
                m_positionst=m_positions(v_timetIndicesForSamples,:);
                %estimate
                
                [t_kRRestimate(v_timetIndicesForSignals,:)]=...
                    nonParametricGraphFunctionEstimator.estimate...
                    (m_samplest,m_positionst,s_mu);
                
            end
            %% 4.3 bandlimited estimate
                       
                       
            for s_bandInd=1:size(v_bandwidthBL,2)
                s_bandwidth=v_bandwidthBL(s_bandInd);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %create take diagonals from extended graph
                    m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_adjacency);
                    
                    %bandlimited estimate
                    bandlimitedGraphFunctionEstimator= ...
                        BandlimitedGraphFunctionEstimator('m_laplacian'...
                        ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                    t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_bandInd)=...
                        bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                    
                end
                
                
            end
            
            %% 4.4 DLSR            
            
            for s_bandInd=1:size(v_bandwidthBL,2)
                s_bandwidth=v_bandwidthDLSR(s_bandInd);
                distributedFullTrackingAlgorithmEstimator=...
                    DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                    's_bandwidth',s_bandwidth,'graph',graph);
                t_distrEstimate(:,:,s_bandInd)=...
                    distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                
                
            end
            %% 4.5 LMS
            for s_bandInd=1:size(v_bandwidthBL,2)
                s_bandwidth=v_bandwidthLMS(s_bandInd);
                m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                grapht=Graph('m_adjacency',m_adjacency);
                lMSFullTrackingAlgorithmEstimator=...
                    LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                    's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
                t_lmsEstimate(:,:,s_bandInd)=...
                    lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                
                
            end
            
            
            
            %% 4.6 measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_bandwidthBL,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_bandwidthLMS,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_bandwidthDLSR,2));
            v_allPositions=(1:s_numberOfVertices)';
            v_normOFKFErrors=zeros(s_maximumTime,1);
            v_normOFNotSampled=zeros(s_maximumTime,1);
            v_normOfKrrErrors=zeros(s_maximumTime,1);
            m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidthBL,2));
            m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidthBL,2));
            m_normOfLMSErrors=zeros(s_maximumTime,size(v_bandwidthBL,2));
            for s_timeInd=1:s_maximumTime
                %from the begining up to now
                v_timetIndicesForSignals=1:...
                    (s_timeInd)*s_numberOfVertices;
                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                    (s_timeInd)*s_numberOfSamples;
                
                m_samplest=m_samples(v_timetIndicesForSamples,:);
                m_positionst=m_positions(v_timetIndicesForSamples,:);
                %this vector should be added to the positions of the sa
                
                
                for s_mtind=1:s_monteCarloSimulations
                    v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                    v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                        norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                        ,s_mtind)...
                        -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                        norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                        ,s_mtind)...
                        -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                        norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                end
                
                s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                m_relativeErrorKf(s_timeInd)...
                    =sum(v_normOFKFErrors(1:s_timeInd))/...
                    s_summedNorm;%s_timeInd*s_numberOfVertices;
                m_relativeErrorKRR(s_timeInd)...
                    =sum(v_normOfKrrErrors(1:s_timeInd))/...
                    s_summedNorm;%s_timeInd*s_numberOfVertices;
                for s_bandInd=1:size(v_bandwidthBL,2)
                    
                    for s_mtind=1:s_monteCarloSimulations
                        m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                            norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_bandInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                            norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_bandInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        m_normOfLMSErrors(s_timeInd,s_bandInd)=m_normOfLMSErrors(s_timeInd,s_bandInd)+...
                            norm(t_lmsEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_bandInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_bandwidth=v_bandwidthBL(s_bandInd);
                    m_relativeErrorDistr(s_timeInd,s_bandInd)=...
                        sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorLms(s_timeInd,s_bandInd)=...
                        sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd))/...
                        s_summedNorm;
                    m_relativeErrorbandLimitedEstimate(s_timeInd,s_bandInd)...
                        =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    s_bandwidth=v_bandwidthDLSR(s_bandInd);
                    myLegendDLSR{s_bandInd}=strcat('DLSR',...
                        sprintf(' B=%g',s_bandwidth));
                    s_bandwidth=v_bandwidthLMS(s_bandInd);
                    myLegendLMS{s_bandInd}=strcat('LMS',...
                        sprintf(' B=%g',s_bandwidth));
                    s_bandwidth=v_bandwidthBL(s_bandInd);
                    myLegendBan{s_bandInd}=...
                        strcat('BL-IE ',...
                        sprintf(' B=%g',s_bandwidth));
                end
                myLegendKF{1}='KKF ';
                myLegendKRR{1}='KRR-IE';
                
            end
            
            %% 5. return F_Figure object
            myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKRR myLegendKF  ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate,...
                m_relativeErrorKRR, m_relativeErrorKf]',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F.ylimit=[0 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_diffusionSigma)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_samplePercentage)];
            
        end
        function F = compute_fig_2219(obj,niter)
            F = obj.load_F_structure(2019);
            F.ylimit=[0 1];
            %F.logy = 1;
            F.xlimit=[0 360];
            F.leg{7}='KRR-IE';
            %             F.leg{1}=F.leg{2};
            %             F.leg{2}=F.leg{3};
            %             F.leg{3}=F.leg{5};
            %             F.leg{4}=F.leg{6};
            %             F.leg{5}=F.leg{8};
            %             F.leg{6}=F.leg{9};
            %             F.leg{7}=F.leg{10};
            %             F.leg{8}=F.leg{11};
            %             F.leg=F.leg{1:8};
            %             F.Y(1,:)=[];
            %             F.Y(4,:)=[];
            %             F.Y(7,:)=[];
            F.styles = {'-s','-^','--s','--^',':s',':^','-.*','-.d'};
            F.colorset=[0 0 0;0 .7 0;  1 .5 1;.5 .5 0;0 1 1; .5 0 1;1 .5 0; 1 0 0];
            s_chunk=20;
            s_intSize=size(F.Y,2)-1;
            s_ind=1;
            s_auxind=1;
            auxY(:,1)=F.Y(:,1);
            auxX(:,1)=F.X(:,1);
            while s_ind<s_intSize
                s_ind=s_ind+1;
                if mod(s_ind,s_chunk)==0
                    s_auxind=s_auxind+1;
                    auxY(:,s_auxind)=F.Y(:,s_ind);
                    auxX(:,s_auxind)=F.X(:,s_ind);
                    %s_ind=s_ind-1;
                end
            end
            s_auxind=s_auxind+1;
            auxY(:,s_auxind)=F.Y(:,end);
            auxX(:,s_auxind)=F.X(:,end);
            F.Y=auxY;
            F.X=auxX;
            F.leg_pos='northeast';
            
            F.ylab='NMSE';
            F.xlab='Time [day]';
            
            %F.pos=[680 729 509 249];
            F.tit='';
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        function F = compute_fig_2319(obj,niter)
            F = obj.load_F_structure(2019);
            F.ylimit=[0 0.3];
            %F.logy = 1;
            F.xlimit=[0 360];
            F.styles = {'-s','-^','--s','--^',':s',':^','-.o','-.d'};
            F.colorset=[0 0 0;0 .7 0;0 0 .9 ;.5 .5 0; .9 0 .9 ;.5 0 1;0 .7 .7;1 0 0; 1 .5 0;  1 0 .5; 0 1 .5;0 1 0];
            F.leg{1}='DLSR B=2';
            F.leg{2}='DLSR B=4';
            F.leg{3}='LMS B=2';
            F.leg{4}='LMS B=4';
            F.leg{5}='BL-IE B=2';
            F.leg{6}='BL-IE B=4';
            aux=F.leg{7};
            F.leg{7}=F.leg{8};
            F.leg{8}=aux;
            F.leg{7}='KRR-IE';
            aux=F.Y(7,:);
            F.Y(7,:)=F.Y(8,:);
            F.Y(8,:)=aux;
            F.Y(1:7,:)=[];
            F.leg={'KKF'};
            s_chunk=20;
            s_intSize=size(F.Y,2)-1;
            s_ind=1;
            s_auxind=1;
            auxY(:,1)=F.Y(:,1);
            auxX(:,1)=F.X(:,1);
            while s_ind<s_intSize
                s_ind=s_ind+1;
                if mod(s_ind,s_chunk)==0
                    s_auxind=s_auxind+1;
                    auxY(:,s_auxind)=F.Y(:,s_ind);
                    auxX(:,s_auxind)=F.X(:,s_ind);
                    %s_ind=s_ind-1;
                end
            end
            s_auxind=s_auxind+1;
            auxY(:,s_auxind)=F.Y(:,end);
            auxX(:,s_auxind)=F.X(:,end);
            F.Y=auxY;
            F.X=auxX;
            F.leg_pos='northeast';
            
            F.ylab='NMSE';
            F.xlab='Time [day]';
            
            %F.pos=[680 729 509 249];
            F.tit='';
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        % find optimal bandwidth for LMS
        function F = compute_fig_2419(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=360;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=24;
            s_mu=10^-7;
            s_sigmaForDiffusion=1.8;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            %v_bandwidthPercentage=[0.01,0.1];
            s_stepLMS=2;
            s_muDLSR=4;
            s_betaDLSR=2;
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=0.01; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidthBL=[2,5,8];
            v_bandwidthLMS=[4,5,6,7,8,9,10,11];
            v_bandwidthDLSR=[2,5,8];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            % data normalization
            v_mean = mean(m_temperatureTimeSeries,2);
            v_std = std(m_temperatureTimeSeries')';
            % 			m_temperatureTimeSeries = diag(1./v_std)*(m_temperatureTimeSeries...
            %                 - v_mean*ones(1,size(m_temperatureTimeSeries,2)));
            
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            %
            % 			diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            % 			m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            % 			check expression again
            % 			t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            % 			[t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
            % 				(t_invSpatialDiffusionKernel...
            % 				,-t_timeAdjacencyAtDifferentTimes...
            % 				,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            % 			t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
            % 				,size(v_numberOfSamples,2));
            % 			t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
            % 				,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            % 			t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
            % 				,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            % 			t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
            % 				,size(v_numberOfSamples,2));
            
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthLMS,2));
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                % 				kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                % 					't_previousMinimumSquaredError',t_initialSigma0,...
                % 					'm_previousEstimate',m_initialState);
                % 				for s_timeInd=1:s_maximumTime
                % 					%time t indices
                % 					v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                % 						(s_timeInd)*s_numberOfVertices;
                % 					v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                % 						(s_timeInd)*s_numberOfSamples;
                %
                % 					%samples and positions at time t
                % 					m_samplest=m_samples(v_timetIndicesForSamples,:);
                % 					m_positionst=m_positions(v_timetIndicesForSamples,:);
                % 					%estimate
                %
                % 					[t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                % 						kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                % 						t_transitions(:,:,s_timeInd),...
                % 						t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                % 					% prepare KF for next iteration
                % 					kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                % 					kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                % 						s_sampleInd);
                %
                %                 end
                %% 6. Kernel Ridge Regression
                
                % %                 nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                % %                     ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                % % 				for s_timeInd=1:s_maximumTime
                % % 					%time t indices
                % % 					v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                % % 						(s_timeInd)*s_numberOfVertices;
                % % 					v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                % % 						(s_timeInd)*s_numberOfSamples;
                % %
                % % 					%samples and positions at time t
                % % 					m_samplest=m_samples(v_timetIndicesForSamples,:);
                % % 					m_positionst=m_positions(v_timetIndicesForSamples,:);
                % % 					%estimate
                % %
                % % 					[t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                % % 						nonParametricGraphFunctionEstimator.estimate...
                % %                         (m_samplest,m_positionst,s_mu);
                % %
                % % 				end
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                % 				for s_bandInd=1:size(v_bandwidthBL,2)
                % 					s_bandwidth=v_bandwidthBL(s_bandInd);
                % 					for s_timeInd=1:s_maximumTime
                % 						%time t indices
                % 						v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                % 							(s_timeInd)*s_numberOfVertices;
                % 						v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                % 							(s_timeInd)*s_numberOfSamples;
                %
                % 						%samples and positions at time t
                %
                % 						m_samplest=m_samples(v_timetIndicesForSamples,:);
                % 						m_positionst=m_positions(v_timetIndicesForSamples,:);
                % 						%create take diagonals from extended graph
                % 						m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                % 						grapht=Graph('m_adjacency',m_adjacency);
                %
                % 						%bandlimited estimate
                % 						bandlimitedGraphFunctionEstimator= ...
                % 							BandlimitedGraphFunctionEstimator('m_laplacian'...
                % 							,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                % 						t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                % 							bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                %
                % 					end
                %
                %
                % 				end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                %
                %
                % 				for s_bandInd=1:size(v_bandwidthBL,2)
                % 					s_bandwidth=v_bandwidthDLSR(s_bandInd);
                % 					distributedFullTrackingAlgorithmEstimator=...
                % 						DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                % 						's_bandwidth',s_bandwidth,'graph',graph);
                % 					t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                % 						distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                %
                %
                % 				end
                %% . LMS
                for s_bandInd=1:size(v_bandwidthLMS,2)
                    s_bandwidth=v_bandwidthLMS(s_bandInd);
                    m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_adjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
                    t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    
                end
            end
            
            
            %% 9. measure difference
            
            % 			m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidthBL,2));
            % 			m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            %             m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            %
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidthLMS,2));
            
            % 			m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidthBL,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                %             v_normOfKrrErrors=zeros(s_maximumTime,1);
                %             m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidthBL,2));
                %             m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidthBL,2));
                m_normOfLMSErrors=zeros(s_maximumTime,size(v_bandwidthLMS,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        %                     v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                        %                         norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                        %                         ,s_mtind,s_sampleInd)...
                        % 						-m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        %                     v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                        %                         norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                        %                         ,s_mtind,s_sampleInd)...
                        % 						-m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                    % 					m_relativeErrorKf(s_timeInd, s_sampleInd)...
                    % 						=sum(v_normOFKFErrors(1:s_timeInd))/...
                    % 					s_summedNorm;%s_timeInd*s_numberOfVertices;
                    %                     m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                    % 						=sum(v_normOfKrrErrors(1:s_timeInd))/...
                    % 					s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidthLMS,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            %                     m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                            %                         norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            %                         ,s_mtind,s_sampleInd,s_bandInd)...
                            % 						-m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            %                     m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                            %                         norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            %                         ,s_mtind,s_sampleInd,s_bandInd)...
                            % 						-m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfLMSErrors(s_timeInd,s_bandInd)=m_normOfLMSErrors(s_timeInd,s_bandInd)+...
                                norm(t_lmsEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        %				s_bandwidth=v_bandwidthLMS(s_bandInd);
                        % 						m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd)=...
                        % 								sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                        %                                 s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorLms(s_timeInd,(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;
                        % 						m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd)...
                        % 							=sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                        %                                 s_summedNorm;%s_timeInd*s_numberOfVertices;
                        % 						s_bandwidth=v_bandwidthDLSR(s_bandInd);
                        % 						myLegendDLSR{(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd}=strcat('DLSR',...
                        % 							sprintf(' B=%g',s_bandwidth));
                        s_bandwidth=v_bandwidthLMS(s_bandInd);
                        myLegendLMS{(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd}=strcat('LMS',...
                            sprintf(' B=%g',s_bandwidth));
                        %                         s_bandwidth=v_bandwidthBL(s_bandInd);
                        % 						myLegendBan{(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd}=...
                        % 							strcat('BL-IE ',...
                        % 							sprintf(' B=%g',s_bandwidth));
                    end
                    % 					myLegendKF{s_sampleInd}='KKF ';
                    %                     myLegendKRR{s_sampleInd}='KRR-TA';
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendLMS  ];
            F = F_figure('X',(1:s_maximumTime),'Y',m_relativeErrorLms',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F.ylimit=[0 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_samplePercentage)];
            
        end
        function F = compute_fig_2519(obj,niter)
            F = obj.load_F_structure(2419);
            F.ylimit=[0.15 0.45];
            %F.logy = 1;
            F.xlimit=[0 360];
            F.styles = {'--s','--^','-*','--o','-.^','-.o','-.s','-.d',':o','-.*','-.d'};
            F.colorset=[0 0 0;0 .7 0;1 0 0;0 0 .9 ;  1 .5 1;.5 .5 0; .9 0 .9; 1 1 0; .5 0 1;0 .7 .7;1 .5 0];
            s_chunk=20;
            s_intSize=size(F.Y,2)-1;
            s_ind=1;
            s_auxind=1;
            auxY(:,1)=F.Y(:,1);
            auxX(:,1)=F.X(:,1);
            while s_ind<s_intSize
                s_ind=s_ind+1;
                if mod(s_ind,s_chunk)==0
                    s_auxind=s_auxind+1;
                    auxY(:,s_auxind)=F.Y(:,s_ind);
                    auxX(:,s_auxind)=F.X(:,s_ind);
                    %s_ind=s_ind-1;
                end
            end
            s_auxind=s_auxind+1;
            auxY(:,s_auxind)=F.Y(:,end);
            auxX(:,s_auxind)=F.X(:,end);
            F.Y=auxY;
            F.X=auxX;
            F.leg_pos='northeast';
            
            F.ylab='NMSE';
            F.xlab='Time [day]';
            
            %F.pos=[680 729 509 249];
            F.tit='';
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        % find optimal stepsize for LMS
        function F = compute_fig_2619(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=360;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=24;
            s_mu=10^-7;
            s_sigmaForDiffusion=1.8;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            %v_bandwidthPercentage=[0.01,0.1];
            v_stepLMS=[0.5,1,1.5,2,2.5,3];
            s_muDLSR=4;
            s_betaDLSR=2;
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=0.01; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidthBL=[2,5,8];
            v_bandwidthLMS=7;
            v_bandwidthDLSR=[2,5,8];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            % data normalization
            v_mean = mean(m_temperatureTimeSeries,2);
            v_std = std(m_temperatureTimeSeries')';
            % 			m_temperatureTimeSeries = diag(1./v_std)*(m_temperatureTimeSeries...
            %                 - v_mean*ones(1,size(m_temperatureTimeSeries,2)));
            
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            %
            % 			diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            % 			m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            % 			check expression again
            % 			t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            % 			[t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
            % 				(t_invSpatialDiffusionKernel...
            % 				,-t_timeAdjacencyAtDifferentTimes...
            % 				,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            % 			t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
            % 				,size(v_numberOfSamples,2));
            % 			t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
            % 				,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            % 			t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
            % 				,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            % 			t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
            % 				,size(v_numberOfSamples,2));
            
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthLMS,2));
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                % 				kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                % 					't_previousMinimumSquaredError',t_initialSigma0,...
                % 					'm_previousEstimate',m_initialState);
                % 				for s_timeInd=1:s_maximumTime
                % 					%time t indices
                % 					v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                % 						(s_timeInd)*s_numberOfVertices;
                % 					v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                % 						(s_timeInd)*s_numberOfSamples;
                %
                % 					%samples and positions at time t
                % 					m_samplest=m_samples(v_timetIndicesForSamples,:);
                % 					m_positionst=m_positions(v_timetIndicesForSamples,:);
                % 					%estimate
                %
                % 					[t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                % 						kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                % 						t_transitions(:,:,s_timeInd),...
                % 						t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                % 					% prepare KF for next iteration
                % 					kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                % 					kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                % 						s_sampleInd);
                %
                %                 end
                %% 6. Kernel Ridge Regression
                
                % %                 nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                % %                     ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                % % 				for s_timeInd=1:s_maximumTime
                % % 					%time t indices
                % % 					v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                % % 						(s_timeInd)*s_numberOfVertices;
                % % 					v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                % % 						(s_timeInd)*s_numberOfSamples;
                % %
                % % 					%samples and positions at time t
                % % 					m_samplest=m_samples(v_timetIndicesForSamples,:);
                % % 					m_positionst=m_positions(v_timetIndicesForSamples,:);
                % % 					%estimate
                % %
                % % 					[t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                % % 						nonParametricGraphFunctionEstimator.estimate...
                % %                         (m_samplest,m_positionst,s_mu);
                % %
                % % 				end
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                % 				for s_bandInd=1:size(v_bandwidthBL,2)
                % 					s_bandwidth=v_bandwidthBL(s_bandInd);
                % 					for s_timeInd=1:s_maximumTime
                % 						%time t indices
                % 						v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                % 							(s_timeInd)*s_numberOfVertices;
                % 						v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                % 							(s_timeInd)*s_numberOfSamples;
                %
                % 						%samples and positions at time t
                %
                % 						m_samplest=m_samples(v_timetIndicesForSamples,:);
                % 						m_positionst=m_positions(v_timetIndicesForSamples,:);
                % 						%create take diagonals from extended graph
                % 						m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                % 						grapht=Graph('m_adjacency',m_adjacency);
                %
                % 						%bandlimited estimate
                % 						bandlimitedGraphFunctionEstimator= ...
                % 							BandlimitedGraphFunctionEstimator('m_laplacian'...
                % 							,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                % 						t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                % 							bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                %
                % 					end
                %
                %
                % 				end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                %
                %
                % 				for s_bandInd=1:size(v_bandwidthBL,2)
                % 					s_bandwidth=v_bandwidthDLSR(s_bandInd);
                % 					distributedFullTrackingAlgorithmEstimator=...
                % 						DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                % 						's_bandwidth',s_bandwidth,'graph',graph);
                % 					t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                % 						distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                %
                %
                % 				end
                %% . LMS
                for s_stepInd=1:size(v_stepLMS,2)
                    s_stepLMS=v_stepLMS(s_stepInd);
                    m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_adjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',v_bandwidthLMS,'graph',grapht,'s_stepLMS',s_stepLMS);
                    t_lmsEstimate(:,:,s_sampleInd,s_stepInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    
                end
            end
            
            
            %% 9. measure difference
            
            % 			m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidthBL,2));
            % 			m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            %             m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            %
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_stepLMS,2));
            
            % 			m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidthBL,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                %             v_normOfKrrErrors=zeros(s_maximumTime,1);
                %             m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidthBL,2));
                %             m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidthBL,2));
                m_normOfLMSErrors=zeros(s_maximumTime,size(v_stepLMS,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        %                     v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                        %                         norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                        %                         ,s_mtind,s_sampleInd)...
                        % 						-m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        %                     v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                        %                         norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                        %                         ,s_mtind,s_sampleInd)...
                        % 						-m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                    % 					m_relativeErrorKf(s_timeInd, s_sampleInd)...
                    % 						=sum(v_normOFKFErrors(1:s_timeInd))/...
                    % 					s_summedNorm;%s_timeInd*s_numberOfVertices;
                    %                     m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                    % 						=sum(v_normOfKrrErrors(1:s_timeInd))/...
                    % 					s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_stepInd=1:size(v_stepLMS,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            %                     m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                            %                         norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            %                         ,s_mtind,s_sampleInd,s_bandInd)...
                            % 						-m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            %                     m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                            %                         norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            %                         ,s_mtind,s_sampleInd,s_bandInd)...
                            % 						-m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfLMSErrors(s_timeInd,s_stepInd)=m_normOfLMSErrors(s_timeInd,s_stepInd)+...
                                norm(t_lmsEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_stepInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        %				s_bandwidth=v_bandwidthLMS(s_bandInd);
                        % 						m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd)=...
                        % 								sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                        %                                 s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorLms(s_timeInd,(s_sampleInd-1)*size(v_stepLMS,2)+s_stepInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_stepInd))/...
                            s_summedNorm;
                        % 						m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd)...
                        % 							=sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                        %                                 s_summedNorm;%s_timeInd*s_numberOfVertices;
                        % 						s_bandwidth=v_bandwidthDLSR(s_bandInd);
                        % 						myLegendDLSR{(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd}=strcat('DLSR',...
                        % 							sprintf(' B=%g',s_bandwidth));
                        s_stepLMS=v_stepLMS(s_stepInd);
                        myLegendLMS{(s_sampleInd-1)*size(v_stepLMS,2)+s_stepInd}=strcat('\mu_{LMS} ',...
                            sprintf('=%g',s_stepLMS));
                        %                         s_bandwidth=v_bandwidthBL(s_bandInd);
                        % 						myLegendBan{(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd}=...
                        % 							strcat('BL-IE ',...
                        % 							sprintf(' B=%g',s_bandwidth));
                    end
                    % 					myLegendKF{s_sampleInd}='KKF ';
                    %                     myLegendKRR{s_sampleInd}='KRR-TA';
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendLMS  ];
            F = F_figure('X',(1:s_maximumTime),'Y',m_relativeErrorLms',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F.ylimit=[0 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('Bandwidth =%g\n',v_bandwidthLMS)...
                sprintf('sampling size =%g\n',v_samplePercentage)];
            
        end
        function F = compute_fig_2719(obj,niter)
            F = obj.load_F_structure(2619);
            F.ylimit=[0.15 0.6];
            %F.logy = 1;
            F.xlimit=[0 360];
            F.styles = {'--s','--^','-*','--o','-.^','-.o','-.s','-.d',':o','-.*','-.d'};
            F.colorset=[0 0 0;0 .7 0;1 0 0;0 0 .9 ;  1 .5 1;.5 .5 0; .9 0 .9; 1 1 0; .5 0 1;0 .7 .7;1 .5 0];
            s_chunk=20;
            s_intSize=size(F.Y,2)-1;
            s_ind=1;
            s_auxind=1;
            auxY(:,1)=F.Y(:,1);
            auxX(:,1)=F.X(:,1);
            while s_ind<s_intSize
                s_ind=s_ind+1;
                if mod(s_ind,s_chunk)==0
                    s_auxind=s_auxind+1;
                    auxY(:,s_auxind)=F.Y(:,s_ind);
                    auxX(:,s_auxind)=F.X(:,s_ind);
                    %s_ind=s_ind-1;
                end
            end
            s_auxind=s_auxind+1;
            auxY(:,s_auxind)=F.Y(:,end);
            auxX(:,s_auxind)=F.X(:,end);
            F.Y=auxY;
            F.X=auxX;
            F.leg_pos='northeast';
            
            F.ylab='NMSE';
            F.xlab='Time [day]';
            
            %F.pos=[680 729 509 249];
            F.tit='';
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare perfomance of Kalman filter up to time t as I
        %  change the parameters diffusion parameter reg parameter and
        %  weight of scaled Identity as time evolves
        % sampling: uniformly random each t
        %Kernel used: Diffusion Kernel in space
        function F = compute_fig_2020(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=360 ;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=24;
            v_mu=[10^-7]; %opt
            v_sigmaForDiffusion=[1,1.2,1.5,1.8,2]; %opt
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=[10^-4,10^-3,10^-2,10^-1,10^1,10^2,10^3,10^4,10^5,10^6]; % weight of edges between the same node opt
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            %
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            t_kfEstimate=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,size(v_numberOfSamples,2),s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_relativeErrorKf=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,s_maximumTime,size(v_numberOfSamples,2));
            v_allPositions=(1:s_numberOfVertices)';
            
            for s_scalingInd=1:size(v_propagationWeight,2)
                s_propagationWeight=v_propagationWeight(s_scalingInd);
                m_timeAdjacency=s_propagationWeight*(eye(size(m_adjacency)));
                
                t_timeAdjacencyAtDifferentTimes=...
                    repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
                
                % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
                % 			graphT=graphGenerator.realization;
                %
                %% 2. choise of Kernel must be positive definite
                % diffusion kernel
                graph=Graph('m_adjacency',m_adjacency);
                for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                    diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_sigmaInd),...
                        'm_laplacian',graph.getLaplacian);
                    m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
                    %m_diffusionKernel=m_diffusionKernel/max(
                    %check expression again
                    t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
                    
                    %% generate transition, correlation matrices
                    m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
                    m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
                    t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
                    for s_ind=1:s_monteCarloSimulations
                        t_sigma0(:,:,s_ind)=m_sigma0;
                    end
                    [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                        (t_invSpatialDiffusionKernel...
                        ,-t_timeAdjacencyAtDifferentTimes...
                        ,s_maximumTime,s_numberOfVertices,m_sigma0);
                    
                    %% 3. generate true signal
                    
                    m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
                    
                    m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
                    
                    
                    %%
                    
                    
                    
                    for s_sampleInd=1:size(v_numberOfSamples,2)
                        %% 4. generate observations
                        s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                        sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                        m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        
                        for s_timeInd=1:s_maximumTime
                            %time t indices
                            v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                (s_timeInd)*s_numberOfVertices;
                            v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                (s_timeInd)*s_numberOfSamples;
                            [m_samples(v_timetIndicesForSamples,:),...
                                m_positions(v_timetIndicesForSamples,:)]...
                                = sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
                            
                        end
                        
                        %% 5. KF estimate
                        kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                            't_previousMinimumSquaredError',t_initialSigma0,...
                            'm_previousEstimate',m_initialState);
                        for s_muInd=1:size(v_mu,2)
                            v_sigmaForKalman=sqrt((1:s_maximumTime)'*v_numberOfSamples*v_mu(s_muInd))';
                            v_normOFKFErrors=zeros(s_maximumTime,1);
                            v_normOFNotSampled=zeros(s_maximumTime,1);
                            for s_timeInd=1:s_maximumTime
                                %time t indices
                                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                    (s_timeInd)*s_numberOfVertices;
                                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                    (s_timeInd)*s_numberOfSamples;
                                
                                %samples and positions at time t
                                m_samplest=m_samples(v_timetIndicesForSamples,:);
                                m_positionst=m_positions(v_timetIndicesForSamples,:);
                                %estimate
                                
                                [m_prevEstimate,t_newMSE]=...
                                    kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                                    t_transitions(:,:,s_timeInd),...
                                    t_correlations(:,:,s_timeInd),v_sigmaForKalman(s_sampleInd,s_timeInd));
                                t_kfEstimate(s_scalingInd,s_sigmaInd,...
                                    s_muInd,s_sampleInd,v_timetIndicesForSignals,:)=m_prevEstimate;
                                % prepare KF for next iteration
                                kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                                kFOnGFunctionEstimator.m_previousEstimate=m_prevEstimate;
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                for s_mtind=1:s_monteCarloSimulations
                                    v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                                    v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                                        norm(squeeze(t_kfEstimate(s_scalingInd,s_sigmaInd,s_muInd,s_sampleInd,v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                        ,s_mtind))...
                                        -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                    v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                                        norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                end
                                
                                s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                                t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_timeInd, s_sampleInd)...
                                    =sum(v_normOFKFErrors(1:s_timeInd))/...
                                    s_summedNorm;
                                
                                
                                
                                
                                
                                
                                
                            end
                        end
                        
                    end
                end
            end
            
            
            
            plot(squeeze(t_kfEstimate(1,1,1,1,(0:s_maximumTime-1)*s_numberOfVertices+1,1)))
            for s_scalingInd=1:size(v_propagationWeight,2)
                for s_muInd=1:size(v_mu,2)
                    for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                        m_relativeErrorKf(s_scalingInd,(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd)=...
                            (t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_maximumTime, 1));
                        myLegend{(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd}=...
                            sprintf('KKF mu=%g, sigma=%g',v_mu(s_muInd),v_sigmaForDiffusion(s_sigmaInd));
                        
                    end
                end
            end
            
            F = F_figure('X',v_propagationWeight,'Y',m_relativeErrorKf',...
                'xlab','diagonal scaling','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
            %F.ylimit=[0 0.5];
            
            F.logx=1;
            F.caption=[	sprintf('regularization parameter mu=%g\n',v_mu),...
                sprintf(' diffusion parameter sigma=%g\n',v_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('sampling size =%g\n',v_samplePercentage)...
                sprintf('time samples =%g\n',s_maximumTime)...
                sprintf('hours per sample =%g\n',s_samplePeriod)...
                'sampling scheme : uniformly random at each t\n'];
            
            
            
        end
        function F = compute_fig_2220(obj,niter)
            F = obj.load_F_structure(2020);
            %F.ylimit=[0.1 0.4];
            %F.xlimit=[10 100];
            F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
                ,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Parameter selection'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare perfomance of Kalman filter up to time t as I
        %  change the parameters diffusion parameter reg parameter and
        %  weight of scaled Identity as time evolves s
        % sampling: same each t
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_2021(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=360 ;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=24;
            v_mu=[10^-7]; %opt
            v_sigmaForDiffusion=[0.8,1,1.2,1.5,1.8,2]; %opt
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=[10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3,10^5,10^6]; % weight of edges between the same node opt
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            %
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            t_kfEstimate=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,size(v_numberOfSamples,2),s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_relativeErrorKf=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,s_maximumTime,size(v_numberOfSamples,2));
            v_allPositions=(1:s_numberOfVertices)';
            
            for s_scalingInd=1:size(v_propagationWeight,2)
                s_propagationWeight=v_propagationWeight(s_scalingInd);
                m_timeAdjacency=s_propagationWeight*(eye(size(m_adjacency)));
                
                t_timeAdjacencyAtDifferentTimes=...
                    repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
                
                % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
                % 			graphT=graphGenerator.realization;
                %
                %% 2. choise of Kernel must be positive definite
                % diffusion kernel
                graph=Graph('m_adjacency',m_adjacency);
                for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                    diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_sigmaInd),...
                        'm_laplacian',graph.getLaplacian);
                    m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
                    %m_diffusionKernel=m_diffusionKernel/max(
                    %check expression again
                    t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
                    
                    %% generate transition, correlation matrices
                    m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
                    m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
                    t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
                    for s_ind=1:s_monteCarloSimulations
                        t_sigma0(:,:,s_ind)=m_sigma0;
                    end
                    [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                        (t_invSpatialDiffusionKernel...
                        ,-t_timeAdjacencyAtDifferentTimes...
                        ,s_maximumTime,s_numberOfVertices,m_sigma0);
                    
                    %% 3. generate true signal
                    
                    m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
                    
                    m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
                    
                    
                    %%
                    
                    
                    
                    for s_sampleInd=1:size(v_numberOfSamples,2)
                        %% 4. generate observations
                        s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                        sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                        m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        
                        %Same sample locations needed for distributed algo
                        [m_samples(1:s_numberOfSamples,:),...
                            m_positions(1:s_numberOfSamples,:)]...
                            = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                        for s_timeInd=2:s_maximumTime
                            %time t indices
                            v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                (s_timeInd)*s_numberOfVertices;
                            v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                (s_timeInd)*s_numberOfSamples;
                            m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                            for s_mtId=1:s_monteCarloSimulations
                                m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                                    ((s_timeInd-1)*s_numberOfVertices+...
                                    m_positions(v_timetIndicesForSamples,s_mtId));
                            end
                            
                        end
                        
                        
                        %% 5. KF estimate
                        kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                            't_previousMinimumSquaredError',t_initialSigma0,...
                            'm_previousEstimate',m_initialState);
                        for s_muInd=1:size(v_mu,2)
                            v_sigmaForKalman=sqrt((1:s_maximumTime)'*v_numberOfSamples*v_mu(s_muInd))';
                            v_normOFKFErrors=zeros(s_maximumTime,1);
                            v_normOFNotSampled=zeros(s_maximumTime,1);
                            for s_timeInd=1:s_maximumTime
                                %time t indices
                                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                    (s_timeInd)*s_numberOfVertices;
                                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                    (s_timeInd)*s_numberOfSamples;
                                
                                %samples and positions at time t
                                m_samplest=m_samples(v_timetIndicesForSamples,:);
                                m_positionst=m_positions(v_timetIndicesForSamples,:);
                                %estimate
                                
                                [m_prevEstimate,t_newMSE]=...
                                    kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                                    t_transitions(:,:,s_timeInd),...
                                    t_correlations(:,:,s_timeInd),v_sigmaForKalman(s_sampleInd,s_timeInd));
                                t_kfEstimate(s_scalingInd,s_sigmaInd,...
                                    s_muInd,s_sampleInd,v_timetIndicesForSignals,:)=m_prevEstimate;
                                % prepare KF for next iteration
                                kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                                kFOnGFunctionEstimator.m_previousEstimate=m_prevEstimate;
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                for s_mtind=1:s_monteCarloSimulations
                                    v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                                    v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                                        norm(squeeze(t_kfEstimate(s_scalingInd,s_sigmaInd,s_muInd,s_sampleInd,v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                        ,s_mtind))...
                                        -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                    v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                                        norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                end
                                
                                s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                                t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_timeInd, s_sampleInd)...
                                    =sum(v_normOFKFErrors(1:s_timeInd))/...
                                    s_summedNorm;
                                
                                
                                
                                
                                
                                
                                
                            end
                        end
                        
                    end
                end
            end
            
            
            
            plot(squeeze(t_kfEstimate(1,1,1,1,(0:s_maximumTime-1)*s_numberOfVertices+1,1)))
            for s_scalingInd=1:size(v_propagationWeight,2)
                for s_muInd=1:size(v_mu,2)
                    for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                        m_relativeErrorKf(s_scalingInd,(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd)=...
                            (t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_maximumTime, 1));
                        myLegend{(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd}=...
                            sprintf('KKF mu=%g, sigma=%g',v_mu(s_muInd),v_sigmaForDiffusion(s_sigmaInd));
                        
                    end
                end
            end
            
            F = F_figure('X',v_propagationWeight,'Y',m_relativeErrorKf',...
                'xlab','diagonal scaling','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
            %F.ylimit=[0 0.5];
            
            F.logx=1;
            F.caption=[	sprintf('regularization parameter mu=%g\n',v_mu),...
                sprintf(' diffusion parameter sigma=%g\n',v_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('sampling size =%g\n',v_samplePercentage)...
                sprintf('time samples =%g\n',s_maximumTime)...
                sprintf('hours per sample =%g\n',s_samplePeriod)...
                'sampling scheme : uniformly random at each t\n'];
            
            
            
        end
        function F = compute_fig_2221(obj,niter)
            F = obj.load_F_structure(2021);
            %F.ylimit=[0.1 0.4];
            %F.xlimit=[10 100];
            F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
                ,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Parameter selection'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare perfomance of Kalman filter up to time t as I
        %  change the parameters diffusion parameter reg parameter and
        %  weight of scaled Identity as time evolves and laplacian
        %  regularized kernel
        % sampling: same-set each t
        %Kernel used: Diffusion-Laplacian Kernel in space
        function F = compute_fig_2023(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=100 ;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=24;
            v_mu=[10^-7]; %opt
            v_sigmaForDiffusion=[0.5,1,1.5]; %opt
            v_sigmaForLaplReg=[0.5,1,1.5];
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            v_propagationWeight=[10^-10,10^-8,10^-6,10^-4,10^-2,1,10^2,10^4,10^6,10^8,10^10,10^12,10^14];
            %v_propagationWeight=[10^-3,10^-2,10^-1,1,10^1,10^2,10^3,10^4,5*10^4,10^5,5*10^5,10^6,5*10^6,10^7,5*10^7]; % weight of edges between the same node opt
            % 			v_propagationWeight=[10^-6,10^-5,10^-4,10^-3,...
            %                 10^-2,10^-1,1,10^1,10^2,10^3,10^4,10^5,10^6,10^7,10^8];
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            %
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            t_kfEstimateDF=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,size(v_numberOfSamples,2),s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_kfEstimateL=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,size(v_numberOfSamples,2),s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_relativeErrorKf=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,s_maximumTime,size(v_numberOfSamples,2));
            v_allPositions=(1:s_numberOfVertices)';
            
            for s_scalingInd=1:size(v_propagationWeight,2)
                s_propagationWeight=v_propagationWeight(s_scalingInd);
                m_timeAdjacency=s_propagationWeight*(eye(size(m_adjacency)));
                
                t_timeAdjacencyAtDifferentTimes=...
                    repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
                
                % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
                % 			graphT=graphGenerator.realization;
                %
                %% 2. choise of Kernel must be positive definite
                % diffusion kernel
                graph=Graph('m_adjacency',m_adjacency);
                for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                    %diffusion regularizaiton
                    diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_sigmaInd),...
                        'm_laplacian',graph.getLaplacian);
                    m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
                    t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
                    
                    %laplacian regularization
                    %epsilon = .01;
                    rLaplacianReg = @(lambda,epsilon)1+ lambda * epsilon^2;
                    h_rFun_inv = @(lambda) 1./rLaplacianReg(lambda,v_sigmaForLaplReg(s_sigmaInd));
                    kG = LaplacianKernel('m_laplacian',graph.getLaplacian,'h_r_inv',{h_rFun_inv});
                    m_laplacianKernel = kG.getKernelMatrix;
                    t_invSpatialLaplacianKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_laplacianKernel,m_timeAdjacency,s_maximumTime);
                    
                    %% generate transition, correlation matrices
                    m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
                    m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
                    t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
                    for s_ind=1:s_monteCarloSimulations
                        t_sigma0(:,:,s_ind)=m_sigma0;
                    end
                    [t_correlationsDF,t_transitionsDF]=KFonGSimulations.kernelRegressionRecursion...
                        (t_invSpatialDiffusionKernel...
                        ,-t_timeAdjacencyAtDifferentTimes...
                        ,s_maximumTime,s_numberOfVertices,m_sigma0);
                    [t_correlationsL,t_transitionsL]=KFonGSimulations.kernelRegressionRecursion...
                        (t_invSpatialLaplacianKernel...
                        ,-t_timeAdjacencyAtDifferentTimes...
                        ,s_maximumTime,s_numberOfVertices,m_sigma0);
                    %% 3. generate true signal
                    
                    m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
                    
                    m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
                    
                    
                    %%
                    
                    
                    
                    for s_sampleInd=1:size(v_numberOfSamples,2)
                        %% 4. generate observations
                        s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                        sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                        m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        
                        %Same sample locations needed for distributed algo
                        [m_samples(1:s_numberOfSamples,:),...
                            m_positions(1:s_numberOfSamples,:)]...
                            = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                        for s_timeInd=2:s_maximumTime
                            %time t indices
                            v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                (s_timeInd)*s_numberOfVertices;
                            v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                (s_timeInd)*s_numberOfSamples;
                            m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                            for s_mtId=1:s_monteCarloSimulations
                                m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                                    ((s_timeInd-1)*s_numberOfVertices+...
                                    m_positions(v_timetIndicesForSamples,s_mtId));
                            end
                            %                     [m_samples(v_timetIndicesForSamples,:),...
                            % 					m_positions(v_timetIndicesForSamples,:)]...
                            % 					= sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
                            
                        end
                        
                        %% 5. KF estimate
                        kFOnGFunctionEstimatorDF=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                            't_previousMinimumSquaredError',t_initialSigma0,...
                            'm_previousEstimate',m_initialState);
                        kFOnGFunctionEstimatorL=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                            't_previousMinimumSquaredError',t_initialSigma0,...
                            'm_previousEstimate',m_initialState);
                        for s_muInd=1:size(v_mu,2)
                            v_sigmaForKalman=sqrt((1:s_maximumTime)'*v_numberOfSamples*v_mu(s_muInd))';
                            v_normOFKFErrors=zeros(s_maximumTime,1);
                            v_normOFKFLErrors=zeros(s_maximumTime,1);
                            v_normOFNotSampled=zeros(s_maximumTime,1);
                            for s_timeInd=1:s_maximumTime
                                %time t indices
                                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                    (s_timeInd)*s_numberOfVertices;
                                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                    (s_timeInd)*s_numberOfSamples;
                                
                                %samples and positions at time t
                                m_samplest=m_samples(v_timetIndicesForSamples,:);
                                m_positionst=m_positions(v_timetIndicesForSamples,:);
                                %estimate
                                
                                [m_prevEstimateDF,t_newMSEDF]=...
                                    kFOnGFunctionEstimatorDF.oneStepKF(m_samplest,m_positionst,...
                                    t_transitionsDF(:,:,s_timeInd),...
                                    t_correlationsDF(:,:,s_timeInd),v_sigmaForKalman(s_sampleInd,s_timeInd));
                                t_kfEstimateDF(s_scalingInd,s_sigmaInd,...
                                    s_muInd,s_sampleInd,v_timetIndicesForSignals,:)=m_prevEstimateDF;
                                % prepare KF for next iteration
                                kFOnGFunctionEstimatorDF.t_previousMinimumSquaredError=t_newMSEDF;
                                kFOnGFunctionEstimatorDF.m_previousEstimate=m_prevEstimateDF;
                                
                                
                                [m_prevEstimateL,t_newMSEL]=...
                                    kFOnGFunctionEstimatorL.oneStepKF(m_samplest,m_positionst,...
                                    t_transitionsL(:,:,s_timeInd),...
                                    t_correlationsL(:,:,s_timeInd),v_sigmaForKalman(s_sampleInd,s_timeInd));
                                t_kfEstimateL(s_scalingInd,s_sigmaInd,...
                                    s_muInd,s_sampleInd,v_timetIndicesForSignals,:)=m_prevEstimateL;
                                % prepare KF for next iteration
                                kFOnGFunctionEstimatorL.t_previousMinimumSquaredError=t_newMSEL;
                                kFOnGFunctionEstimatorL.m_previousEstimate=m_prevEstimateL;
                                
                                
                                
                                
                                
                                
                                
                                
                                for s_mtind=1:s_monteCarloSimulations
                                    %v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                                    v_notSampledPositions=v_allPositions;
                                    v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                                        norm(squeeze(t_kfEstimateDF(s_scalingInd,s_sigmaInd,s_muInd,s_sampleInd,v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                        ,s_mtind))...
                                        -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    v_normOFKFLErrors(s_timeInd)=v_normOFKFLErrors(s_timeInd)+...
                                        norm(squeeze(t_kfEstimateL(s_scalingInd,s_sigmaInd,s_muInd,s_sampleInd,v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                        ,s_mtind))...
                                        -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                                        norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                end
                                
                                s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                                t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_timeInd, s_sampleInd)...
                                    =sum(v_normOFKFErrors(1:s_timeInd))/...
                                    s_summedNorm;
                                t_relativeErrorKfL(s_scalingInd,s_sigmaInd,s_muInd,s_timeInd, s_sampleInd)...
                                    =sum(v_normOFKFLErrors(1:s_timeInd))/...
                                    s_summedNorm;
                                
                                
                                
                                
                                
                                
                                
                            end
                        end
                        
                    end
                end
            end
            
            
            
            plot(squeeze(t_kfEstimateDF(1,1,1,1,(0:s_maximumTime-1)*s_numberOfVertices+1,1)))
            for s_scalingInd=1:size(v_propagationWeight,2)
                for s_muInd=1:size(v_mu,2)
                    for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                        m_relativeErrorKf(s_scalingInd,(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd)=...
                            (t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_maximumTime, 1));
                        m_relativeErrorKfL(s_scalingInd,(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd)=...
                            (t_relativeErrorKfL(s_scalingInd,s_sigmaInd,s_muInd,s_maximumTime, 1));
                        myLegendDF{(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd}=...
                            sprintf('KKF-DF, sigma=%g',v_sigmaForDiffusion(s_sigmaInd));
                        myLegendL{(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd}=...
                            sprintf('KKF-L, sigma=%g',v_sigmaForLaplReg(s_sigmaInd));
                    end
                end
            end
            
            F = F_figure('X',v_propagationWeight,'Y',[m_relativeErrorKfL,m_relativeErrorKf]',...
                'xlab','diagonal scaling','ylab','NMSE','tit',...
                sprintf('Title'),'leg',[myLegendL,myLegendDF]);
            %F.ylimit=[0 0.5];
            
            F.logx=1;
            F.caption=[	sprintf('regularization parameter mu=%g\n',v_mu),...
                sprintf(' diffusion parameter sigma=%g\n',v_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('sampling size =%g\n',v_samplePercentage)...
                sprintf('time samples =%g\n',s_maximumTime)...
                sprintf('hours per sample =%g\n',s_samplePeriod)...
                'sampling scheme : same vertices at each t\n'];
            
            
            
        end
        function F = compute_fig_2223(obj,niter)
            F = obj.load_F_structure(2023);
            F.styles = {'--s','--^','--o',':s',':^',':o'};
            % % 			 Columns 1 through 5
            % %
            % %     'KKF-DF, sigma=0.4'    'KKF-DF, sigma=0.6'    'KKF-DF, sigma=0.8'    'KKF-DF, sigma=1'    'KKF-DF, sigma=1.2'
            % %
            % %   Columns 6 through 10
            % %
            % %     'KKF-DF, sigma=1.4'    'KKF-L, sigma=0.4'    'KKF-L, sigma=0.6'    'KKF-L, sigma=0.8'    'KKF-L, sigma=1'
            % %
            % %   Columns 11 through 12
            % %
            % %     'KKF-L, sigma=1.2'    'KKF-L, sigma=1.4'
            %              s_indaux=0;
            %             for s_ind=1:size(F.Y,1)
            % 			  if mod(s_ind,2)==0
            % 				  s_indaux=s_indaux+1;
            % 			      auxL{s_indaux}=F.leg{s_ind};
            % 				  auxY(s_indaux,:)=F.Y(s_ind,:);
            % 			  end
            % 			end
            %             F.leg=auxL;
            % 			F.Y=auxY;
            %
            % 			aux=F.Y(1:3,:);
            %             F.Y(1:3,:)=F.Y(4:6,:);
            % 			F.Y(4:6,:)=aux;
            %F.logy = 1;
            
            F.leg{4}='KKR-DF, \sigma=0.5';
            F.leg{5}='KKR-DF, \sigma=1';
            F.leg{6}='KKR-DF, \sigma=1.5';
            F.leg{1}= 'KKR-L, \sigma=0.5';
            F.leg{2}='KKR-L, \sigma=1';
            F.leg{3}='KKR-L, \sigma=1.5';
            %F.X=[10^-4,10^-3,...
            %   10^-2,10^-1,1,10^1,10^2,10^3,10^4,10^5,10^6,10^7,10^8];
            F.xlimit=[10^-10 10^12];
            F.leg_pos='northeast';
            %F.logx=0;
            F.ylab='NMSE';
            F.xlab='scaling parameter';
            
            %F.pos=[680 729 509 249];
            F.tit='';
        end
        function F = compute_fig_2323(obj,niter)
            F = obj.load_F_structure(2223);
            F.styles = {'--s','--^','--o',':s',':^',':o'};
            % % 			 Columns 1 through 5
            % %
            % %     'KKF-DF, sigma=0.4'    'KKF-DF, sigma=0.6'    'KKF-DF, sigma=0.8'    'KKF-DF, sigma=1'    'KKF-DF, sigma=1.2'
            % %
            % %   Columns 6 through 10
            % %
            % %     'KKF-DF, sigma=1.4'    'KKF-L, sigma=0.4'    'KKF-L, sigma=0.6'    'KKF-L, sigma=0.8'    'KKF-L, sigma=1'
            % %
            % %   Columns 11 through 12
            % %
            % %     'KKF-L, sigma=1.2'    'KKF-L, sigma=1.4'
            %              s_indaux=0;
            %             for s_ind=1:size(F.Y,1)
            % 			  if mod(s_ind,2)==0
            % 				  s_indaux=s_indaux+1;
            % 			      auxL{s_indaux}=F.leg{s_ind};
            % 				  auxY(s_indaux,:)=F.Y(s_ind,:);
            % 			  end
            % 			end
            %             F.leg=auxL;
            % 			F.Y=auxY;
            %
            % 			aux=F.Y(1:3,:);
            %             F.Y(1:3,:)=F.Y(4:6,:);
            % 			F.Y(4:6,:)=aux;
            %F.logy = 1;
            
            F.leg{4}='KKR-DF, \sigma=0.5';
            F.leg{5}='KKR-DF, \sigma=1';
            F.leg{6}='KKR-DF, \sigma=1.5';
            F.leg{1}= 'KKR-L, \sigma=0.5';
            F.leg{2}='KKR-L, \sigma=1';
            F.leg{3}='KKR-L, \sigma=1.5';
            % 			F.X=[10^-8,10^-7,10^-6,10^-5,10^-4,10^-3,...
            %                 10^-2,10^-1,1,10^1,10^2,10^3,10^4,10^5,10^6,10^7,10^8];
            % 			F.xlimit=[10^-4 10^8];
            F.leg_pos='northeast';
            %F.logx=0;
            F.ylab='NMSE';
            F.xlab='scaling parameter';
            
            F.leg_pos_vec = [0.68 0.118 0.25 0.805];
            F.tit='';
        end
        
        
        
        
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare perfomance of Kalman filter up to time t as I
        %  change the parameters diffusion parameter reg parameter and
        %  weight of scaled Identity as time evolves and laplacian
        %  regularized kernel
        % sampling: uniformly random each t
        %Kernel used: Diffusion-Laplacian Kernel in space
        function F = compute_fig_2024(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=100 ;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=8;
            v_mu=[10^-7]; %opt
            v_sigmaForDiffusion=[0.5,1,1.5]; %opt
            v_sigmaForLaplReg=[0.5,1,1.5];
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            %v_propagationWeight=[10^-6,10^-4,10^-2,1,10^2,10^4,10^6];
            v_propagationWeight=[10^-3,10^-2,10^-1,1,10^1,10^2,10^3,10^4,5*10^4,10^5,5*10^5,10^6,5*10^6,10^7,5*10^7]; % weight of edges between the same node opt
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
            %
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_temperatureTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_temperatureTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_temperatureTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_temperatureTimeSeriesSampled=m_temperatureTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            t_kfEstimateDF=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,size(v_numberOfSamples,2),s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_kfEstimateL=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,size(v_numberOfSamples,2),s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_relativeErrorKf=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,s_maximumTime,size(v_numberOfSamples,2));
            v_allPositions=(1:s_numberOfVertices)';
            
            for s_scalingInd=1:size(v_propagationWeight,2)
                s_propagationWeight=v_propagationWeight(s_scalingInd);
                m_timeAdjacency=s_propagationWeight*(eye(size(m_adjacency)));
                
                t_timeAdjacencyAtDifferentTimes=...
                    repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
                
                % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
                % 			graphT=graphGenerator.realization;
                %
                %% 2. choise of Kernel must be positive definite
                % diffusion kernel
                graph=Graph('m_adjacency',m_adjacency);
                for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                    %diffusion regularizaiton
                    diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_sigmaInd),...
                        'm_laplacian',graph.getLaplacian);
                    m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
                    t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
                    
                    %laplacian regularization
                    %epsilon = .01;
                    rLaplacianReg = @(lambda,epsilon)1+ lambda * epsilon^2;
                    h_rFun_inv = @(lambda) 1./rLaplacianReg(lambda,v_sigmaForLaplReg(s_sigmaInd));
                    kG = LaplacianKernel('m_laplacian',graph.getLaplacian,'h_r_inv',{h_rFun_inv});
                    m_laplacianKernel = kG.getKernelMatrix;
                    t_invSpatialLaplacianKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_laplacianKernel,m_timeAdjacency,s_maximumTime);
                    
                    %% generate transition, correlation matrices
                    m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
                    m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
                    t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
                    for s_ind=1:s_monteCarloSimulations
                        t_sigma0(:,:,s_ind)=m_sigma0;
                    end
                    [t_correlationsDF,t_transitionsDF]=KFonGSimulations.kernelRegressionRecursion...
                        (t_invSpatialDiffusionKernel...
                        ,-t_timeAdjacencyAtDifferentTimes...
                        ,s_maximumTime,s_numberOfVertices,m_sigma0);
                    [t_correlationsL,t_transitionsL]=KFonGSimulations.kernelRegressionRecursion...
                        (t_invSpatialLaplacianKernel...
                        ,-t_timeAdjacencyAtDifferentTimes...
                        ,s_maximumTime,s_numberOfVertices,m_sigma0);
                    %% 3. generate true signal
                    
                    m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
                    
                    m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
                    
                    
                    %%
                    
                    
                    
                    for s_sampleInd=1:size(v_numberOfSamples,2)
                        %% 4. generate observations
                        s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                        sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                        m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        
                        %Same sample locations needed for distributed algo
                        [m_samples(1:s_numberOfSamples,:),...
                            m_positions(1:s_numberOfSamples,:)]...
                            = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                        for s_timeInd=2:s_maximumTime
                            %time t indices
                            v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                (s_timeInd)*s_numberOfVertices;
                            v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                (s_timeInd)*s_numberOfSamples;
                            m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                            for s_mtId=1:s_monteCarloSimulations
                                m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                                    ((s_timeInd-1)*s_numberOfVertices+...
                                    m_positions(v_timetIndicesForSamples,s_mtId));
                            end
                            %                     [m_samples(v_timetIndicesForSamples,:),...
                            % 					m_positions(v_timetIndicesForSamples,:)]...
                            % 					= sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
                            
                        end
                        
                        %% 5. KF estimate
                        kFOnGFunctionEstimatorDF=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                            't_previousMinimumSquaredError',t_initialSigma0,...
                            'm_previousEstimate',m_initialState);
                        kFOnGFunctionEstimatorL=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                            't_previousMinimumSquaredError',t_initialSigma0,...
                            'm_previousEstimate',m_initialState);
                        for s_muInd=1:size(v_mu,2)
                            v_sigmaForKalman=sqrt((1:s_maximumTime)'*v_numberOfSamples*v_mu(s_muInd))';
                            v_normOFKFErrors=zeros(s_maximumTime,1);
                            v_normOFKFLErrors=zeros(s_maximumTime,1);
                            v_normOFNotSampled=zeros(s_maximumTime,1);
                            for s_timeInd=1:s_maximumTime
                                %time t indices
                                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                    (s_timeInd)*s_numberOfVertices;
                                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                    (s_timeInd)*s_numberOfSamples;
                                
                                %samples and positions at time t
                                m_samplest=m_samples(v_timetIndicesForSamples,:);
                                m_positionst=m_positions(v_timetIndicesForSamples,:);
                                %estimate
                                
                                [m_prevEstimateDF,t_newMSEDF]=...
                                    kFOnGFunctionEstimatorDF.oneStepKF(m_samplest,m_positionst,...
                                    t_transitionsDF(:,:,s_timeInd),...
                                    t_correlationsDF(:,:,s_timeInd),v_sigmaForKalman(s_sampleInd,s_timeInd));
                                t_kfEstimateDF(s_scalingInd,s_sigmaInd,...
                                    s_muInd,s_sampleInd,v_timetIndicesForSignals,:)=m_prevEstimateDF;
                                % prepare KF for next iteration
                                kFOnGFunctionEstimatorDF.t_previousMinimumSquaredError=t_newMSEDF;
                                kFOnGFunctionEstimatorDF.m_previousEstimate=m_prevEstimateDF;
                                
                                
                                [m_prevEstimateL,t_newMSEL]=...
                                    kFOnGFunctionEstimatorL.oneStepKF(m_samplest,m_positionst,...
                                    t_transitionsL(:,:,s_timeInd),...
                                    t_correlationsL(:,:,s_timeInd),v_sigmaForKalman(s_sampleInd,s_timeInd));
                                t_kfEstimateL(s_scalingInd,s_sigmaInd,...
                                    s_muInd,s_sampleInd,v_timetIndicesForSignals,:)=m_prevEstimateL;
                                % prepare KF for next iteration
                                kFOnGFunctionEstimatorL.t_previousMinimumSquaredError=t_newMSEL;
                                kFOnGFunctionEstimatorL.m_previousEstimate=m_prevEstimateL;
                                
                                
                                
                                
                                
                                
                                for s_mtind=1:s_monteCarloSimulations
                                    v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                                    v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                                        norm(squeeze(t_kfEstimateDF(s_scalingInd,s_sigmaInd,s_muInd,s_sampleInd,v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                        ,s_mtind))...
                                        -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    v_normOFKFLErrors(s_timeInd)=v_normOFKFLErrors(s_timeInd)+...
                                        norm(squeeze(t_kfEstimateL(s_scalingInd,s_sigmaInd,s_muInd,s_sampleInd,v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                        ,s_mtind))...
                                        -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                                        norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                end
                                
                                s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                                t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_timeInd, s_sampleInd)...
                                    =sum(v_normOFKFErrors(1:s_timeInd))/...
                                    s_summedNorm;
                                t_relativeErrorKfL(s_scalingInd,s_sigmaInd,s_muInd,s_timeInd, s_sampleInd)...
                                    =sum(v_normOFKFLErrors(1:s_timeInd))/...
                                    s_summedNorm;
                                
                                
                                
                                
                                
                                
                                
                            end
                        end
                        
                    end
                end
            end
            
            
            
            plot(squeeze(t_kfEstimateDF(1,1,1,1,(0:s_maximumTime-1)*s_numberOfVertices+1,1)))
            for s_scalingInd=1:size(v_propagationWeight,2)
                for s_muInd=1:size(v_mu,2)
                    for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                        m_relativeErrorKf(s_scalingInd,(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd)=...
                            (t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_maximumTime, 1));
                        m_relativeErrorKfL(s_scalingInd,(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd)=...
                            (t_relativeErrorKfL(s_scalingInd,s_sigmaInd,s_muInd,s_maximumTime, 1));
                        myLegendDF{(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd}=...
                            sprintf('KKF-DF, sigma=%g',v_sigmaForDiffusion(s_sigmaInd));
                        myLegendL{(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd}=...
                            sprintf('KKF-L, sigma=%g',v_sigmaForLaplReg(s_sigmaInd));
                    end
                end
            end
            
            F = F_figure('X',v_propagationWeight,'Y',[m_relativeErrorKfL,m_relativeErrorKf]',...
                'xlab','diagonal scaling','ylab','NMSE','tit',...
                sprintf('Title'),'leg',[myLegendL,myLegendDF]);
            %F.ylimit=[0 0.5];
            
            F.logx=1;
            F.caption=[	sprintf('regularization parameter mu=%g\n',v_mu),...
                sprintf(' diffusion parameter sigma=%g\n',v_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('sampling size =%g\n',v_samplePercentage)...
                sprintf('time samples =%g\n',s_maximumTime)...
                sprintf('hours per sample =%g\n',s_samplePeriod)...
                'sampling scheme : uniformly random at each t\n'];
            
            
            
        end
        function F = compute_fig_2224(obj,niter)
            F = obj.load_F_structure(2023);
            F.styles = {'--s','--^','--o',':s',':^',':o'};
            % % 			 Columns 1 through 5
            % %
            % %     'KKF-DF, sigma=0.4'    'KKF-DF, sigma=0.6'    'KKF-DF, sigma=0.8'    'KKF-DF, sigma=1'    'KKF-DF, sigma=1.2'
            % %
            % %   Columns 6 through 10
            % %
            % %     'KKF-DF, sigma=1.4'    'KKF-L, sigma=0.4'    'KKF-L, sigma=0.6'    'KKF-L, sigma=0.8'    'KKF-L, sigma=1'
            % %
            % %   Columns 11 through 12
            % %
            % %     'KKF-L, sigma=1.2'    'KKF-L, sigma=1.4'
            %              s_indaux=0;
            %             for s_ind=1:size(F.Y,1)
            % 			  if mod(s_ind,2)==0
            % 				  s_indaux=s_indaux+1;
            % 			      auxL{s_indaux}=F.leg{s_ind};
            % 				  auxY(s_indaux,:)=F.Y(s_ind,:);
            % 			  end
            % 			end
            %             F.leg=auxL;
            % 			F.Y=auxY;
            %
            % 			aux=F.Y(1:3,:);
            %             F.Y(1:3,:)=F.Y(4:6,:);
            % 			F.Y(4:6,:)=aux;
            %F.logy = 1;
            
            F.leg{4}='KKR-DE-DF, \sigma=0.5';
            F.leg{5}='KKR-DE-DF, \sigma=1';
            F.leg{6}='KKR-DE-DF, \sigma=1.5';
            F.leg{1}= 'KKR-DE-L, \sigma=0.5';
            F.leg{2}='KKR-DE-L, \sigma=1';
            F.leg{3}='KKR-DE-L, \sigma=1.5';
            F.X=[10^-3,10^-2,10^-1,1,10^1,10^2,10^3,10^4,5*10^4,10^5,5*10^5,10^6,5*10^6,10^7,5*10^7];
            F.xlimit=[10^-3 5*10^7];
            F.leg_pos='northeast';
            %F.logx=0;
            F.ylab='NMSE';
            F.xlab='scaling parameter';
            
            %F.pos=[680 729 509 249];
            F.tit='';
        end
        %%Simulation
        % Goal overall error Vs constant of scaled of diagonal identity
        % TODO check reguralization for online
        % TODO plot cumulative error in online instead of instanuous.
        % Check error for time evolving batch approach
        
        %% Real data simulations
        %  Data used: Kramer, M.A., Kolaczyk, E.D., and Kirsch, H.E. (2008). Emergent network
        %   topology at seizure onset in humans. Epilepsy Research,79, 173-186.
        %   measurements on a human patient with epilepsy.
        %   These are electrocorticogram (ECoG) data, in the
        %  form of time series at each of 76 electrodes.
        % We consider the case of tracking the brain signal in one
        % experiment of one individual
        %  Adjacency: obtained from data file 'sz1_ict' using partial correlation
        %  Goal: Compare Batch Approach up to time t
        %  with Bandlimited model at each time t
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_3001(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=15;
            % Each file is 4000 X 76 (i.e., 4000 time points by 76
            %electrodes). The sampling rate is 400 Hz, so each file contains 10
            % seconds of data recorded at the 76 electrodes.
            
            s_samplePeriod=5;
            s_mu=10^-6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.2:1);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=1; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            [m_adjacency,m_timeAdjacency,t_brainSignalTimeSeries] = readBrainSignalTimeEvolvingDataset;
            % take only the first brain signal to test
            m_brainSignalTimeSeries=t_brainSignalTimeSeries(:,:,2);
            
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            m_timeAdjacency=m_timeAdjacency/max(max(m_adjacency));
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_brainSignalTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_brainSignalTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=m_brainSignalTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_brainSignalTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_brainSignalTimeSeriesSampled=m_brainSignalTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            graphT=graphGenerator.realization;
            
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            s_sigma=0.4;
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigma,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime)
            
            extendedGraphKernel = ExtendedGraphKernel...
                ('t_invSpatialKernel',t_invDiffusionKernel,...
                't_invTemporalKernel',-t_timeAdjacencyAtDifferentTimes);
            
            m_extendedGraphKernel=extendedGraphKernel.generateKernelMatrix;
            % make kernel great again
            s_beta=0;
            m_extendedGraphKernel=m_extendedGraphKernel+s_beta*eye(size(m_extendedGraphKernel));
            [~,p1]=chol(m_extendedGraphKernel);%check if PD
            [~,p2]=chol(m_extendedGraphKernel((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
                s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
                *s_numberOfVertices));
            if (p1+p2~=0)
                assert(1==0,'Not Positive Definite Kernel')
            end
            
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_brainSignalTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    [m_samples(v_timetIndicesForSamples,:),...
                        m_positions(v_timetIndicesForSamples,:)]...
                        = sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
                end
                
                %% 5. batch estimate
                nonParametricBatchGraphFunctionEstimator=NonParametricBatchGraphFunctionEstimator...
                    ('m_kernels',m_extendedGraphKernel,...
                    's_mu',s_mu,'s_maximumTime',s_maximumTime);
                t_batchEstimate(:,:,s_sampleInd)=nonParametricBatchGraphFunctionEstimator.estimate...
                    (m_samples,m_positions,s_numberOfSamples);
                
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                myLegend={};
                v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_extendedAdjancency=graphT.m_adjacency;
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=m_extendedAdjancency(v_timetIndicesForSignals,...
                            v_timetIndicesForSignals);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    v_relativeErrorBatch(s_sampleInd)=norm(t_batchEstimate(:,:,s_sampleInd)...
                        -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                    
                    m_relativeErrorbandLimitedEstimate(s_sampleInd,s_bandInd)...
                        =norm(t_bandLimitedEstimate(:,:,s_sampleInd,s_bandInd)...
                        -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                    
                    myLegend{s_bandInd}=strcat('Bandlimited  ',...
                        sprintf(' W=%g',s_bandwidth));
                end
            end
            %% 7. measure difference
            
            myLegend{s_bandInd+1}='Batch approach';
            F = F_figure('X',v_numberOfSamples,'Y',[m_relativeErrorbandLimitedEstimate,...
                v_relativeErrorBatch']',...
                'xlab','Number of observed vertices (S)','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
            
        end
        function F = compute_fig_3201(obj,niter)
            F = obj.load_F_structure(3001);
            F.ylimit=[0 2];
            %F.xlimit=[10 100];
            F.styles = {'-','--','-o','-x','--^','--','--'};
            F.pos=[680 729 509 249];
            F.tit='Brain Signal'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        
        %% Real data simulations
        %  Data used: Kramer, M.A., Kolaczyk, E.D., and Kirsch, H.E. (2008). Emergent network
        %  Adjacency: obtained from Yannings svar method
        %  Time adjacency also provided by yannings svar
        %  Goal: Compare Kalman filter up to time t Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model at each time t and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_3002(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=200;
            % Each file is 4000 X 76 (i.e., 4000 time points by 76
            %electrodes). The sampling rate is 400 Hz, so each file contains 10
            % seconds of data recorded at the 76 electrodes.
            
            s_samplePeriod=2;
            s_mu=10^-2;
            s_sigmaForDiffusion=0.4;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.1:0.1);
            v_bandwidthPercentage=(0.01:0.02:0.01);
            v_propagationWeight=1; % weight of edges between the same node
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            [m_adjacency,m_timeAdjacency,t_brainSignalTimeSeries] = readBrainSignalTimeEvolvingDataset;
            % take only the first brain signal to test of the pre period..
            % smoother...
            
            m_brainSignalTimeSeries=t_brainSignalTimeSeries(:,:,9);
            %m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            %threshold
            m_adjacency(m_adjacency<10^-2)=0;
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency)); %similar
            
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(t_brainSignalTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            t_brainSignalTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_temperatureTimeSeries=t_brainSignalTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_temperatureTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                t_brainSignalTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            t_brainSignalTimeSeriesSampled=t_brainSignalTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            m_diffusionKernel=m_diffusionKernel+0.05*eye(size(m_diffusionKernel));
            % need each spatial kernel invertible
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(t_brainSignalTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samples(v_timetIndicesForSamples,:)=m_samples(1:s_numberOfSamples,:);
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 7.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
            end
            
            
            %% 8. measure difference
            
            t_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2),size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            t_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2),size(v_bandwidth,2));
            for s_sampleInd=1:size(v_numberOfSamples,2)
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =norm(t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd)...
                        -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                        norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                    for s_bandInd=1:size(v_bandwidth,2)
                        s_bandwidth=v_bandwidth(s_bandInd);
                        t_relativeErrorDistr(s_timeInd,s_sampleInd,s_bandInd)=...
                            norm( t_distrEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)...
                            -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                            norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                        t_relativeErrorbandLimitedEstimate(s_timeInd,s_sampleInd,s_bandInd)...
                            =norm(t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)...
                            -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                            norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                        
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('Bandlimited, ',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                    end
                    myLegendKF{s_sampleInd}=strcat('Kernel Kalman Filter, ',...
                        sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                    
                end
            end
            myLegend=[myLegendDLSR myLegendBan myLegendKF];
            F = F_figure('X',(1:s_maximumTime),'Y',[t_relativeErrorDistr(:,:,1)...
                ,t_relativeErrorbandLimitedEstimate(:,:,1),...
                m_relativeErrorKf]',...
                'xlab','Time evolution','ylab','NMSE','tit',...
                sprintf('Brain Signal'),'leg',myLegend);
        end
        function F = compute_fig_3202(obj,niter)
            F = obj.load_F_structure(3002);
            F.ylimit=[0 2];
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Brain Signal'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        
        %% Real data simulations
        %  Data used: BRAIN
        %  Adjacency: obtained from Yannings svar method WIth Meng cov
        %  method
        %  Time adjacency also provided by yannings svar
        %  Goal: Compare Kalman filter up to time t Plot reconstruct error
        %  as time evolves
        %  Kernel used: Diffusion Kernel in space
        % TODO use as spatial kernel the covariance the signal
        % Try greater scaling for off diagonal elements
        function F = compute_fig_3003(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=100;
            % Each file is 4000 X 76 (i.e., 4000 time points by 76
            %electrodes). The sampling rate is 400 Hz, so each file contains 10
            % seconds of data recorded at the 76 electrodes.
            
            s_samplePeriod=1;
            s_mu=10^-5;
            s_sigmaForDiffusion=1.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.2:0.2:0.2);
            v_bandwidthPercentage=(0.01:0.02:0.01);
            v_propagationWeight=1; % weight of edges between the same node
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            
            [m_adjacency,m_timeAdjacency,t_brainSignalTimeSeries] = readBrainSignalTimeEvolvingDataset;
            % take only the first brain signal to test of the pre period..
            % smoother...
            
            m_brainSignalTimeSeries=t_brainSignalTimeSeries(:,:,9);
            % data normalization
            v_mean = mean(m_brainSignalTimeSeries,2);
            v_std = std(m_brainSignalTimeSeries')';
            % 			m_brainSignalTimeSeries = diag(1./v_std)*(m_brainSignalTimeSeries...
            %                 - v_mean*ones(1,size(m_brainSignalTimeSeries,2)));
            %             m_adjacency(m_adjacency<0)=0;
            %             m_adjacency=m_adjacency-diag(diag(m_adjacency));
            
            % covariance of normalized data
            m_covInv = MultikernelSimulations.learnInverseCov( cov(m_brainSignalTimeSeries') , m_adjacency );
            
            % approximation of inverse covariance via constrained Laplacian
            % 			m_covInv = inv(C);
            m_constrainedLaplacian = MultikernelSimulations.approximateWithLaplacian(m_covInv,m_adjacency);
            m_constrainedLaplacian=full(m_constrainedLaplacian);
            m_adjacency=-m_constrainedLaplacian+diag(diag(m_constrainedLaplacian));
            
            %m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            %threshold
            %m_adjacency(m_adjacency<10^-2)=0;
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency)); %similar
            
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_brainSignalTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_brainSignalTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_brainSignalTimeSeries=m_brainSignalTimeSeries(s_vertInd,:);
                v_brainSignalTimeSeriesSampledWhole=...
                    v_brainSignalTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_brainSignalTimeSeriesSampled(s_vertInd,:)=...
                    v_brainSignalTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_brainSignalTimeSeriesSampled=m_brainSignalTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            %check SOMETHING WRONG
            m_diffusionKernel=m_diffusionKernel+0.05*eye(size(m_diffusionKernel));
            % need each spatial kernel invertible
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_brainSignalTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    
                    
                    [m_samples(v_timetIndicesForSamples,:),...
                        m_positions(v_timetIndicesForSamples,:)]...
                        = sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                
            end
            
            
            %% 8. measure difference
            
            t_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2),size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            t_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2),size(v_bandwidth,2));
            for s_sampleInd=1:size(v_numberOfSamples,2)
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =norm(t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd)...
                        -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                        norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                    for s_bandInd=1:size(v_bandwidth,2)
                        s_bandwidth=v_bandwidth(s_bandInd);
                        
                        t_relativeErrorbandLimitedEstimate(s_timeInd,s_sampleInd,s_bandInd)...
                            =norm(t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)...
                            -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                            (s_timeInd*s_numberOfVertices);
                        norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                        
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('Bandlimited, ',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                    end
                    myLegendKF{s_sampleInd}=strcat('Kernel Kalman Filter, ',...
                        sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                    
                end
            end
            myLegend=[myLegendBan myLegendKF];
            F = F_figure('X',(1:s_maximumTime),'Y',[t_relativeErrorbandLimitedEstimate(:,:,1),...
                m_relativeErrorKf]',...
                'xlab','Time evolution','ylab','NMSE','tit',...
                sprintf('Brain Signal'),'leg',myLegend);
        end
        function F = compute_fig_3203(obj,niter)
            F = obj.load_F_structure(3003);
            F.ylimit=[0 2];
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Brain Signal'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        %% Real data simulations
        %  Data used: Kramer, M.A., Kolaczyk, E.D., and Kirsch, H.E. (2008). Emergent network
        %  Goal: Compare perfomance of Kalman filter up to time t as I
        %  change the parameters diffusion parameter reg parameter and
        %  weight of scaled Identity as time evolves
        %  Sample set uniformly at random every time.
        %  Kernel used: Diffusion Kernel in space
        % WHY Not PSD ? when I add the diagonal scaling the estimation is
        % not longer the online KRR that is why problem... solution ?
        
        function F = compute_fig_3004(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=100;
            s_samplePeriod=1;
            v_mu=[10^-8];% opt
            v_sigmaForDiffusion=[0.05]; %opt
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.1:0.1);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=[0.1]; % weight of edges between the same node opt
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            [m_adjacency,m_timeAdjacency,t_brainSignalTimeSeries] = readBrainSignalTimeEvolvingDataset;
            % take only the first brain signal to test of the pre period..
            % smoother...
            
            m_brainSignalTimeSeries=t_brainSignalTimeSeries(:,:,9);
            %m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            %threshold
            %m_adjacency(m_adjacency<10^-2)=0;
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            %select a subset of measurements
            s_totalTimeSamples=size(m_brainSignalTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_brainSignalTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_brainSignalTimeSeries=m_brainSignalTimeSeries(s_vertInd,:);
                v_brainSignalTimeSeriesSampledWhole=...
                    v_brainSignalTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_brainSignalTimeSeriesSampled(s_vertInd,:)=...
                    v_brainSignalTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_brainSignalTimeSeriesSampled=m_brainSignalTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            % 			t_spaceAdjacencyAtDifferentTimes=...
            % 				repmat(m_adjacency,[1,1,s_maximumTime]);
            
            t_kfEstimate=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,size(v_numberOfSamples,2),s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_relativeErrorKf=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,s_maximumTime,size(v_numberOfSamples,2));
            
            for s_scalingInd=1:size(v_propagationWeight,2)
                s_propagationWeight=v_propagationWeight(s_scalingInd);
                m_timeAdjacency=s_propagationWeight*eye(size(m_adjacency));
                
                t_timeAdjacencyAtDifferentTimes=...
                    repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
                
                % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
                % 			graphT=graphGenerator.realization;
                %
                %% 2. choise of Kernel must be positive definite
                % diffusion kernel
                graph=Graph('m_adjacency',m_adjacency);
                for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                    diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_sigmaInd),...
                        'm_laplacian',graph.getLaplacian);
                    m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
                    t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
                    %% generate transition, correlation matrices
                    m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
                    m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
                    t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
                    for s_ind=1:s_monteCarloSimulations
                        t_sigma0(:,:,s_ind)=m_sigma0;
                    end
                    [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                        (t_invSpatialDiffusionKernel...
                        ,-t_timeAdjacencyAtDifferentTimes...
                        ,s_maximumTime,s_numberOfVertices,m_sigma0);
                    
                    %% 3. generate true signal
                    
                    m_graphFunction=reshape(m_brainSignalTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
                    
                    m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
                    
                    
                    %%
                    
                    
                    
                    for s_sampleInd=1:size(v_numberOfSamples,2)
                        %% 4. generate observations
                        s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                        sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                        m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        %Diffenert sample locations
                        
                        for s_timeInd=1:s_maximumTime
                            %time t indices
                            v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                (s_timeInd)*s_numberOfVertices;
                            v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                (s_timeInd)*s_numberOfSamples;
                            
                            [m_samples(v_timetIndicesForSamples,:),...
                                m_positions(v_timetIndicesForSamples,:)]...
                                = sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
                            
                        end
                        % % % % 						%Same sample locations needed for distributed algo
                        % % % % 				[m_samples(1:s_numberOfSamples,:),...
                        % % % % 					m_positions(1:s_numberOfSamples,:)]...
                        % % % % 					= sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                        % % % % 				for s_timeInd=2:s_maximumTime
                        % % % % 					%time t indices
                        % % % % 					v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        % % % % 						(s_timeInd)*s_numberOfVertices;
                        % % % % 					v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        % % % % 						(s_timeInd)*s_numberOfSamples;
                        % % % %
                        % % % % 					m_samples(v_timetIndicesForSamples,:)=m_samples(1:s_numberOfSamples,:);
                        % % % % 					m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                        % % % %
                        % % % % 				end
                        
                        %% 5. KF estimate
                        kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                            't_previousMinimumSquaredError',t_initialSigma0,...
                            'm_previousEstimate',m_initialState);
                        for s_muInd=1:size(v_mu,2)
                            v_sigmaForKalman=sqrt((1:s_maximumTime)'*v_numberOfSamples*v_mu(s_muInd))';
                            for s_timeInd=1:s_maximumTime
                                %time t indices
                                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                    (s_timeInd)*s_numberOfVertices;
                                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                    (s_timeInd)*s_numberOfSamples;
                                
                                %samples and positions at time t
                                m_samplest=m_samples(v_timetIndicesForSamples,:);
                                m_positionst=m_positions(v_timetIndicesForSamples,:);
                                %estimate
                                
                                [m_prevEstimate,t_newMSE]=...
                                    kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                                    t_transitions(:,:,s_timeInd),...
                                    t_correlations(:,:,s_timeInd),v_sigmaForKalman(s_sampleInd,s_timeInd));
                                t_kfEstimate(s_scalingInd,s_sigmaInd,...
                                    s_muInd,s_sampleInd,v_timetIndicesForSignals,:)=m_prevEstimate;
                                % prepare KF for next iteration
                                kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                                kFOnGFunctionEstimator.m_previousEstimate=m_prevEstimate;
                                
                                
                                
                                if s_timeInd>1
                                    s_prevTimeInd=s_timeInd-1;
                                else
                                    s_prevTimeInd=1;
                                end
                                
                                m_prevEstimate=t_kfEstimate(s_scalingInd,s_sigmaInd,s_muInd,s_sampleInd...
                                    ,v_timetIndicesForSignals,:);
                                m_prevEstimate=squeeze(m_prevEstimate);
                                t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_timeInd, s_sampleInd)...
                                    =t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_prevTimeInd, s_sampleInd)...
                                    +norm(m_prevEstimate...
                                    -m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                                    s_numberOfVertices%norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                                myLegendKF{s_scalingInd,s_sigmaInd,s_muInd, s_sampleInd}...
                                    =strcat('Kernel Kalman Filter, ',...
                                    sprintf('samples=%g, mu=%g, sigmaDif=%g, scaling=%g,',...
                                    v_numberOfSamples(s_sampleInd),v_mu(s_muInd),...
                                    v_sigmaForDiffusion(s_sigmaInd),v_propagationWeight(s_scalingInd)));
                                
                                
                            end
                        end
                        
                    end
                end
            end
            
            
            
            m_relativeErrorKf=squeeze(t_relativeErrorKf);
            m_normalizer=repmat((1:s_maximumTime)',1,min(size(m_relativeErrorKf)));
            m_relativeErrorKf=m_relativeErrorKf./m_normalizer;
            myLegendKF=squeeze(myLegendKF);
            % 			myLegendKF=reshape(myLegendKF,[1,size(v_propagationWeight,2)*...
            % 				size(v_sigmaForDiffusion,2)*size(v_mu,2)*size(v_numberOfSamples,2)]);
            myLegend=[myLegendKF];
            F = F_figure('X',(1:s_maximumTime),'Y',m_relativeErrorKf',...
                'xlab','Time evolution','ylab','NMSE','tit',...
                sprintf('Brain Signals'),'leg',myLegend);
        end
        function F = compute_fig_3204(obj,niter)
            F = obj.load_F_structure(3004);
            F.ylimit=[0 1];
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Brain Signals Tracking'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        %% Real data simulations
        %  Data used: Kramer, M.A., Kolaczyk, E.D., and Kirsch, H.E. (2008). Emergent network
        %  Goal: Compare perfomance of Kalman filter up to time t as I
        %  change the parameters diffusion parameter reg parameter and
        %  weight of scaled Identity as time evolves
        %  Sample set uniformly at random every time.
        %  Kernel used: Covariance Kernel in space
        function F = compute_fig_3005(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=2;
            s_samplePeriod=1;
            v_mu=[10^-8];% opt
            v_sigmaForDiffusion=[0.9,1.1]; %opt
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.2:0.2:0.2);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=[1]; % weight of edges between the same node opt
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            [m_adjacency,m_timeAdjacency,t_brainSignalTimeSeries] = readBrainSignalTimeEvolvingDataset;
            % take only the first brain signal to test of the pre period..
            % smoother...
            
            m_brainSignalTimeSeries=t_brainSignalTimeSeries(:,:,9);
            %m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            %threshold
            %m_adjacency(m_adjacency<10^-2)=0;
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            %select a subset of measurements
            s_totalTimeSamples=size(m_brainSignalTimeSeries,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_brainSignalTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_brainSignalTimeSeries=m_brainSignalTimeSeries(s_vertInd,:);
                v_brainSignalTimeSeriesSampledWhole=...
                    v_brainSignalTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_brainSignalTimeSeriesSampled(s_vertInd,:)=...
                    v_brainSignalTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_brainSignalTimeSeriesSampled=m_brainSignalTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            % 			t_spaceAdjacencyAtDifferentTimes=...
            % 				repmat(m_adjacency,[1,1,s_maximumTime]);
            
            t_kfEstimate=zeros(size(v_propagationWeight,2),size(v_mu,2)...
                ,size(v_numberOfSamples,2),s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_relativeErrorKf=zeros(size(v_propagationWeight,2),size(v_mu,2)...
                ,s_maximumTime,size(v_numberOfSamples,2));
            
            for s_scalingInd=1:size(v_propagationWeight,2)
                s_propagationWeight=v_propagationWeight(s_scalingInd);
                m_timeAdjacency=s_propagationWeight*eye(size(m_adjacency));
                
                t_timeAdjacencyAtDifferentTimes=...
                    repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
                
                % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
                % 			graphT=graphGenerator.realization;
                %
                %% 2. choise of Kernel must be positive definite
                % diffusion kernel
                graph=Graph('m_adjacency',m_adjacency);
                
                m_covarianceKernel=m_brainSignalTimeSeries*m_brainSignalTimeSeries'/size(m_brainSignalTimeSeries,2);
                %check expression again
                t_invSpatialCovarianceKernel=repmat(inv(m_covarianceKernel)+...
                    diag(sum(m_timeAdjacency)),[1,1,s_maximumTime]);
                m_invExtendedKernel=KFonGSimulations.createInvExtendedGraphKernel...
                    (t_invSpatialCovarianceKernel,-t_timeAdjacencyAtDifferentTimes);
                
                %%make kernel great again
                s_beta=0;
                m_invExtendedKernel=m_invExtendedKernel+s_beta*eye(size(m_invExtendedKernel));
                [~,p1]=chol(m_invExtendedKernel);%check if PD
                [~,p2]=chol(m_invExtendedKernel((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
                    s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
                    *s_numberOfVertices));
                if (p1+p2~=0)
                    assert(1==0,'Not Positive Definite Kernel')
                end
                t_eyeTens=repmat(eye(size(m_covarianceKernel)),[1,1,s_maximumTime]);
                t_invSpatialCovarianceKernel=t_invSpatialCovarianceKernel+s_beta*t_eyeTens;
                %% generate transition, correlation matrices
                m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
                m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
                t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
                for s_ind=1:s_monteCarloSimulations
                    t_sigma0(:,:,s_ind)=m_sigma0;
                end
                [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                    (t_invSpatialCovarianceKernel...
                    ,-t_timeAdjacencyAtDifferentTimes...
                    ,s_maximumTime,s_numberOfVertices,m_sigma0);
                
                %% 3. generate true signal
                
                m_graphFunction=reshape(m_brainSignalTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
                
                m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
                
                
                %%
                
                
                
                for s_sampleInd=1:size(v_numberOfSamples,2)
                    %% 4. generate observations
                    s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                    sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                    m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                    m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                    
                    % % % % 						%Same sample locations needed for distributed algo
                    % % % % 				[m_samples(1:s_numberOfSamples,:),...
                    % % % % 					m_positions(1:s_numberOfSamples,:)]...
                    % % % % 					= sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                    % % % % 				for s_timeInd=2:s_maximumTime
                    % % % % 					%time t indices
                    % % % % 					v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                    % % % % 						(s_timeInd)*s_numberOfVertices;
                    % % % % 					v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                    % % % % 						(s_timeInd)*s_numberOfSamples;
                    % % % %
                    % % % % 					m_samples(v_timetIndicesForSamples,:)=m_samples(1:s_numberOfSamples,:);
                    % % % % 					m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    % % % %
                    % % % % 				end
                    
                    %% 5. KF estimate
                    kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                        't_previousMinimumSquaredError',t_initialSigma0,...
                        'm_previousEstimate',m_initialState);
                    for s_muInd=1:size(v_mu,2)
                        v_sigmaForKalman=sqrt((1:s_maximumTime)'*v_numberOfSamples*v_mu(s_muInd))';
                        v_normOFKFErrors=zeros(s_maximumTime,1);
                        v_normOfKrrErrors=zeros(s_maximumTime,1);
                        v_normOFNotSampled=zeros(s_maximumTime,1);
                        v_allPositions=(1:s_numberOfVertices)';
                        for s_timeInd=1:s_maximumTime
                            %time t indices
                            v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                (s_timeInd)*s_numberOfVertices;
                            v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                (s_timeInd)*s_numberOfSamples;
                            
                            [m_samples(v_timetIndicesForSamples,:),...
                                m_positions(v_timetIndicesForSamples,:)]...
                                = sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
                            
                            
                            %samples and positions at time t
                            m_samplest=m_samples(v_timetIndicesForSamples,:);
                            m_positionst=m_positions(v_timetIndicesForSamples,:);
                            %estimate
                            
                            [m_prevEstimate,t_newMSE]=...
                                kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                                t_transitions(:,:,s_timeInd),...
                                t_correlations(:,:,s_timeInd),v_sigmaForKalman(s_sampleInd,s_timeInd));
                            t_kfEstimate(s_scalingInd,...
                                s_muInd,s_sampleInd,v_timetIndicesForSignals,:)=m_prevEstimate;
                            % prepare KF for next iteration
                            kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                            kFOnGFunctionEstimator.m_previousEstimate=m_prevEstimate;
                            
                            
                            
                            %% 6. Kernel Ridge Regression
                            
                            nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                                ('m_kernels',m_covarianceKernel,'s_lambda',v_mu(s_muInd));
                            
                            
                            
                            [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                                nonParametricGraphFunctionEstimator.estimate...
                                (m_samplest,m_positionst,v_mu(s_muInd));
                            
                            
                            
                            
                            for s_mtind=1:s_monteCarloSimulations
                                v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                                v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                                    norm(squeeze(t_kfEstimate(s_scalingInd,s_muInd,s_sampleInd,v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                    ,s_mtind))...
                                    -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                
                                v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                                    norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                                    norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                    ,s_mtind,s_sampleInd)...
                                    -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            end
                            
                            s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                            t_relativeErrorKf(s_scalingInd,s_muInd,s_timeInd, s_sampleInd)...
                                =sum(v_normOFKFErrors(1:s_timeInd))/...
                                s_summedNorm;
                            m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                                =sum(v_normOfKrrErrors(1:s_timeInd))/...
                                s_summedNorm;%s_timeInd*s_numberOfVertices;
                            
                            % 								m_prevEstimate=t_kfEstimate(s_scalingInd,s_muInd,s_sampleInd...
                            % 									,v_timetIndicesForSignals,:);
                            % 								m_prevEstimate=squeeze(m_prevEstimate);
                            % 								t_relativeErrorKf(s_scalingInd,s_muInd,s_timeInd, s_sampleInd)...
                            % 									=t_relativeErrorKf(s_scalingInd,s_muInd,s_prevTimeInd, s_sampleInd)...
                            % 								    +norm(m_prevEstimate...
                            % 									-m_graphFunction(v_timetIndicesForSignals,:),'fro')/...
                            % 									s_numberOfVertices%norm(m_graphFunction(v_timetIndicesForSignals,:),'fro');
                            myLegendKF{s_scalingInd,s_muInd, s_sampleInd}...
                                =strcat('Kernel Kalman Filter, ',...
                                sprintf('samples=%g, mu=%g, scaling=%g,',...
                                v_numberOfSamples(s_sampleInd),v_mu(s_muInd),...
                                v_propagationWeight(s_scalingInd)));
                            myLegendKRR{s_sampleInd}=strcat('KRR time agnostic, ',...
                                sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                            
                            
                        end
                        
                        
                    end
                end
            end
            
            
            
            m_relativeErrorKf=squeeze(t_relativeErrorKf);
            % 			m_normalizer=repmat((1:s_maximumTime)',1,min(size(m_relativeErrorKf)));
            % 			m_relativeErrorKf=m_relativeErrorKf./m_normalizer;
            myLegendKF=squeeze(myLegendKF);
            % 			myLegendKF=reshape(myLegendKF,[1,size(v_propagationWeight,2)*...
            % 				size(v_sigmaForDiffusion,2)*size(v_mu,2)*size(v_numberOfSamples,2)]);
            myLegend=[myLegendKF ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorKf'],...
                'xlab','Time evolution','ylab','NMSE','tit',...
                sprintf('Brain Signals'),'leg',myLegend);
        end
        function F = compute_fig_3205(obj,niter)
            F = obj.load_F_structure(3005);
            F.ylimit=[0 1];
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Brain Signals Tracking'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare perfomance of Kalman filter up to time t as I
        %  change the parameters diffusion parameter reg parameter and
        %  weight of scaled Identity as time evolves
        %  Kernel used: Diffusion Kernel in space
        
        function F = compute_fig_3009(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=100  ;
            
            s_samplePeriod=1;
            v_mu=[10^-6]; %opt
            v_sigmaForDiffusion=[3.4]; %opt
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.2:0.2:0.2);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=[0,0.001,0.01,0.1,1,10,100]; % weight of edges between the same node opt
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            [m_adjacency,m_timeAdjacency,t_brainSignalTimeSeries] = readBrainSignalTimeEvolvingDataset;
            % take only the first brain signal to test of the pre period..
            % smoother...
            m_brainSignalTimeSeries=t_brainSignalTimeSeries(:,:,9);
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_brainSignalTimeSeries,2);
            %               % data normalization
            % 			v_mean = mean(m_temperatureTimeSeries,2);
            % 			v_std = std(m_temperatureTimeSeries')';
            % 			m_temperatureTimeSeries = diag(1./v_std)*(m_temperatureTimeSeries...
            %                 - v_mean*ones(1,size(m_temperatureTimeSeries,2)));
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_brainSignalTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_brainSignalTimeSeries=m_brainSignalTimeSeries(s_vertInd,:);
                v_brainSignalTimeSeriesSampledWhole=...
                    v_brainSignalTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_brainSignalTimeSeriesSampled(s_vertInd,:)=...
                    v_brainSignalTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_brainSignalTimeSeriesSampled=m_brainSignalTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            % 			t_spaceAdjacencyAtDifferentTimes=...
            % 				repmat(m_adjacency,[1,1,s_maximumTime]);
            
            t_kfEstimate=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,size(v_numberOfSamples,2),s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_relativeErrorKf=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,s_maximumTime,size(v_numberOfSamples,2));
            v_allPositions=(1:s_numberOfVertices)';
            
            for s_scalingInd=1:size(v_propagationWeight,2)
                s_propagationWeight=v_propagationWeight(s_scalingInd);
                m_timeAdjacency=s_propagationWeight*(eye(size(m_adjacency)));
                
                t_timeAdjacencyAtDifferentTimes=...
                    repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
                
                % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
                % 			graphT=graphGenerator.realization;
                %
                %% 2. choise of Kernel must be positive definite
                % diffusion kernel
                graph=Graph('m_adjacency',m_adjacency);
                for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                    diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_sigmaInd),...
                        'm_laplacian',graph.getNormalizedLaplacian);
                    m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
                    %m_covarianceKernel=m_brainSignalTimeSeries*m_brainSignalTimeSeries'/size(m_brainSignalTimeSeries,2);
                    %m_diffusionKernel=m_diffusionKernel/max(
                    %check expression again
                    t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
                    m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
                    m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
                    t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
                    for s_ind=1:s_monteCarloSimulations
                        t_sigma0(:,:,s_ind)=m_sigma0;
                    end
                    [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                        (t_invSpatialDiffusionKernel...
                        ,-t_timeAdjacencyAtDifferentTimes...
                        ,s_maximumTime,s_numberOfVertices,m_sigma0);
                    
                    %% 3. generate true signal
                    
                    m_graphFunction=reshape(m_brainSignalTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
                    
                    m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
                    
                    
                    %%
                    
                    
                    
                    for s_sampleInd=1:size(v_numberOfSamples,2)
                        %% 4. generate observations
                        s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                        sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                        m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        %Same sample locations needed for distributed algo
                        [m_samples(1:s_numberOfSamples,:),...
                            m_positions(1:s_numberOfSamples,:)]...
                            = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                        for s_timeInd=2:s_maximumTime
                            %time t indices
                            v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                (s_timeInd)*s_numberOfVertices;
                            v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                (s_timeInd)*s_numberOfSamples;
                            m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                            for s_mtId=1:s_monteCarloSimulations
                                m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                                    ((s_timeInd-1)*s_numberOfVertices+...
                                    m_positions(v_timetIndicesForSamples,s_mtId));
                            end
                            
                        end
                        
                        %% 5. KF estimate
                        kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                            't_previousMinimumSquaredError',t_initialSigma0,...
                            'm_previousEstimate',m_initialState);
                        for s_muInd=1:size(v_mu,2)
                            v_sigmaForKalman=sqrt((1:s_maximumTime)'*v_numberOfSamples*v_mu(s_muInd))';
                            v_normOFKFErrors=zeros(s_maximumTime,1);
                            v_normOFNotSampled=zeros(s_maximumTime,1);
                            for s_timeInd=1:s_maximumTime
                                %time t indices
                                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                    (s_timeInd)*s_numberOfVertices;
                                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                    (s_timeInd)*s_numberOfSamples;
                                
                                %samples and positions at time t
                                m_samplest=m_samples(v_timetIndicesForSamples,:);
                                m_positionst=m_positions(v_timetIndicesForSamples,:);
                                %estimate
                                
                                [m_prevEstimate,t_newMSE]=...
                                    kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                                    t_transitions(:,:,s_timeInd),...
                                    t_correlations(:,:,s_timeInd),v_sigmaForKalman(s_sampleInd,s_timeInd));
                                t_kfEstimate(s_scalingInd,s_sigmaInd,...
                                    s_muInd,s_sampleInd,v_timetIndicesForSignals,:)=m_prevEstimate;
                                % prepare KF for next iteration
                                kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                                kFOnGFunctionEstimator.m_previousEstimate=m_prevEstimate;
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                for s_mtind=1:s_monteCarloSimulations
                                    v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                                    v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                                        norm(squeeze(t_kfEstimate(s_scalingInd,s_sigmaInd,s_muInd,s_sampleInd,v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                        ,s_mtind))...
                                        -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                    v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                                        norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                end
                                
                                s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                                t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_timeInd, s_sampleInd)...
                                    =sum(v_normOFKFErrors(1:s_timeInd))/...
                                    s_summedNorm;
                                
                                
                                
                                myLegendKF{s_scalingInd,s_sigmaInd,s_muInd, s_sampleInd}...
                                    =strcat('KKF, ',...
                                    sprintf('mu=%g, sigmaDif=%g, scaling=%g,',...
                                    v_mu(s_muInd),...
                                    v_sigmaForDiffusion(s_sigmaInd),v_propagationWeight(s_scalingInd)));...
                                    
                                
                            end
                        end
                        
                    end
                end
            end
            
            
            
            plot(squeeze(t_kfEstimate(1,1,1,1,(0:s_maximumTime-1)*s_numberOfVertices+1,1)))
            m_relativeErrorKf=squeeze(t_relativeErrorKf);
            m_relativeErrorKf=reshape(t_relativeErrorKf,[size(v_propagationWeight,2)*...
                size(v_sigmaForDiffusion,2)*size(v_mu,2)*size(v_numberOfSamples,2),s_maximumTime,]);
            myLegendKF=squeeze(myLegendKF);
            myLegendKF=reshape(myLegendKF,[1,size(v_propagationWeight,2)*...
                size(v_sigmaForDiffusion,2)*size(v_mu,2)*size(v_numberOfSamples,2)]);
            myLegend=[myLegendKF];
            F = F_figure('X',(1:s_maximumTime),'Y',m_relativeErrorKf,...
                'xlab','Time evolution','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
            F.caption=[	sprintf('regularization parameter mu=%g\n',v_mu),...
                sprintf(' diffusion parameter sigma=%g\n',v_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)];
        end
        function F = compute_fig_3209(obj,niter)
            F = obj.load_F_structure(3009);
            F.ylimit=[0.1 0.4];
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Parameter selection'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        %% Real data simulations
        %  Data used:  Kramer, M.A., Kolaczyk, E.D., and Kirsch, H.E. (2008).
        %  Goal: Compare Kalman filter up to time t reconstruction error
        %  measured seperately on the unobserved at each time, summed and normalize.
        %  Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model at each time t and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        
        function F = compute_fig_3012(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=100;
            s_samplePeriod=1;
            s_mu=10^-7;
            s_sigmaForDiffusion=1.4;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.2:0.2:0.2);
            v_bandwidthPercentage=[0.01,0.1];
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=1; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            [m_adjacency,m_timeAdjacency,t_brainSignalTimeSeries] = readBrainSignalTimeEvolvingDataset;
            % take only the first brain signal to test of the pre period..
            % smoother...
            
            
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            m_brainSignalTimeSeries=t_brainSignalTimeSeries(:,:,9);
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_brainSignalTimeSeries,2);
            % data normalization
            v_mean = mean(m_brainSignalTimeSeries,2);
            v_std = std(m_brainSignalTimeSeries')';
            % 			m_brainSignalTimeSeries = diag(1./v_std)*(m_brainSignalTimeSeries...
            %                 - v_mean*ones(1,size(m_brainSignalTimeSeries,2)));
            
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_brainSignalTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_brainSignalTimeSeries=m_brainSignalTimeSeries(s_vertInd,:);
                v_brainSignalTimeSeriesSampledWhole=...
                    v_brainSignalTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_brainSignalTimeSeriesSampled(s_vertInd,:)=...
                    v_brainSignalTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_brainSignalTimeSeriesSampled=m_brainSignalTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_brainSignalTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
            end
            
            
            %% 9. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            v_normOFKFErrors=zeros(s_maximumTime,1);
            v_normOFNotSampled=zeros(s_maximumTime,1);
            v_normOfKrrErrors=zeros(s_maximumTime,1);
            m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
            m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    %v_normOFNotSampled(s_timeInd)=norm(m_samplest,'fro');
                    s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOFKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrrErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                        
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('Bandlimited, ',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                    end
                    myLegendKF{s_sampleInd}=strcat('Kernel Kalman Filter, ',...
                        sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                    myLegendKRR{s_sampleInd}=strcat('KRR time agnostic, ',...
                        sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendBan myLegendKF myLegendKRR];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorbandLimitedEstimate,...
                m_relativeErrorKf, m_relativeErrorKRR]',...
                'xlab','Time evolution','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
            F.ylimit=[0 1];
        end
        function F = compute_fig_3212(obj,niter)
            F = obj.load_F_structure(3012);
            F.ylimit=[0 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Temperature across contiguous USA'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        % find covariance on the train data and
        % use yanninigs for the other data
        % do normalization before sub the mean and devide
        % and then add back
        % if this does not work use covariance
        % use export fig and then
        %% Real data simulations
        %  Data used:brains
        %  Goal: Compare Kalman filter up to time t reconstruction error
        %  measured seperately on the unobserved at each time, summed and normalize.
        %  Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model KRR at each time t and WangWangGuo paper LMS
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_3019(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=250;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=8;
            s_mu=10^-4;
            s_sigmaForDiffusion=1.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.7:0.7:0.7);
            %v_bandwidthPercentage=[0.01,0.1];
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=10^-3; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            [m_adjacency,m_timeAdjacency,t_brainSignalTimeSeries] = readBrainSignalTimeEvolvingDataset;
            % take only the first brain signal to test of the pre period..
            % smoother...
            
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2,4];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            
            % data normalization
            
            %train data
            s_trainingsignal=10;
            s_testsignal=1;
            s_maximumTimeTrain=4000;
            m_brainSignalTimeSeries=t_brainSignalTimeSeries(:,:,s_testsignal);
            s_totalTimeSamples=size(m_brainSignalTimeSeries,2);
            v_mean = mean(t_brainSignalTimeSeries(:,1:s_maximumTimeTrain,s_trainingsignal),2);
            v_std = std(t_brainSignalTimeSeries(:,1:s_maximumTimeTrain,s_trainingsignal)')';
            m_brainSignalTimeSeriesForTrainGraph = diag(1./v_std)*(t_brainSignalTimeSeries(:,1:s_maximumTimeTrain,s_testsignal)...
                - v_mean*ones(1,size(t_brainSignalTimeSeries(:,1:s_maximumTimeTrain,s_testsignal),2)));
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_brainSignalimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_brainSignalTimeSeries=m_brainSignalTimeSeries(s_vertInd,:);
                v_brainSignalTimeSeriesSampledWhole=...
                    v_brainSignalTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_brainSignalimeSeriesSampled(s_vertInd,:)=...
                    v_brainSignalTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_brainSignalimeSeriesSampled=m_brainSignalimeSeriesSampled(:,1:s_maximumTime);
            
            %v_meandata = mean(m_brainSignalimeSeriesSampled,2);
            %v_stddata = std(m_brainSignalimeSeriesSampled')';
            m_brainSignalimeSeriesSampled = diag(1./v_std)*(m_brainSignalimeSeriesSampled...
                - v_mean*ones(1,size(m_brainSignalimeSeriesSampled,2)));
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            
            m_covarianceKernel=cov(m_brainSignalTimeSeriesForTrainGraph');
            %m_covarianceKernel=m_diffusionKernel;
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_covarianceKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_brainSignalimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            m_graphFunctionMean=reshape(v_mean*ones(1,size(m_brainSignalimeSeriesSampled,2)),[s_maximumTime*s_numberOfVertices,1]);
            m_graphFunctionStd=reshape(v_std*ones(1,size(m_brainSignalimeSeriesSampled,2)),[s_maximumTime*s_numberOfVertices,1]);
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            m_graphFunctionMean=repmat(m_graphFunctionMean,1,s_monteCarloSimulations);
            m_graphFunctionStd=repmat(m_graphFunctionStd,1,s_monteCarloSimulations);
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                %                 nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                %                     ('m_kernels',m_covarianceKernel,'s_lambda',s_mu);
                % 				for s_timeInd=1:s_maximumTime
                % 					time t indices
                % 					v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                % 						(s_timeInd)*s_numberOfVertices;
                % 					v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                % 						(s_timeInd)*s_numberOfSamples;
                %
                % 					samples and positions at time t
                % 					m_samplest=m_samples(v_timetIndicesForSamples,:);
                % 					m_positionst=m_positions(v_timetIndicesForSamples,:);
                % 					estimate
                %
                % 					[t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                % 						nonParametricGraphFunctionEstimator.estimate...
                %                         (m_samplest,m_positionst,s_mu);
                %
                % 				end
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                %
                %
                % 				for s_bandInd=1:size(v_bandwidth,2)
                % 					s_bandwidth=v_bandwidth(s_bandInd);
                % 					for s_timeInd=1:s_maximumTime
                % 						time t indices
                % 						v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                % 							(s_timeInd)*s_numberOfVertices;
                % 						v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                % 							(s_timeInd)*s_numberOfSamples;
                %
                % 						samples and positions at time t
                %
                % 						m_samplest=m_samples(v_timetIndicesForSamples,:);
                % 						m_positionst=m_positions(v_timetIndicesForSamples,:);
                % 						create take diagonals from extended graph
                % 						m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                % 						grapht=Graph('m_adjacency',m_adjacency);
                %
                % 						bandlimited estimate
                % 						bandlimitedGraphFunctionEstimator= ...
                % 							BandlimitedGraphFunctionEstimator('m_laplacian'...
                % 							,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                % 						t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                % 							bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                %
                % 					end
                %
                %
                % 				end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
                %% . LMS
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_adjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
                    t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    
                end
            end
            
            
            %% 9. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfLMSErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        % add back the mean only on the denominator in the numerator is canceled anyways..
                        % also no need to divide by std since canceled again
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(...
                            m_graphFunctionMean(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind)+...
                            m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOFKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrrErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfLMSErrors(s_timeInd,s_bandInd)=m_normOfLMSErrors(s_timeInd,s_bandInd)+...
                                norm(t_lmsEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorLms(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' B=%g',s_bandwidth));
                        
                        myLegendLMS{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('LMS',...
                            sprintf(' B=%g',s_bandwidth));
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('BL-IE, ',...
                            sprintf(' B=%g',s_bandwidth));
                    end
                    myLegendKF{s_sampleInd}='KKF ';
                    myLegendKRR{s_sampleInd}='KRR-IE';
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendLMS myLegendKF];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,...
                m_relativeErrorKf]',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            %F.ylimit=[0 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_samplePercentage)];
            
        end
        function F = compute_fig_3219(obj,niter)
            F = obj.load_F_structure(3019);
            %F.ylimit=[0.01 0.03];
            %F.logy = 1;
            F.styles = {'-s','-^','--s','--^','-.d'};
            F.colorset=[0 0 0;0 .7 0;0 0 .9 ;.5 .5 0; 1 0 0 ;.5 0 1;0 .7 .7;.9 0 .9; 1 .5 0;  1 0 .5; 0 1 .5;0 1 0];
            %F.pos=[680 729 509 249];
            
            s_chunk=20;
            s_intSize=size(F.Y,2)-1;
            s_ind=1;
            s_auxind=1;
            auxY(:,1)=F.Y(:,1);
            auxX(:,1)=F.X(:,1);
            while s_ind<s_intSize
                s_ind=s_ind+1;
                if mod(s_ind,s_chunk)==0
                    s_auxind=s_auxind+1;
                    auxY(:,s_auxind)=F.Y(:,s_ind);
                    auxX(:,s_auxind)=F.X(:,s_ind);
                    %s_ind=s_ind-1;
                end
            end
            s_auxind=s_auxind+1;
            auxY(:,s_auxind)=F.Y(:,end);
            auxX(:,s_auxind)=F.X(:,end);
            F.Y=auxY;
            F.X=auxX;
            F.tit='';
            F.ylab='NMSE';
            F.xlab='Time';
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            F.leg_pos_vec = [0.604 0.46 0.059 0.184];
        end
        
        %% Real data simulations
        %  Data used: brains
        %  Goal: Compare perfomance of Kalman filter up to time t as I
        %  change the parameters diffusion parameter reg parameter and
        %  weight of scaled Identity as time evolves s
        % sampling: same each t
        %  Kernel used: Cov kernels
        function F = compute_fig_3021(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=200 ;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            v_mu=10.^[-3,-4,-5,-6]; %opt
            v_sigmaForDiffusion=[0.4]; %opt
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.8:0.8:0.8);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=10.^[-8,-6,-4,-2,0,1,2,3,4]; % weight of edges between the same node opt
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            [m_adjacency,m_timeAdjacency,t_brainSignalTimeSeries] = readBrainSignalTimeEvolvingDataset;
            % take only the first brain signal to test of the pre period..
            % smoother...
            m_brainSignalTimeSeries=t_brainSignalTimeSeries(:,:,9);
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            
            
            
            
            
            %train data
            s_trainingsignal=10;
            s_maximumTimeTrain=4000;
            v_mean = mean(t_brainSignalTimeSeries(:,1:s_maximumTimeTrain,s_trainingsignal),2);
            v_std = std(t_brainSignalTimeSeries(:,1:s_maximumTimeTrain,s_trainingsignal)')';
            m_brainSignalTimeSeriesForTrainGraph = diag(1./v_std)*(t_brainSignalTimeSeries(:,1:s_maximumTimeTrain,s_trainingsignal)...
                - v_mean*ones(1,size(t_brainSignalTimeSeries(:,1:s_maximumTimeTrain,s_trainingsignal),2)));
            
            m_covarianceKernel=cov(m_brainSignalTimeSeriesForTrainGraph');
            %select a subset of measurements
            s_totalTimeSamples=size(m_brainSignalTimeSeries,2);
            %
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_timeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_timeSeries=m_brainSignalTimeSeries(s_vertInd,:);
                v_timeSeriesSampledWhole=...
                    v_timeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_timeSeriesSampled(s_vertInd,:)=...
                    v_timeSeriesSampledWhole(1:s_maximumTime);
            end
            m_timeSeriesSampled=m_timeSeriesSampled(:,1:s_maximumTime);
            
            v_mean = mean(m_timeSeriesSampled,2);
            v_std = std(m_timeSeriesSampled')';
            m_timeSeriesSampled = diag(1./v_std)*(m_timeSeriesSampled...
                - v_mean*ones(1,size(m_timeSeriesSampled,2)));
            
            t_kfEstimate=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,size(v_numberOfSamples,2),s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_relativeErrorKf=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,s_maximumTime,size(v_numberOfSamples,2));
            v_allPositions=(1:s_numberOfVertices)';
            
            for s_scalingInd=1:size(v_propagationWeight,2)
                s_propagationWeight=v_propagationWeight(s_scalingInd);
                m_timeAdjacency=s_propagationWeight*(eye(size(m_adjacency)));
                
                t_timeAdjacencyAtDifferentTimes=...
                    repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
                
                % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
                % 			graphT=graphGenerator.realization;
                %
                %% 2. choise of Kernel must be positive definite
                % diffusion kernel
                graph=Graph('m_adjacency',m_adjacency);
                for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                    % 					diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_sigmaInd),...
                    % 						'm_laplacian',graph.getLaplacian);
                    %m_covarianceKernel=diffusionGraphKernel.generateKernelMatrix;
                    
                    %m_diffusionKernel=m_diffusionKernel/max(
                    %check expression again
                    t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_covarianceKernel,m_timeAdjacency,s_maximumTime);
                    
                    %% generate transition, correlation matrices
                    m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
                    m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
                    t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
                    for s_ind=1:s_monteCarloSimulations
                        t_sigma0(:,:,s_ind)=m_sigma0;
                    end
                    [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                        (t_invSpatialDiffusionKernel...
                        ,-t_timeAdjacencyAtDifferentTimes...
                        ,s_maximumTime,s_numberOfVertices,m_sigma0);
                    
                    %% 3. generate true signal
                    
                    m_graphFunction=reshape(m_timeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
                    
                    m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
                    
                    
                    %%
                    
                    
                    
                    for s_sampleInd=1:size(v_numberOfSamples,2)
                        %% 4. generate observations
                        s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                        sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                        m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        
                        %Same sample locations needed for distributed algo
                        [m_samples(1:s_numberOfSamples,:),...
                            m_positions(1:s_numberOfSamples,:)]...
                            = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                        for s_timeInd=2:s_maximumTime
                            %time t indices
                            v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                (s_timeInd)*s_numberOfVertices;
                            v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                (s_timeInd)*s_numberOfSamples;
                            m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                            for s_mtId=1:s_monteCarloSimulations
                                m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                                    ((s_timeInd-1)*s_numberOfVertices+...
                                    m_positions(v_timetIndicesForSamples,s_mtId));
                            end
                            
                        end
                        
                        
                        %% 5. KF estimate
                        kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                            't_previousMinimumSquaredError',t_initialSigma0,...
                            'm_previousEstimate',m_initialState);
                        for s_muInd=1:size(v_mu,2)
                            v_sigmaForKalman=sqrt((1:s_maximumTime)'*v_numberOfSamples*v_mu(s_muInd))';
                            v_normOFKFErrors=zeros(s_maximumTime,1);
                            v_normOFNotSampled=zeros(s_maximumTime,1);
                            for s_timeInd=1:s_maximumTime
                                %time t indices
                                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                    (s_timeInd)*s_numberOfVertices;
                                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                    (s_timeInd)*s_numberOfSamples;
                                
                                %samples and positions at time t
                                m_samplest=m_samples(v_timetIndicesForSamples,:);
                                m_positionst=m_positions(v_timetIndicesForSamples,:);
                                %estimate
                                
                                [m_prevEstimate,t_newMSE]=...
                                    kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                                    t_transitions(:,:,s_timeInd),...
                                    t_correlations(:,:,s_timeInd),v_sigmaForKalman(s_sampleInd,s_timeInd));
                                t_kfEstimate(s_scalingInd,s_sigmaInd,...
                                    s_muInd,s_sampleInd,v_timetIndicesForSignals,:)=m_prevEstimate;
                                % prepare KF for next iteration
                                kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                                kFOnGFunctionEstimator.m_previousEstimate=m_prevEstimate;
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                for s_mtind=1:s_monteCarloSimulations
                                    v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                                    v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                                        norm(squeeze(t_kfEstimate(s_scalingInd,s_sigmaInd,s_muInd,s_sampleInd,v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                        ,s_mtind))...
                                        -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                    v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                                        norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                end
                                
                                s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                                t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_timeInd, s_sampleInd)...
                                    =sum(v_normOFKFErrors(1:s_timeInd))/...
                                    s_summedNorm;
                                
                                
                                
                                
                                
                                
                                
                            end
                        end
                        
                    end
                end
            end
            
            
            
            %plot(squeeze(t_kfEstimate(1,1,1,1,(0:s_maximumTime-1)*s_numberOfVertices+1,1)))
            for s_scalingInd=1:size(v_propagationWeight,2)
                for s_muInd=1:size(v_mu,2)
                    for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                        m_relativeErrorKf(s_scalingInd,(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd)=...
                            mean(t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,:, 1));
                        myLegend{(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd}=...
                            sprintf('KKF mu=%g, sigma=%g',v_mu(s_muInd),v_sigmaForDiffusion(s_sigmaInd));
                        
                    end
                end
            end
            
            F = F_figure('X',v_propagationWeight,'Y',m_relativeErrorKf',...
                'xlab','diagonal scaling','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
            %F.ylimit=[0 0.5];
            
            F.logx=1;
            F.caption=[	sprintf('regularization parameter mu=%g\n',v_mu),...
                sprintf(' diffusion parameter sigma=%g\n',v_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('sampling size =%g\n',v_samplePercentage)...
                sprintf('time samples =%g\n',s_maximumTime)...
                sprintf('hours per sample =%g\n',s_samplePeriod)...
                'sampling scheme : uniformly random at each t\n'];
            
            
            
        end
        function F = compute_fig_3221(obj,niter)
            F = obj.load_F_structure(3021);
            %F.ylimit=[0.1 0.4];
            %F.xlimit=[10 100];
            F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
                ,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Parameter selection'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        %% Real data simulations
        %  Data used: Economic Sectors
        %  Goal: Compare Kalman filter up to time t Plot reconstruction
        %  error as increase the sampling set
        %  with Bandlimited model at each time t and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        %TODO why error increases
        function F = compute_fig_4015(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=100;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            s_diffusionSigma=6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.1:1);
            %v_bandwidthPercentage=[0.01,0.05];
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=0.01; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            [m_adjacency,m_economicSectorsSignals] = readEconomicSectorSignalTimeEvolvingDataset;
            
            
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=[1:14];           %no point in sampling all.
            %round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2,4];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_economicSectorsSignals,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_economicSectorTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_economicSectorsTimeSeries=m_economicSectorsSignals(s_vertInd,:);
                v_economicSectorsTimeSeriesSampledWhole=...
                    v_economicSectorsTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_economicSectorTimeSeriesSampled(s_vertInd,:)=...
                    v_economicSectorsTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_economicSectorTimeSeriesSampled=m_economicSectorTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_diffusionSigma,'m_laplacian',graph.getNormalizedLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_economicSectorTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_bandwidth,2));
            v_relativeErrorFinalKf=zeros(size(v_numberOfSamples,2),1);
            v_relativeErrorFinalKRR=zeros(size(v_numberOfSamples,2),1);
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_bandwidth,2));
            m_relativeFinalErrorDistr=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            m_relativeFinalErrorBL=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getNormalizedLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                end
                
                %% 7.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                end
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd));
                        m_relativeErrorbandLimitedEstimate(s_timeInd,s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd));
                        
                    end
                    
                    
                end
                
                s_summedNorm=sum(v_normOFNotSampled(1:s_maximumTime));
                v_relativeErrorFinalKf(s_sampleInd)...
                    =sum(v_normOFKFErrors(1:s_maximumTime))/...
                    s_summedNorm;%s_timeInd*s_numberOfVertices;
                v_relativeErrorFinalKRR(s_sampleInd)...
                    =sum(v_normOfKrrErrors(1:s_maximumTime))/...
                    s_summedNorm;%s_timeInd*s_numberOfVertices;
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_relativeFinalErrorDistr(s_sampleInd,s_bandInd)...
                        =m_relativeErrorDistr(s_maximumTime,s_bandInd)/...
                        s_summedNorm;;
                    
                    m_relativeFinalErrorBL(s_sampleInd,s_bandInd)...
                        =m_relativeErrorbandLimitedEstimate(s_maximumTime,s_bandInd)/...
                        s_summedNorm;;
                    myLegendDLSR{s_bandInd}=strcat('DLSR',...
                        sprintf(' W=%g',s_bandwidth ));
                    myLegendBan{s_bandInd}=...
                        strcat('BL-TA, ',...
                        sprintf(' W=%g',s_bandwidth));
                end
                
                
                
                
            end
            
            myLegendKF{1}='KKF';
            myLegendKRR{1}='KRR-TA';
            
            %% 8. measure difference
            
            myLegend=[myLegendDLSR myLegendBan myLegendKF myLegendKRR];
            F = F_figure('X',v_numberOfSamples,'Y',[m_relativeFinalErrorDistr...
                ,m_relativeFinalErrorBL,...
                v_relativeErrorFinalKf,v_relativeErrorFinalKRR]',...
                'xlab','Number economic sectors where value known','ylab','NMSE','tit',...
                sprintf('Gross output by industry'),'leg',myLegend);
            F.ylimit=[0 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_diffusionSigma)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)];
            
        end
        function F = compute_fig_4215(obj,niter)
            F = obj.load_F_structure(4015);
            F.ylimit=[0 2];
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Temperature across contiguous USA'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        %% Real data simulations
        %  Data used: Economic Sectors
        %  Goal: Compare perfomance of Kalman filter up to time t as I
        %  change the parameters diffusion parameter reg parameter and
        %  weight of scaled Identity as time evolves
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_4009(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=100  ;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            v_mu=[10^-7]; %opt
            v_sigmaForDiffusion=[6]; %opt
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.2:0.2:0.2);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=[0,0.001,0.01,0.1,1]; % weight of edges between the same node opt
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            [m_adjacency,m_economicSectorsSignals] = readEconomicSectorSignalTimeEvolvingDataset;
            
            
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_economicSectorsSignals,2);
            %               % data normalization
            % 			v_mean = mean(m_temperatureTimeSeries,2);
            % 			v_std = std(m_temperatureTimeSeries')';
            % 			m_temperatureTimeSeries = diag(1./v_std)*(m_temperatureTimeSeries...
            %                 - v_mean*ones(1,size(m_temperatureTimeSeries,2)));
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_economicSectorTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_economicSectorTimeSeries=m_economicSectorsSignals(s_vertInd,:);
                v_economicSectorTimeSeriesSampledWhole=...
                    v_economicSectorTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_economicSectorTimeSeriesSampled(s_vertInd,:)=...
                    v_economicSectorTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_economicSectorTimeSeriesSampled=m_economicSectorTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            % 			t_spaceAdjacencyAtDifferentTimes=...
            % 				repmat(m_adjacency,[1,1,s_maximumTime]);
            
            t_kfEstimate=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,size(v_numberOfSamples,2),s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_relativeErrorKf=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,s_maximumTime,size(v_numberOfSamples,2));
            v_allPositions=(1:s_numberOfVertices)';
            
            for s_scalingInd=1:size(v_propagationWeight,2)
                s_propagationWeight=v_propagationWeight(s_scalingInd);
                m_timeAdjacency=s_propagationWeight*(eye(size(m_adjacency)));
                
                t_timeAdjacencyAtDifferentTimes=...
                    repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
                
                % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
                % 			graphT=graphGenerator.realization;
                %
                %% 2. choise of Kernel must be positive definite
                % diffusion kernel
                graph=Graph('m_adjacency',m_adjacency);
                for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                    diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_sigmaInd),...
                        'm_laplacian',graph.getNormalizedLaplacian);
                    m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
                    %m_diffusionKernel=m_diffusionKernel/max(
                    %check expression again
                    t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
                    %% generate transition, correlation matrices
                    m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
                    m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
                    t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
                    for s_ind=1:s_monteCarloSimulations
                        t_sigma0(:,:,s_ind)=m_sigma0;
                    end
                    [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                        (t_invSpatialDiffusionKernel...
                        ,-t_timeAdjacencyAtDifferentTimes...
                        ,s_maximumTime,s_numberOfVertices,m_sigma0);
                    
                    %% 3. generate true signal
                    
                    m_graphFunction=reshape(m_economicSectorTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
                    
                    m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
                    
                    
                    %%
                    
                    
                    
                    for s_sampleInd=1:size(v_numberOfSamples,2)
                        %% 4. generate observations
                        s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                        sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                        m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        %Same sample locations needed for distributed algo
                        [m_samples(1:s_numberOfSamples,:),...
                            m_positions(1:s_numberOfSamples,:)]...
                            = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                        for s_timeInd=2:s_maximumTime
                            %time t indices
                            v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                (s_timeInd)*s_numberOfVertices;
                            v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                (s_timeInd)*s_numberOfSamples;
                            m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                            for s_mtId=1:s_monteCarloSimulations
                                m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                                    ((s_timeInd-1)*s_numberOfVertices+...
                                    m_positions(v_timetIndicesForSamples,s_mtId));
                            end
                            
                        end
                        
                        %% 5. KF estimate
                        kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                            't_previousMinimumSquaredError',t_initialSigma0,...
                            'm_previousEstimate',m_initialState);
                        for s_muInd=1:size(v_mu,2)
                            v_sigmaForKalman=sqrt((1:s_maximumTime)'*v_numberOfSamples*v_mu(s_muInd))';
                            v_normOFKFErrors=zeros(s_maximumTime,1);
                            v_normOFNotSampled=zeros(s_maximumTime,1);
                            for s_timeInd=1:s_maximumTime
                                %time t indices
                                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                    (s_timeInd)*s_numberOfVertices;
                                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                    (s_timeInd)*s_numberOfSamples;
                                
                                %samples and positions at time t
                                m_samplest=m_samples(v_timetIndicesForSamples,:);
                                m_positionst=m_positions(v_timetIndicesForSamples,:);
                                %estimate
                                
                                [m_prevEstimate,t_newMSE]=...
                                    kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                                    t_transitions(:,:,s_timeInd),...
                                    t_correlations(:,:,s_timeInd),v_sigmaForKalman(s_sampleInd,s_timeInd));
                                t_kfEstimate(s_scalingInd,s_sigmaInd,...
                                    s_muInd,s_sampleInd,v_timetIndicesForSignals,:)=m_prevEstimate;
                                % prepare KF for next iteration
                                kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                                kFOnGFunctionEstimator.m_previousEstimate=m_prevEstimate;
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                for s_mtind=1:s_monteCarloSimulations
                                    v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                                    v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                                        norm(squeeze(t_kfEstimate(s_scalingInd,s_sigmaInd,s_muInd,s_sampleInd,v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                        ,s_mtind))...
                                        -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                    v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                                        norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                end
                                
                                s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                                t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_timeInd, s_sampleInd)...
                                    =sum(v_normOFKFErrors(1:s_timeInd))/...
                                    s_summedNorm;
                                
                                
                                
                                myLegendKF{s_scalingInd,s_sigmaInd,s_muInd, s_sampleInd}...
                                    =strcat('KKF, ',...
                                    sprintf('mu=%g, sigmaDif=%g, scaling=%g,',...
                                    v_mu(s_muInd),...
                                    v_sigmaForDiffusion(s_sigmaInd),v_propagationWeight(s_scalingInd)));...
                                    
                                
                            end
                        end
                        
                    end
                end
            end
            
            
            
            
            plot(squeeze(t_kfEstimate(1,1,1,1,(0:s_maximumTime-1)*s_numberOfVertices+1,1)))
            m_relativeErrorKf=squeeze(t_relativeErrorKf);
            m_relativeErrorKf=reshape(t_relativeErrorKf,[size(v_propagationWeight,2)*...
                size(v_sigmaForDiffusion,2)*size(v_mu,2)*size(v_numberOfSamples,2),s_maximumTime,]);
            myLegendKF=squeeze(myLegendKF);
            myLegendKF=reshape(myLegendKF,[1,size(v_propagationWeight,2)*...
                size(v_sigmaForDiffusion,2)*size(v_mu,2)*size(v_numberOfSamples,2)]);
            myLegend=[myLegendKF];
            F = F_figure('X',(1:s_maximumTime),'Y',m_relativeErrorKf,...
                'xlab','Time evolution','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
            %F.ylimit=[0 0.5];
        end
        function F = compute_fig_4209(obj,niter)
            F = obj.load_F_structure(4209);
            F.ylimit=[0.1 0.4];
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Parameter selection'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        
        %% Real data simulations
        %  Data used: Economic Sectors
        %  Goal: Compare Kalman filter up to time t reconstruction error
        %  measured seperately on the unobserved at each time, summed and normalize.
        %  Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model at each time t and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_4012(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=50;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-4;
            s_sigmaForDiffusion=15;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.1:0.1);
            v_bandwidthPercentage=[0.01,0.1];
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=20; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            [m_adjacency,m_economicSectorsSignals] = readEconomicSectorSignalTimeEvolvingDataset;
            
            
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2,4];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_economicSectorsSignals,2);
            
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_economicSectorTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_economicSectorTimeSeries=m_economicSectorsSignals(s_vertInd,:);
                v_economicSectorTimeSeriesSampledWhole=...
                    v_economicSectorTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_economicSectorTimeSeriesSampled(s_vertInd,:)=...
                    v_economicSectorTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_economicSectorTimeSeriesSampled=m_economicSectorTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getNormalizedLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_economicSectorTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,graph.getNormalizedLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
            end
            
            
            %% 9. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            v_normOFKFErrors=zeros(s_maximumTime,1);
            v_normOFNotSampled=zeros(s_maximumTime,1);
            v_normOfKrrErrors=zeros(s_maximumTime,1);
            m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
            m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOFKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrrErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                        
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('Bandlimited, ',...
                            sprintf(' W=%g, Sample Size=%g',s_bandwidth,v_numberOfSamples(s_sampleInd)));
                    end
                    myLegendKF{s_sampleInd}=strcat('Kernel Kalman Filter, ',...
                        sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                    myLegendKRR{s_sampleInd}=strcat('KRR time agnostic, ',...
                        sprintf(' Sample Size=%g',v_numberOfSamples(s_sampleInd)));
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendBan myLegendKF myLegendKRR];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorbandLimitedEstimate,...
                m_relativeErrorKf, m_relativeErrorKRR]',...
                'xlab','Time evolution','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
            F.ylimit=[0 1];
        end
        function F = compute_fig_4212(obj,niter)
            F = obj.load_F_structure(4012);
            F.ylimit=[0 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Economic Sectors'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        %% Real data simulations
        %  Data used: Economic Sectors
        %  Goal: Compare Kalman filter up to time t reconstruction error
        %  measured seperately on the unobserved at each time, summed and normalize.
        %  Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model KRR at each time t and LMS and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_4019(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=50;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            s_sigmaForDiffusion=6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.2:0.2:0.2);
            %v_bandwidthPercentage=[0.01,0.1];
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=0.01; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            [m_adjacency,m_economicSectorsSignals] = readEconomicSectorSignalTimeEvolvingDataset;
            
            
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=[2];
            v_bandwidth=[2,4];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_economicSectorsSignals,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_economicSectorTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_economicSectorsTimeSeries=m_economicSectorsSignals(s_vertInd,:);
                v_economicSectorsTimeSeriesSampledWhole=...
                    v_economicSectorsTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_economicSectorTimeSeriesSampled(s_vertInd,:)=...
                    v_economicSectorsTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_economicSectorTimeSeriesSampled=m_economicSectorTimeSeriesSampled(:,1:s_maximumTime);
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getNormalizedLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_economicSectorTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getNormalizedLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
                %% . LMS
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_adjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
                    t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    
                end
            end
            
            
            %% 9. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfLMSErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOFKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrrErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfLMSErrors(s_timeInd,s_bandInd)=m_normOfLMSErrors(s_timeInd,s_bandInd)+...
                                norm(t_lmsEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorLms(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' W=%g',s_bandwidth));
                        
                        myLegendLMS{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('LMS',...
                            sprintf(' W=%g',s_bandwidth))
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('BL-TA, ',...
                            sprintf(' W=%g',s_bandwidth));
                    end
                    myLegendKF{s_sampleInd}='KKF ';
                    myLegendKRR{s_sampleInd}='KRR-TA';
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKF myLegendKRR ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate,...
                m_relativeErrorKf, m_relativeErrorKRR]',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            %F.ylimit=[0 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_numberOfSamples)];
            
        end
        function F = compute_fig_4219(obj,niter)
            F = obj.load_F_structure(4019);
            F.ylimit=[0.5 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Economic Sectors'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        
        %% Real data simulations
        %  Data used:  Economic Sectors
        %  Goal: Compare Kalman filter up to time t Plot reconstruction
        %  error as increase the sampling set
        %  with Bandlimited model at each time t and WangWangGuo paper and
        %  LMS Lorenzo
        %  Kernel used: Diffusion Kernel in space
        %TODO why error increases
        function F = compute_fig_4016(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=360;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            s_diffusionSigma=6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.05:0.9);
            v_bandwidthPercentage=[0.02];
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=0.01; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            [m_adjacency,m_economicSectorsSignals] = readEconomicSectorSignalTimeEvolvingDataset;
            
            
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=[1:14];           %no point in sampling all.
            %round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2,4];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_economicSectorsSignals,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_economicSectorTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_economicSectorsTimeSeries=m_economicSectorsSignals(s_vertInd,:);
                v_economicSectorsTimeSeriesSampledWhole=...
                    v_economicSectorsTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_economicSectorTimeSeriesSampled(s_vertInd,:)=...
                    v_economicSectorsTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_economicSectorTimeSeriesSampled=m_economicSectorTimeSeriesSampled(:,1:s_maximumTime);
            
            
            m_timeAdjacency=v_propagationWeight*(eye(size(m_adjacency))+m_adjacency);
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_diffusionSigma,'m_laplacian',graph.getNormalizedLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            % 			m_invExtendedKernel=KFonGSimulations.createInvExtendedGraphKernel...
            % 				(t_invDiffusionKernel,-t_timeAdjacencyAtDifferentTimes);
            %
            % make kernel great again
            % 			s_beta=0;
            % 			m_invExtendedKernel=m_invExtendedKernel+s_beta*eye(size(m_invExtendedKernel));
            % 			[~,p1]=chol(m_invExtendedKernel);%check if PD
            % 			[~,p2]=chol(m_invExtendedKernel((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
            % 				s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
            % 				*s_numberOfVertices));
            % 			if (p1+p2~=0)
            % 				assert(1==0,'Not Positive Definite Kernel')
            % 			end
            % 			t_eyeTens=repmat(eye(size(m_diffusionKernel)),[1,1,s_maximumTime]);
            % 			t_invDiffusionKernel=t_invDiffusionKernel+s_beta*t_eyeTens;
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_economicSectorTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_bandwidth,2));
            m_relativeErrorLms=zeros(s_maximumTime,size(v_bandwidth,2));
            v_relativeErrorFinalKf=zeros(size(v_numberOfSamples,2),1);
            v_relativeErrorFinalKRR=zeros(size(v_numberOfSamples,2),1);
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_bandwidth,2));
            m_relativeFinalErrorDistr=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            m_relativeFinalErrorLms=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            m_relativeFinalErrorBL=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                v_relativeErrorKRR(s_sampleInd)=norm(t_kRRestimate(:,:,s_sampleInd)...
                    -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                v_relativeErrorKf(s_sampleInd)=norm(t_kfEstimate(:,:,s_sampleInd)...
                    -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getNormalizedLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    m_relativeErrorbandLimitedEstimate(s_sampleInd,s_bandInd)...
                        =norm(t_bandLimitedEstimate(:,:,s_sampleInd,s_bandInd)...
                        -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                    
                    myLegend{s_bandInd}=strcat('Bandlimited  ',...
                        sprintf(' W=%g',s_bandwidth));
                end
                
                %% 7.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_adjacency);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
                %% 8. LMS
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_adjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
                    t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    
                end
                
                
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfLMSErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfLMSErrors(s_timeInd,s_bandInd)=m_normOfLMSErrors(s_timeInd,s_bandInd)+...
                                norm(t_lmsEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd));
                        m_relativeErrorLms(s_timeInd,s_bandInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd));
                        m_relativeErrorbandLimitedEstimate(s_timeInd,s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd));
                        
                    end
                    
                    
                end
                
                s_summedNorm=sum(v_normOFNotSampled(1:s_maximumTime));
                v_relativeErrorFinalKf(s_sampleInd)...
                    =sum(v_normOFKFErrors(1:s_maximumTime))/...
                    s_summedNorm;%s_timeInd*s_numberOfVertices;
                v_relativeErrorFinalKRR(s_sampleInd)...
                    =sum(v_normOfKrrErrors(1:s_maximumTime))/...
                    s_summedNorm;%s_timeInd*s_numberOfVertices;
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_relativeFinalErrorDistr(s_sampleInd,s_bandInd)...
                        =m_relativeErrorDistr(s_maximumTime,s_bandInd)/...
                        s_summedNorm;
                    m_relativeFinalErrorLms(s_sampleInd,s_bandInd)...
                        =m_relativeErrorLms(s_maximumTime,s_bandInd)/...
                        s_summedNorm;
                    
                    m_relativeFinalErrorBL(s_sampleInd,s_bandInd)...
                        =m_relativeErrorbandLimitedEstimate(s_maximumTime,s_bandInd)/...
                        s_summedNorm;
                    myLegendDLSR{s_bandInd}=strcat('DLSR',...
                        sprintf(' W=%g',s_bandwidth ));
                    myLegendLMS{s_bandInd}=strcat('LMS',...
                        sprintf(' W=%g',s_bandwidth ));
                    myLegendBan{s_bandInd}=...
                        strcat('BL-TA, ',...
                        sprintf(' W=%g',s_bandwidth));
                end
                
            end
            
            
            
            
            myLegendKF{1}='KKF';
            myLegendKRR{1}='KRR-TA';
            
            %% 8. measure difference
            
            myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKF myLegendKRR];
            F = F_figure('X',v_numberOfSamples,'Y',[m_relativeFinalErrorDistr...
                ,m_relativeFinalErrorLms,m_relativeFinalErrorBL,...
                v_relativeErrorFinalKf,v_relativeErrorFinalKRR]',...
                'xlab','Measuring Stations','ylab','NMSE','leg',myLegend);
            %F.ylimit=[0 1];
            F.caption=[	  'NMSE vs sampling size Temperature\n',...
                sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_diffusionSigma)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_numberOfSamples)];
        end
        function F = compute_fig_4216(obj,niter)
            F = obj.load_F_structure(4016);
            F.ylimit=[0 1];
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        %% Real data simulations
        %  Data used: Economic Sectors
        %  Goal: Compare perfomance of Kalman filter DLSR LMS Bandlimited and
        %  KRR agnostic up to time t as I
        %  on tracking the signal.
        function F = compute_fig_4018(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=100;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            s_sigmaForDiffusion=6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.05:0.9);
            v_bandwidthPercentage=[0.02];
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=0.01; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            [m_adjacency,m_economicSectorsSignals] = readEconomicSectorSignalTimeEvolvingDataset;
            
            
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=[2];           %no point in sampling all.
            %round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_economicSectorsSignals,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_economicSectorTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_economicSectorsTimeSeries=m_economicSectorsSignals(s_vertInd,:);
                v_economicSectorsTimeSeriesSampledWhole=...
                    v_economicSectorsTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_economicSectorTimeSeriesSampled(s_vertInd,:)=...
                    v_economicSectorsTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_economicSectorTimeSeriesSampled=m_economicSectorTimeSeriesSampled(:,1:s_maximumTime);
            
            
            m_timeAdjacency=v_propagationWeight*(eye(size(m_adjacency))+m_adjacency);
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getNormalizedLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %m_diffusionKernel=graph.getLaplacian();
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_economicSectorTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                    
                end
                myLegendKF{s_sampleInd}='KKF';
                
                %% 6. Kernel Ridge Regression
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                    
                end
                myLegendKRR{s_sampleInd}='KRR-TA';
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                        
                    end
                    
                    
                    myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                        'BL-TA';
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}='DLSR';
                end
                %% 9. LMS
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_adjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
                    t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    myLegendLMS{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}='LMS';
                end
                
            end
            
            for s_vertInd=1:s_numberOfVertices
                
                
                m_meanEstKF(s_vertInd,:)=mean(t_kfEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,1,:),2)';
                m_meanEstKRR(s_vertInd,:)=mean(t_kRRestimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,1,:),2)';
                m_meanEstDLSR(s_vertInd,:)=mean(t_distrEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,:,1,1),2)';
                m_meanEstBan(s_vertInd,:)=mean(t_bandLimitedEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,:,1,1),2)';
                m_meanEstLMS(s_vertInd,:)=mean(t_lmsEstimate((0:s_maximumTime-1)*...
                    s_numberOfVertices+s_vertInd,:,1,1),2)';
                
            end
            %% 9. measure difference
            %normalize errors
            myLegandTrueSignal{1}='True Signal';
            s_vertexToPlot=1;
            myLegend=[myLegandTrueSignal myLegendKF  myLegendKRR  myLegendDLSR myLegendLMS myLegendBan ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_economicSectorTimeSeriesSampled(s_vertexToPlot,:);...
                m_meanEstKF(s_vertexToPlot,:);m_meanEstKRR(s_vertexToPlot,:);m_meanEstDLSR(s_vertexToPlot,:);...
                m_meanEstLMS(s_vertexToPlot,:);m_meanEstBan(s_vertexToPlot,:)],...
                'xlab','Time evolution','ylab','function value','leg',myLegend);
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_numberOfSamples)];
            
        end
        function F = compute_fig_4218(obj,niter)
            F = obj.load_F_structure(2018);
            %F.ylimit=[0 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Temperature tracking'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        %% Real data simulations
        %  Data used:  Economic Extended Sectors
        %  Goal: Compare Kalman filter up to time t Plot reconstruction
        %  error as increase the sampling set
        %  with Bandlimited model at each time t and WangWangGuo paper and
        %  LMS Lorenzo
        %  Kernel used: Diffusion Kernel in space
        %TODO why error increases
        function F = compute_fig_5016(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=360;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            s_diffusionSigma=3;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.1:0.05:0.9);
            v_bandwidthPercentage=[0.02];
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=10^10; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('io_data_adj_sig_1998-2011.mat');
            %m_spatialAdjacency
            
            m_spatialAdjacency(m_spatialAdjacency<0.01)=0;
            % vertices 32,35,37,54,58,59,65 are disconected remove those.
            m_spatialAdjacency([32,35,37,54,58,59,65],:)=[];
            m_spatialAdjacency(:,[32,35,37,54,58,59,65])=[];
            m_economicSectorsSignals([32,35,37,54,58,59,65],:)=[];
            s_numberOfVertices=size(m_spatialAdjacency,1);  % size of the graph
            
            v_numberOfSamples=round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2,4];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_economicSectorsSignals,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_economicSectorTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_economicSectorsTimeSeries=m_economicSectorsSignals(s_vertInd,:);
                v_economicSectorsTimeSeriesSampledWhole=...
                    v_economicSectorsTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_economicSectorTimeSeriesSampled(s_vertInd,:)=...
                    v_economicSectorsTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_economicSectorTimeSeriesSampled=m_economicSectorTimeSeriesSampled(:,1:s_maximumTime);
            
            
            m_timeAdjacency=v_propagationWeight*(eye(size(m_spatialAdjacency))+m_spatialAdjacency);
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_spatialAdjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_spatialAdjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_diffusionSigma,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            % 			m_invExtendedKernel=KFonGSimulations.createInvExtendedGraphKernel...
            % 				(t_invDiffusionKernel,-t_timeAdjacencyAtDifferentTimes);
            %
            % make kernel great again
            % 			s_beta=0;
            % 			m_invExtendedKernel=m_invExtendedKernel+s_beta*eye(size(m_invExtendedKernel));
            % 			[~,p1]=chol(m_invExtendedKernel);%check if PD
            % 			[~,p2]=chol(m_invExtendedKernel((s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime*...
            % 				s_numberOfVertices,(s_maximumTime-1)*s_numberOfVertices+1:s_maximumTime...
            % 				*s_numberOfVertices));
            % 			if (p1+p2~=0)
            % 				assert(1==0,'Not Positive Definite Kernel')
            % 			end
            % 			t_eyeTens=repmat(eye(size(m_diffusionKernel)),[1,1,s_maximumTime]);
            % 			t_invDiffusionKernel=t_invDiffusionKernel+s_beta*t_eyeTens;
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_economicSectorTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_bandwidth,2));
            m_relativeErrorLms=zeros(s_maximumTime,size(v_bandwidth,2));
            v_relativeErrorFinalKf=zeros(size(v_numberOfSamples,2),1);
            v_relativeErrorFinalKRR=zeros(size(v_numberOfSamples,2),1);
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_bandwidth,2));
            m_relativeFinalErrorDistr=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            m_relativeFinalErrorLms=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            m_relativeFinalErrorBL=zeros(size(v_numberOfSamples,2),size(v_bandwidth,2));
            
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                v_relativeErrorKRR(s_sampleInd)=norm(t_kRRestimate(:,:,s_sampleInd)...
                    -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                v_relativeErrorKf(s_sampleInd)=norm(t_kfEstimate(:,:,s_sampleInd)...
                    -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                %% 6. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_spatialAdjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_spatialAdjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    m_relativeErrorbandLimitedEstimate(s_sampleInd,s_bandInd)...
                        =norm(t_bandLimitedEstimate(:,:,s_sampleInd,s_bandInd)...
                        -m_graphFunction,'fro')/norm(m_graphFunction,'fro');
                    
                    myLegend{s_bandInd}=strcat('Bandlimited  ',...
                        sprintf(' W=%g',s_bandwidth));
                end
                
                %% 7.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_spatialAdjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_spatialAdjacency);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
                %% 8. LMS
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_spatialAdjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_spatialAdjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
                    t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    
                end
                
                
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfLMSErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfLMSErrors(s_timeInd,s_bandInd)=m_normOfLMSErrors(s_timeInd,s_bandInd)+...
                                norm(t_lmsEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd));
                        m_relativeErrorLms(s_timeInd,s_bandInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd));
                        m_relativeErrorbandLimitedEstimate(s_timeInd,s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd));
                        
                    end
                    
                    
                end
                
                s_summedNorm=sum(v_normOFNotSampled(1:s_maximumTime));
                v_relativeErrorFinalKf(s_sampleInd)...
                    =sum(v_normOFKFErrors(1:s_maximumTime))/...
                    s_summedNorm;%s_timeInd*s_numberOfVertices;
                v_relativeErrorFinalKRR(s_sampleInd)...
                    =sum(v_normOfKrrErrors(1:s_maximumTime))/...
                    s_summedNorm;%s_timeInd*s_numberOfVertices;
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_relativeFinalErrorDistr(s_sampleInd,s_bandInd)...
                        =m_relativeErrorDistr(s_maximumTime,s_bandInd)/...
                        s_summedNorm;
                    m_relativeFinalErrorLms(s_sampleInd,s_bandInd)...
                        =m_relativeErrorLms(s_maximumTime,s_bandInd)/...
                        s_summedNorm;
                    
                    m_relativeFinalErrorBL(s_sampleInd,s_bandInd)...
                        =m_relativeErrorbandLimitedEstimate(s_maximumTime,s_bandInd)/...
                        s_summedNorm;
                    myLegendDLSR{s_bandInd}=strcat('DLSR',...
                        sprintf(' W=%g',s_bandwidth ));
                    myLegendLMS{s_bandInd}=strcat('LMS',...
                        sprintf(' W=%g',s_bandwidth ));
                    myLegendBan{s_bandInd}=...
                        strcat('BL-TA, ',...
                        sprintf(' W=%g',s_bandwidth));
                end
                
            end
            
            
            
            
            myLegendKF{1}='KKF';
            myLegendKRR{1}='KRR-TA';
            
            %% 8. measure difference
            
            myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKF myLegendKRR];
            F = F_figure('X',v_numberOfSamples,'Y',[m_relativeFinalErrorDistr...
                ,m_relativeFinalErrorLms,m_relativeFinalErrorBL,...
                v_relativeErrorFinalKf,v_relativeErrorFinalKRR]',...
                'xlab','Measuring Stations','ylab','NMSE','leg',myLegend);
            %F.ylimit=[0 1];
            F.caption=[	  'NMSE vs sampling size Temperature\n',...
                sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_diffusionSigma)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_numberOfSamples)];
        end
        function F = compute_fig_5216(obj,niter)
            F = obj.load_F_structure(5016);
            F.ylimit=[0 1];
            %F.xlimit=[10 100];
            %F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
            %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        %% Real data simulations
        %  Data used: Extended Economic Sectors
        %  Goal: Compare Kalman filter up to time t reconstruction error
        %  measured seperately on the unobserved at each time, summed and normalize.
        %  Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model KRR at each time t and LMS and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_5019(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=50;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-4;
            s_sigmaForDiffusion=4.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            %v_bandwidthPercentage=[0.01,0.1];
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=10^3; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('io_data_adj_sig_1998-2011.mat');
            
            
            %m_spatialAdjacency=m_spatialAdjacency/max(max(m_spatialAdjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            %m_spatialAdjacency(m_spatialAdjacency<0.01)=0;
            m_timeAdjacency=v_propagationWeight*eye(size(m_spatialAdjacency));
            
            s_numberOfVertices=size(m_spatialAdjacency,1);  % size of the graph
            
            v_numberOfSamples=round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[3,5,7];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_economicSectorsSignals,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_economicSectorTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_economicSectorsTimeSeries=m_economicSectorsSignals(s_vertInd,:);
                v_economicSectorsTimeSeriesSampledWhole=...
                    v_economicSectorsTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_economicSectorTimeSeriesSampled(s_vertInd,:)=...
                    v_economicSectorsTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_economicSectorTimeSeriesSampled=m_economicSectorTimeSeriesSampled(:,1:s_maximumTime);
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_spatialAdjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_spatialAdjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_economicSectorTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                    ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_spatialAdjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_spatialAdjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
                %% . LMS
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_spatialAdjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_spatialAdjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
                    t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    
                end
            end
            
            
            %% 9. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfLMSErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOFKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrrErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfLMSErrors(s_timeInd,s_bandInd)=m_normOfLMSErrors(s_timeInd,s_bandInd)+...
                                norm(t_lmsEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorLms(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' W=%g',s_bandwidth));
                        
                        myLegendLMS{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('LMS',...
                            sprintf(' W=%g',s_bandwidth))
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('BL-TA, ',...
                            sprintf(' W=%g',s_bandwidth));
                    end
                    myLegendKF{s_sampleInd}='KKF ';
                    myLegendKRR{s_sampleInd}='KRR-TA';
                    
                end
            end
            %normalize errors
            
            %myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKF myLegendKRR ];
            myLegend=[myLegendDLSR myLegendLMS myLegendKF];
            
            % % 			F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
            % % 				,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate,...
            % % 				m_relativeErrorKf, m_relativeErrorKRR]',...
            % % 				'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,...
                m_relativeErrorKf]',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F.ylimit=[0.5 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_numberOfSamples)];
            
        end
        function F = compute_fig_5219(obj,niter)
            F = obj.load_F_structure(5019);
            %F.ylimit=[0.5 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            F.styles = {'-','-.','--o',':','--^','--','--*'};...
                %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            % remove points out of picture
            F.leg{6}=F.leg{7};
            F.leg=F.leg(1:6);
            F.Y(6,:)=[]
            %F.pos=[580 729 509 249];
            F.tit='Economic Sectors';
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            F.leg_pos_vec = [0.233 0.512 0.217 0.202];
        end
        
        %% Real data simulations
        %  Data used: Extended Economic Sectors
        %  Goal: Compare Kalman filter up to time t reconstruction error
        %  measured seperately on the unobserved at each time, summed and normalize.
        %  Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model KRR at each time t and LMS and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_5022(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=50;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-4;
            s_sigmaForDiffusion=3.6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            %v_bandwidthPercentage=[0.01,0.1];
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=10^5; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('io_data_tensoradj_sig_1998-2011.mat');
            %GENERATE NEW FILE FOR THIS SIM
            %ADD new function for extended kernel generation
            %tensor of adj.
            
            %m_spatialAdjacency=m_spatialAdjacency/max(max(m_spatialAdjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            %m_spatialAdjacency(m_spatialAdjacency<0.01)=0;
            m_timeAdjacency=v_propagationWeight*eye(size(m_spatialAdjacency));
            
            s_numberOfVertices=size(m_spatialAdjacency,1);  % size of the graph
            
            v_numberOfSamples=round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[3,5,7];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_economicSectorsSignals,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_economicSectorTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_economicSectorsTimeSeries=m_economicSectorsSignals(s_vertInd,:);
                v_economicSectorsTimeSeriesSampledWhole=...
                    v_economicSectorsTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_economicSectorTimeSeriesSampled(s_vertInd,:)=...
                    v_economicSectorsTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_economicSectorTimeSeriesSampled=m_economicSectorTimeSeriesSampled(:,1:s_maximumTime);
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=t_adjacency;
            %repmat(m_spatialAdjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            for s_timeInd=1:s_maximumTime
                graph=Graph('m_adjacency',t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd));
                
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
                t_invdiffusionKernel(:,:,s_timeInd)=inv(diffusionGraphKernel.generateKernelMatrix);
            end
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelMultDifKer(t_invdiffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_economicSectorTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                        ('m_kernels',inv(t_invdiffusionKernel(:,:,s_timeInd)),'s_lambda',s_mu);
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_spatialAdjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_spatialAdjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                graph=Graph('m_adjacency',m_spatialAdjacency);
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
                %% . LMS
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_spatialAdjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_spatialAdjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
                    t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    
                end
            end
            
            
            %% 9. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfLMSErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOFKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrrErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfLMSErrors(s_timeInd,s_bandInd)=m_normOfLMSErrors(s_timeInd,s_bandInd)+...
                                norm(t_lmsEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorLms(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' W=%g',s_bandwidth));
                        
                        myLegendLMS{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('LMS',...
                            sprintf(' W=%g',s_bandwidth))
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('BL-TA, ',...
                            sprintf(' W=%g',s_bandwidth));
                    end
                    myLegendKF{s_sampleInd}='KKF ';
                    myLegendKRR{s_sampleInd}='KRR-TA';
                    
                end
            end
            %normalize errors
            
            %myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKF myLegendKRR ];
            myLegend=[myLegendDLSR myLegendLMS myLegendKF];
            
            % % 			F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
            % % 				,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate,...
            % % 				m_relativeErrorKf, m_relativeErrorKRR]',...
            % % 				'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,...
                m_relativeErrorKf]',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F.ylimit=[0.5 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_numberOfSamples)];
            
        end
        function F = compute_fig_5222(obj,niter)
            F = obj.load_F_structure(5022);
            %F.ylimit=[0.5 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            F.styles = {'-','-.','--o',':','--^','--','--*'};...
                %	,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            % remove points out of picture
            F.leg{6}=F.leg{7}
            F.leg=F.leg(1:6);
            F.Y(6,:)=[]
            %F.pos=[580 729 509 249];
            F.tit='Economic Sectors';
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            F.leg_pos_vec = [0.233 0.512 0.217 0.202];
        end
        
        %% Real data simulations
        %  Data used: Extended Economic Sectors
        %  Goal: Compare Kalman filter up to time t reconstruction error
        %  measured seperately on the unobserved at each time, summed and normalize.
        %  Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model KRR at each time t and LMS and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_5032(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=50;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-4;
            s_sigmaForDiffusion=4.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.6:0.6:0.6);
            %v_bandwidthPercentage=[0.01,0.1];
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=10^-2; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('io_data_tensoradj_sig_Thresholded1998-2011.mat');
            %GENERATE NEW FILE FOR THIS SIM
            %ADD new function for extended kernel generation
            %tensor of adj.
            
            %m_spatialAdjacency=m_spatialAdjacency/max(max(m_spatialAdjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            %m_spatialAdjacency(m_spatialAdjacency<0.01)=0;
            m_timeAdjacency=v_propagationWeight*eye(size(m_spatialAdjacency));
            
            s_numberOfVertices=size(m_spatialAdjacency,1);  % size of the graph
            
            v_numberOfSamples=round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2,4];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_economicSectorsSignals,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_economicSectorTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_economicSectorsTimeSeries=m_economicSectorsSignals(s_vertInd,:);
                v_economicSectorsTimeSeriesSampledWhole=...
                    v_economicSectorsTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_economicSectorTimeSeriesSampled(s_vertInd,:)=...
                    v_economicSectorsTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_economicSectorTimeSeriesSampled=m_economicSectorTimeSeriesSampled(:,1:s_maximumTime);
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=t_adjacency;
            %repmat(m_spatialAdjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            for s_timeInd=1:s_maximumTime
                graph=Graph('m_adjacency',t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd));
                
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getNormalizedLaplacian);
                t_invdiffusionKernel(:,:,s_timeInd)=inv(diffusionGraphKernel.generateKernelMatrix);
            end
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelMultDifKer(t_invdiffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_economicSectorTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                        ('m_kernels',inv(t_invdiffusionKernel(:,:,s_timeInd)),'s_lambda',s_mu);
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_spatialAdjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_spatialAdjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                graph=Graph('m_adjacency',m_spatialAdjacency);
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
                %% . LMS
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_spatialAdjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_spatialAdjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
                    t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    
                end
            end
            
            
            %% 9. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfLMSErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOFKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrrErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfLMSErrors(s_timeInd,s_bandInd)=m_normOfLMSErrors(s_timeInd,s_bandInd)+...
                                norm(t_lmsEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorLms(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' W=%g',s_bandwidth));
                        
                        myLegendLMS{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('LMS',...
                            sprintf(' W=%g',s_bandwidth))
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('BL-TA, ',...
                            sprintf(' W=%g',s_bandwidth));
                    end
                    myLegendKF{s_sampleInd}='KKF ';
                    myLegendKRR{s_sampleInd}='KRR-TA';
                    
                end
            end
            %normalize errors
            
            %myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKF myLegendKRR ];
            myLegend=[myLegendDLSR myLegendLMS myLegendKF];
            
            % % 			F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
            % % 				,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate,...
            % % 				m_relativeErrorKf, m_relativeErrorKRR]',...
            % % 				'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,...
                m_relativeErrorKf]',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F.ylimit=[0.4 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_numberOfSamples)];
            
        end
        function F = compute_fig_5232(obj,niter)
            F = obj.load_F_structure(5032);
            %F.ylimit=[0.5 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            F.styles = {'-s','-^','--s','--^','-.d'};
            F.X=(1998:2011);
            F.colorset=[0 0 0;0 .7 0;0 0 .9 ;.5 .5 0; 1 0 0 ;.5 0 1;0 .7 .7;.9 0 .9; 1 .5 0;  1 0 .5; 0 1 .5;0 1 0];
            F.leg{1}='DLSR B=2';
            F.leg{2}='DLSR B=4';
            F.leg{3}='LMS B=2';
            F.leg{4}='LMS B=4';
            F.leg_pos_vec = [0.694 0.512 0.134 0.096];
            F.ylab='NMSE';
            F.xlimit=[1998 2011];
            F.xlab='Time [year]';
        end
        function F = compute_fig_5332(obj,niter)
            load('io_data_tensoradj_sig_Thresholded1998-2011.mat');
            v_sum=sum(m_economicSectorsSignals,2);
            [v_sortedSum,v_sortedInd] =sort(v_sum,1,'descend');
            F = F_figure('X',(1998:2011),'Y',[m_economicSectorsSignals(v_sortedInd(1:5),:)],...
                'xlab','Time [year]','ylab','Production [trillion of dollars]');
            %F.styles = {'-*','-^','--*','--^','--.'};
            F.X=(1998:2011);
            
            
            F.xlimit=[1998 2011];
        end
        function F = compute_fig_5432(obj,niter)
            F = obj.load_F_structure(5032);
            %F.ylimit=[0.5 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            F.styles = {'-s','-^','--s','--^','-.d'};
            F.X=(1998:2011);
            F.colorset=[0 0 0;0 .7 0;0 0 .9 ;.5 .5 0; 1 0 0 ;.5 0 1;0 .7 .7;.9 0 .9; 1 .5 0;  1 0 .5; 0 1 .5;0 1 0];
            F.leg{1}='DLSR B=2';
            F.leg{2}='DLSR B=4';
            F.leg{3}='LMS B=2';
            F.leg{4}='LMS B=4';
            F.leg{5}='KRR-DE';
            F.leg_pos_vec = [0.694 0.512 0.134 0.096];
            F.ylab='NMSE';
            F.xlimit=[1998 2011];
            F.xlab='Time [year]';
        end
        
        
        %% Real data simulations
        %  Data used: Extended Economic Sectors
        %  Goal: Compare Kalman filter up to time t reconstruction error
        %  measured seperately on the unobserved at each time, summed and normalize.
        %  Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model KRR at each time t and LMS and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_5033(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=50;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-4;
            s_sigmaForDiffusion=4.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.6:0.6:0.6);
            %v_bandwidthPercentage=[0.01,0.1];
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=[10^-2,10^-2,10^-2,10^-7,10^-2,10^-2,10^-2,10^-2,10^-2,10^-2,10^-2,10^-2,10^-2]; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('io_data_tensoradj_sig_Thresholded1998-2011.mat');
            %GENERATE NEW FILE FOR THIS SIM
            %ADD new function for extended kernel generation
            %tensor of adj.
            
            %m_spatialAdjacency=m_spatialAdjacency/max(max(m_spatialAdjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            %m_spatialAdjacency(m_spatialAdjacency<0.01)=0;
            
            
            s_numberOfVertices=size(m_spatialAdjacency,1);  % size of the graph
            
            v_numberOfSamples=round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2,4];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_economicSectorsSignals,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_economicSectorTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_economicSectorsTimeSeries=m_economicSectorsSignals(s_vertInd,:);
                v_economicSectorsTimeSeriesSampledWhole=...
                    v_economicSectorsTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_economicSectorTimeSeriesSampled(s_vertInd,:)=...
                    v_economicSectorsTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_economicSectorTimeSeriesSampled=m_economicSectorTimeSeriesSampled(:,1:s_maximumTime);
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=t_adjacency;
            %repmat(m_spatialAdjacency,[1,1,s_maximumTime]);
            
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            for s_timeInd=1:s_maximumTime
                graph=Graph('m_adjacency',t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd));
                
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getNormalizedLaplacian);
                t_invdiffusionKernel(:,:,s_timeInd)=inv(diffusionGraphKernel.generateKernelMatrix);
                if (s_timeInd<s_maximumTime)
                    m_timeAdjacency=v_propagationWeight(s_timeInd)*eye(size(m_spatialAdjacency));
                    t_timeAdjacencyAtDifferentTimes(:,:,s_timeInd)=...
                        m_timeAdjacency;
                end
            end
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelMultDifKer(t_invdiffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_economicSectorTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                        ('m_kernels',inv(t_invdiffusionKernel(:,:,s_timeInd)),'s_lambda',s_mu);
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_spatialAdjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_spatialAdjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                graph=Graph('m_adjacency',m_spatialAdjacency);
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
                %% . LMS
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_spatialAdjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_spatialAdjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
                    t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    
                end
            end
            
            
            %% 9. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfLMSErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOFKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrrErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfLMSErrors(s_timeInd,s_bandInd)=m_normOfLMSErrors(s_timeInd,s_bandInd)+...
                                norm(t_lmsEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorLms(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' W=%g',s_bandwidth));
                        
                        myLegendLMS{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('LMS',...
                            sprintf(' W=%g',s_bandwidth))
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('BL-TA, ',...
                            sprintf(' W=%g',s_bandwidth));
                    end
                    myLegendKF{s_sampleInd}='KKF ';
                    myLegendKRR{s_sampleInd}='KRR-TA';
                    
                end
            end
            %normalize errors
            
            %myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKF myLegendKRR ];
            myLegend=[myLegendDLSR myLegendLMS myLegendKF];
            
            % % 			F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
            % % 				,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate,...
            % % 				m_relativeErrorKf, m_relativeErrorKRR]',...
            % % 				'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,...
                m_relativeErrorKf]',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F.ylimit=[0.4 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_numberOfSamples)];
            
        end
        function F = compute_fig_5233(obj,niter)
            F = obj.load_F_structure(5033);
            %F.ylimit=[0.5 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            F.styles = {'-*','-^','--*','--^','--.'};
            F.X=(1998:2011);
            F.colorset=[0 0 0;0 .7 0;0 0 .9 ;.5 .5 0; 1 0 0 ;.5 0 1;0 .7 .7;.9 0 .9; 1 .5 0;  1 0 .5; 0 1 .5;0 1 0];
            F.leg{1}='DLSR B=2';
            F.leg{2}='DLSR B=4';
            F.leg{3}='LMS B=2';
            F.leg{4}='LMS B=4';
            
            
            F.leg_pos_vec = [0.694 0.512 0.134 0.096];
            F.ylab='NMSE';
            F.xlimit=[1998 2011];
            F.xlab='Time [year]';
        end
        function F = compute_fig_5021(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=20 ;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            v_mu=[10^-3,10^-4,10^-5]; %opt
            v_sigmaForDiffusion=[4,4.2,4.4,4.6,4.8]; %opt
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.6:0.6:0.6);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=[0,10^-3,10^-1,10^1,10^3,10^5,10^6]; % weight of edges between the same node opt
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('io_data_adj_sig_1998-2011.mat');
            %m_spatialAdjacency=m_spatialAdjacency/max(max(m_spatialAdjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_spatialAdjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_economicSectorsSignals,2);
            %
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_economicSectorsTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_economicSectorsTimeSeries=m_economicSectorsSignals(s_vertInd,:);
                v_economicSectorsTimeSeriesSampledWhole=...
                    v_economicSectorsTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_economicSectorsTimeSeriesSampled(s_vertInd,:)=...
                    v_economicSectorsTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_economicSectorsTimeSeriesSampled=m_economicSectorsTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            t_kfEstimate=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,size(v_numberOfSamples,2),s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_relativeErrorKf=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,s_maximumTime,size(v_numberOfSamples,2));
            v_allPositions=(1:s_numberOfVertices)';
            
            for s_scalingInd=1:size(v_propagationWeight,2)
                s_propagationWeight=v_propagationWeight(s_scalingInd);
                m_timeAdjacency=s_propagationWeight*(eye(size(m_spatialAdjacency)));
                
                t_timeAdjacencyAtDifferentTimes=...
                    repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
                
                % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
                % 			graphT=graphGenerator.realization;
                %
                %% 2. choise of Kernel must be positive definite
                % diffusion kernel
                graph=Graph('m_adjacency',m_spatialAdjacency);
                for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                    diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_sigmaInd),...
                        'm_laplacian',graph.getLaplacian);
                    m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
                    %m_diffusionKernel=m_diffusionKernel/max(
                    %check expression again
                    t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
                    
                    %% generate transition, correlation matrices
                    m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
                    m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
                    t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
                    for s_ind=1:s_monteCarloSimulations
                        t_sigma0(:,:,s_ind)=m_sigma0;
                    end
                    [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                        (t_invSpatialDiffusionKernel...
                        ,-t_timeAdjacencyAtDifferentTimes...
                        ,s_maximumTime,s_numberOfVertices,m_sigma0);
                    
                    %% 3. generate true signal
                    
                    m_graphFunction=reshape(m_economicSectorsTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
                    
                    m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
                    
                    
                    %%
                    
                    
                    
                    for s_sampleInd=1:size(v_numberOfSamples,2)
                        %% 4. generate observations
                        s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                        sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                        m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        
                        %Same sample locations needed for distributed algo
                        [m_samples(1:s_numberOfSamples,:),...
                            m_positions(1:s_numberOfSamples,:)]...
                            = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                        for s_timeInd=2:s_maximumTime
                            %time t indices
                            v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                (s_timeInd)*s_numberOfVertices;
                            v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                (s_timeInd)*s_numberOfSamples;
                            m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                            for s_mtId=1:s_monteCarloSimulations
                                m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                                    ((s_timeInd-1)*s_numberOfVertices+...
                                    m_positions(v_timetIndicesForSamples,s_mtId));
                            end
                            
                        end
                        
                        %% 5. KF estimate
                        kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                            't_previousMinimumSquaredError',t_initialSigma0,...
                            'm_previousEstimate',m_initialState);
                        for s_muInd=1:size(v_mu,2)
                            v_sigmaForKalman=sqrt((1:s_maximumTime)'*v_numberOfSamples*v_mu(s_muInd))';
                            v_normOFKFErrors=zeros(s_maximumTime,1);
                            v_normOFNotSampled=zeros(s_maximumTime,1);
                            for s_timeInd=1:s_maximumTime
                                %time t indices
                                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                    (s_timeInd)*s_numberOfVertices;
                                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                    (s_timeInd)*s_numberOfSamples;
                                
                                %samples and positions at time t
                                m_samplest=m_samples(v_timetIndicesForSamples,:);
                                m_positionst=m_positions(v_timetIndicesForSamples,:);
                                %estimate
                                
                                [m_prevEstimate,t_newMSE]=...
                                    kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                                    t_transitions(:,:,s_timeInd),...
                                    t_correlations(:,:,s_timeInd),v_sigmaForKalman(s_sampleInd,s_timeInd));
                                t_kfEstimate(s_scalingInd,s_sigmaInd,...
                                    s_muInd,s_sampleInd,v_timetIndicesForSignals,:)=m_prevEstimate;
                                % prepare KF for next iteration
                                kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                                kFOnGFunctionEstimator.m_previousEstimate=m_prevEstimate;
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                for s_mtind=1:s_monteCarloSimulations
                                    v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                                    v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                                        norm(squeeze(t_kfEstimate(s_scalingInd,s_sigmaInd,s_muInd,s_sampleInd,v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                        ,s_mtind))...
                                        -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                    v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                                        norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                end
                                
                                s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                                t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_timeInd, s_sampleInd)...
                                    =sum(v_normOFKFErrors(1:s_timeInd))/...
                                    s_summedNorm;
                                
                                
                                
                                
                                
                                
                                
                            end
                        end
                        
                    end
                end
            end
            
            
            
            plot(squeeze(t_kfEstimate(1,1,1,1,(0:s_maximumTime-1)*s_numberOfVertices+1,1)))
            for s_scalingInd=1:size(v_propagationWeight,2)
                for s_muInd=1:size(v_mu,2)
                    for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                        m_relativeErrorKf(s_scalingInd,(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd)=...
                            mean(t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,:, 1));
                        myLegend{(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd}=...
                            sprintf('KKF mu=%g, sigma=%g',v_mu(s_muInd),v_sigmaForDiffusion(s_sigmaInd));
                        
                    end
                end
            end
            
            F = F_figure('X',v_propagationWeight,'Y',m_relativeErrorKf',...
                'xlab','diagonal scaling','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
            %F.ylimit=[0 0.5];
            
            F.logx=1;
            F.caption=[	sprintf('regularization parameter mu=%g\n',v_mu),...
                sprintf(' diffusion parameter sigma=%g\n',v_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('sampling size =%g\n',v_samplePercentage)];
            
            
            
        end
        function F = compute_fig_5221(obj,niter)
            F = obj.load_F_structure(5021);
            F.ylimit=[0.4 1];
            %F.xlimit=[10 100];
            F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
                ,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Parameter selection'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        function F = compute_fig_5023(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=20 ;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            v_mu=[10^-3,10^-4,10^-5]; %opt
            v_sigmaForDiffusion=[5,6,7]; %opt
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            v_bandwidthPercentage=(0.01:0.02:0.1);
            
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=[10^-4,10^-3,10^-1,10^5,10^6]; % weight of edges between the same node opt
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('io_data_tensoradj_sig_Thresholded1998-2011.mat');
            %m_spatialAdjacency=m_spatialAdjacency/max(max(m_spatialAdjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_spatialAdjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=round(s_numberOfVertices*v_bandwidthPercentage);
            
            %select a subset of measurements
            s_totalTimeSamples=size(m_economicSectorsSignals,2);
            %
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_economicSectorsTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_economicSectorsTimeSeries=m_economicSectorsSignals(s_vertInd,:);
                v_economicSectorsTimeSeriesSampledWhole=...
                    v_economicSectorsTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_economicSectorsTimeSeriesSampled(s_vertInd,:)=...
                    v_economicSectorsTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_economicSectorsTimeSeriesSampled=m_economicSectorsTimeSeriesSampled(:,1:s_maximumTime);
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=t_adjacency;
            
            %
            
            %check
            
            t_kfEstimate=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,size(v_numberOfSamples,2),s_numberOfVertices*s_maximumTime,s_monteCarloSimulations);
            t_relativeErrorKf=zeros(size(v_propagationWeight,2),size(v_sigmaForDiffusion,2),size(v_mu,2)...
                ,s_maximumTime,size(v_numberOfSamples,2));
            v_allPositions=(1:s_numberOfVertices)';
            
            for s_scalingInd=1:size(v_propagationWeight,2)
                s_propagationWeight=v_propagationWeight(s_scalingInd);
                m_timeAdjacency=s_propagationWeight*(eye(size(m_spatialAdjacency)));
                
                t_timeAdjacencyAtDifferentTimes=...
                    repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
                
                % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
                % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
                % 			graphT=graphGenerator.realization;
                %
                %% 2. choise of Kernel must be positive definite
                % diffusion kernel
                
                for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                    
                    %m_diffusionKernel=m_diffusionKernel/max(
                    %check expression again
                    %% 2. choise of Kernel must be positive definite
                    % diffusion kernel
                    for s_timeInd=1:s_maximumTime
                        graph=Graph('m_adjacency',t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd));
                        
                        diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_sigmaInd),'m_laplacian',graph.getNormalizedLaplacian);
                        t_invdiffusionKernel(:,:,s_timeInd)=inv(diffusionGraphKernel.generateKernelMatrix);
                    end
                    t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelMultDifKer(t_invdiffusionKernel,m_timeAdjacency,s_maximumTime);
                    %% generate transition, correlation matrices
                    m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
                    m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
                    t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
                    for s_ind=1:s_monteCarloSimulations
                        t_sigma0(:,:,s_ind)=m_sigma0;
                    end
                    [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                        (t_invSpatialDiffusionKernel...
                        ,-t_timeAdjacencyAtDifferentTimes...
                        ,s_maximumTime,s_numberOfVertices,m_sigma0);
                    
                    %% 3. generate true signal
                    
                    m_graphFunction=reshape(m_economicSectorsTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
                    
                    m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
                    
                    
                    %%
                    
                    
                    
                    for s_sampleInd=1:size(v_numberOfSamples,2)
                        %% 4. generate observations
                        s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                        sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                        m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                        
                        %Same sample locations needed for distributed algo
                        [m_samples(1:s_numberOfSamples,:),...
                            m_positions(1:s_numberOfSamples,:)]...
                            = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                        for s_timeInd=2:s_maximumTime
                            %time t indices
                            v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                (s_timeInd)*s_numberOfVertices;
                            v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                (s_timeInd)*s_numberOfSamples;
                            m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                            for s_mtId=1:s_monteCarloSimulations
                                m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                                    ((s_timeInd-1)*s_numberOfVertices+...
                                    m_positions(v_timetIndicesForSamples,s_mtId));
                            end
                            
                        end
                        
                        %% 5. KF estimate
                        kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                            't_previousMinimumSquaredError',t_initialSigma0,...
                            'm_previousEstimate',m_initialState);
                        for s_muInd=1:size(v_mu,2)
                            v_sigmaForKalman=sqrt((1:s_maximumTime)'*v_numberOfSamples*v_mu(s_muInd))';
                            v_normOFKFErrors=zeros(s_maximumTime,1);
                            v_normOFNotSampled=zeros(s_maximumTime,1);
                            for s_timeInd=1:s_maximumTime
                                %time t indices
                                v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                                    (s_timeInd)*s_numberOfVertices;
                                v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                                    (s_timeInd)*s_numberOfSamples;
                                
                                %samples and positions at time t
                                m_samplest=m_samples(v_timetIndicesForSamples,:);
                                m_positionst=m_positions(v_timetIndicesForSamples,:);
                                %estimate
                                
                                [m_prevEstimate,t_newMSE]=...
                                    kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                                    t_transitions(:,:,s_timeInd),...
                                    t_correlations(:,:,s_timeInd),v_sigmaForKalman(s_sampleInd,s_timeInd));
                                t_kfEstimate(s_scalingInd,s_sigmaInd,...
                                    s_muInd,s_sampleInd,v_timetIndicesForSignals,:)=m_prevEstimate;
                                % prepare KF for next iteration
                                kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                                kFOnGFunctionEstimator.m_previousEstimate=m_prevEstimate;
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                for s_mtind=1:s_monteCarloSimulations
                                    v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                                    v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                                        norm(squeeze(t_kfEstimate(s_scalingInd,s_sigmaInd,s_muInd,s_sampleInd,v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                        ,s_mtind))...
                                        -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                    v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                                        norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                                    
                                end
                                
                                s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                                t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_timeInd, s_sampleInd)...
                                    =sum(v_normOFKFErrors(1:s_timeInd))/...
                                    s_summedNorm;
                                
                                
                                
                                
                                
                                
                                
                            end
                        end
                        
                    end
                end
            end
            
            
            
            %plot(squeeze(t_kfEstimate(1,1,1,1,(0:s_maximumTime-1)*s_numberOfVertices+1,1)))
            for s_scalingInd=1:size(v_propagationWeight,2)
                for s_muInd=1:size(v_mu,2)
                    for s_sigmaInd=1:size(v_sigmaForDiffusion,2)
                        m_relativeErrorKf(s_scalingInd,(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd)=...
                            (t_relativeErrorKf(s_scalingInd,s_sigmaInd,s_muInd,s_maximumTime, 1));
                        myLegend{(s_muInd-1)*size(v_sigmaForDiffusion,2)+s_sigmaInd}=...
                            sprintf('KKF mu=%g, sigma=%g',v_mu(s_muInd),v_sigmaForDiffusion(s_sigmaInd));
                        
                    end
                end
            end
            
            F = F_figure('X',v_propagationWeight,'Y',m_relativeErrorKf',...
                'xlab','diagonal scaling','ylab','NMSE','tit',...
                sprintf('Title'),'leg',myLegend);
            %F.ylimit=[0 0.5];
            
            F.logx=1;
            F.caption=[	sprintf('regularization parameter mu=%g\n',v_mu),...
                sprintf(' diffusion parameter sigma=%g\n',v_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('sampling size =%g\n',v_samplePercentage)];
            
            
            
        end
        function F = compute_fig_5223(obj,niter)
            F = obj.load_F_structure(5023);
            F.ylimit=[0.4 1];
            %F.xlimit=[10 100];
            F.styles = {'-','--','-o','-x','--^','--','--','-','--','-o','-x','--^','--','--'...
                ,'--','-','--','-o','-x','--^','--','--','--','-','--','-o','-x','--^','--','--'};
            %F.pos=[680 729 509 249];
            F.tit='Parameter selection'
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        %% Real data simulations
        %  Data used: Extended Economic Sectors
        %  Goal: Compare Kalman filter up to time t reconstruction error
        %  measured seperately on the unobserved at each time, summed and normalize.
        %  Plot reconstruct error
        %  as time evolves
        %  with Bandlimited model KRR at each time t and LMS and WangWangGuo paper
        %  Kernel used: Diffusion Kernel in space
        function F = compute_fig_5034(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=50;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=2;
            s_mu=10^-4;
            s_sigmaForDiffusion=6.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.3:0.3:0.3);
            %v_bandwidthPercentage=[0.01,0.1];
            s_stepLMS=2;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            
            %% 1. define graph
            tic
            
            v_propagationWeight=10^-2; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('io_data_tensoradj_sig_Thresholded1998-2015.mat');
            %GENERATE NEW FILE FOR THIS SIM
            %ADD new function for extended kernel generation
            %tensor of adj.
            
            %m_spatialAdjacency=m_spatialAdjacency/max(max(m_spatialAdjacency));  % normalize adjacency  so that the weights
            % of m_adjacency  and
            % v_propagationWeight are similar.
            %m_spatialAdjacency(m_spatialAdjacency<0.01)=0;
            m_timeAdjacency=v_propagationWeight*eye(size(m_spatialAdjacency));
            
            s_numberOfVertices=size(m_spatialAdjacency,1);  % size of the graph
            
            v_numberOfSamples=round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2,4,6];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_economicSectorsSignals,2);
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            m_economicSectorTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_economicSectorsTimeSeries=m_economicSectorsSignals(s_vertInd,:);
                v_economicSectorsTimeSeriesSampledWhole=...
                    v_economicSectorsTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_economicSectorTimeSeriesSampled(s_vertInd,:)=...
                    v_economicSectorsTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_economicSectorTimeSeriesSampled=m_economicSectorTimeSeriesSampled(:,1:s_maximumTime);
            t_adjacencySampled=t_adjacency(:,:,1:s_samplePeriod:s_totalTimeSamples);
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=t_adjacencySampled;
            %repmat(m_spatialAdjacency,[1,1,s_maximumTime]);
            t_timeAdjacencyAtDifferentTimes=...
                repmat(m_timeAdjacency,[1,1,s_maximumTime-1]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            for s_timeInd=1:s_maximumTime
                graph=Graph('m_adjacency',t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd));
                
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getNormalizedLaplacian);
                t_invdiffusionKernel(:,:,s_timeInd)=inv(diffusionGraphKernel.generateKernelMatrix);
            end
            %check expression again
            t_invSpatialDiffusionKernel=KFonGSimulations.createinvSpatialKernelMultDifKer(t_invdiffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_economicSectorTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                m_samples=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                m_positions=zeros(s_numberOfSamples*s_maximumTime,s_monteCarloSimulations);
                %Same sample locations needed for distributed algo
                [m_samples(1:s_numberOfSamples,:),...
                    m_positions(1:s_numberOfSamples,:)]...
                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                for s_timeInd=2:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_positions(v_timetIndicesForSamples,:)=m_positions(1:s_numberOfSamples,:);
                    for s_mtId=1:s_monteCarloSimulations
                        m_samples(v_timetIndicesForSamples,s_mtId)=m_graphFunction...
                            ((s_timeInd-1)*s_numberOfVertices+...
                            m_positions(v_timetIndicesForSamples,s_mtId));
                    end
                    
                end
                
                %% 5. KF estimate
                kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
                        kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
                        t_transitions(:,:,s_timeInd),...
                        t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
                    % prepare KF for next iteration
                    kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
                    kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
                        s_sampleInd);
                    
                end
                %% 6. Kernel Ridge Regression
                
                
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
                        ('m_kernels',inv(t_invdiffusionKernel(:,:,s_timeInd)),'s_lambda',s_mu);
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    
                    [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
                        nonParametricGraphFunctionEstimator.estimate...
                        (m_samplest,m_positionst,s_mu);
                    
                end
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    for s_timeInd=1:s_maximumTime
                        %time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %create take diagonals from extended graph
                        m_spatialAdjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_spatialAdjacency);
                        
                        %bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                % method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                % authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                m_spatialAdjacency=mean(t_spaceAdjacencyAtDifferentTimes,3);
                graph=Graph('m_adjacency',m_spatialAdjacency);
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
                %% . LMS
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    m_spatialAdjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                    grapht=Graph('m_adjacency',m_spatialAdjacency);
                    lMSFullTrackingAlgorithmEstimator=...
                        LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
                    t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
                        lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
                    
                    
                end
            end
            
            
            %% 9. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOFKFErrors=zeros(s_maximumTime,1);
                v_normOFNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                m_normOfLMSErrors=zeros(s_maximumTime,size(v_bandwidth,2));
                for s_timeInd=1:s_maximumTime
                    %from the begining up to now
                    v_timetIndicesForSignals=1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %this vector should be added to the positions of the sa
                    
                    
                    for s_mtind=1:s_monteCarloSimulations
                        v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_normOFKFErrors(s_timeInd)=v_normOFKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOFNotSampled(s_timeInd)=v_normOFNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOFNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOFKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrrErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidth,2)
                        
                        for s_mtind=1:s_monteCarloSimulations
                            m_normOfBLErrors(s_timeInd,s_bandInd)=m_normOfBLErrors(s_timeInd,s_bandInd)+...
                                norm(t_bandLimitedEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfDLSRErrors(s_timeInd,s_bandInd)=m_normOfDLSRErrors(s_timeInd,s_bandInd)+...
                                norm(t_distrEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                            m_normOfLMSErrors(s_timeInd,s_bandInd)=m_normOfLMSErrors(s_timeInd,s_bandInd)+...
                                norm(t_lmsEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                                ,s_mtind,s_sampleInd,s_bandInd)...
                                -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        end
                        
                        s_bandwidth=v_bandwidth(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorLms(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' B=%g',s_bandwidth));
                        
                        myLegendLMS{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('LMS',...
                            sprintf(' B=%g',s_bandwidth))
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('BL-IE, ',...
                            sprintf(' B=%g',s_bandwidth));
                    end
                    myLegendKF{s_sampleInd}='KKF ';
                    myLegendKRR{s_sampleInd}='KRR-IE';
                    
                end
            end
            %normalize errors
            
            %myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKF myLegendKRR ];
            myLegend=[myLegendDLSR myLegendLMS myLegendKF];
            
            % % 			F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
            % % 				,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate,...
            % % 				m_relativeErrorKf, m_relativeErrorKRR]',...
            % % 				'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,...
                m_relativeErrorKf]',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            %F.ylimit=[0.3 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_numberOfSamples)];
            
        end
        function F = compute_fig_5234(obj,niter)
            F = obj.load_F_structure(5034);
            F.ylimit=[0.3 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            
            F.leg{2}=F.leg{3};
            F.leg{3}=F.leg{4};
            F.leg{4}=F.leg{6};
            F.leg{5}=F.leg{7};
            F.Y(2,:)=[];
            F.Y(5,:)=[];
            
            F.styles = {'-s','-^','--s','--^','-.d'};
            F.X=(1998:2:2014);
            F.colorset=[0 0 0;0 .7 0;.5 .5 0; .5 0 1;1 0 0 ;.9 0 .9; 1 .5 0;  1 0 .5; 0 1 .5;0 1 0];
            %             F.leg{1}='DLSR B=2';
            %             F.leg{2}='DLSR B=4';
            %             F.leg{3}='LMS B=2';
            %             F.leg{4}='LMS B=4';
            %F.leg_pos_vec = [0.694 0.512 0.134 0.096];
            F.ylab='NMSE';
            F.xlimit=[1998,2014];
            F.xlab='Time [year]';
        end
        function F = compute_fig_5334(obj,niter)
            load('io_data_tensoradj_sig_Thresholded1998-2011.mat');
            v_sum=sum(m_economicSectorsSignals,2);
            [v_sortedSum,v_sortedInd] =sort(v_sum,1,'descend');
            F = F_figure('X',(1998:2011),'Y',[m_economicSectorsSignals(v_sortedInd(1:5),:)],...
                'xlab','Time [year]','ylab','Production [trillion of dollars]');
            %F.styles = {'-*','-^','--*','--^','--.'};
            F.X=(1998:2011);
            
            
            F.xlimit=[1998 2011];
        end
        function F = compute_fig_5434(obj,niter)
            F = obj.load_F_structure(5034);
            %F.ylimit=[0.5 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            F.styles = {'-s','-^','--s','--^','-.d'};
            F.X=(1998:2011);
            F.colorset=[0 0 0;0 .7 0;0 0 .9 ;.5 .5 0; 1 0 0 ;.5 0 1;0 .7 .7;.9 0 .9; 1 .5 0;  1 0 .5; 0 1 .5;0 1 0];
            F.leg{1}='DLSR B=2';
            F.leg{2}='DLSR B=4';
            F.leg{3}='LMS B=2';
            F.leg{4}='LMS B=4';
            F.leg{5}='KRR-DE';
            F.leg_pos_vec = [0.694 0.512 0.134 0.096];
            F.ylab='NMSE';
            F.xlimit=[1998 2011];
            F.xlab='Time [year]';
        end
        
    end
    
    
    
    
    methods(Static)
        
        %% recursion
        
        function [t_correlations,t_transitions]=kernelRegressionRecursion...
                (t_invSpatialKernel,t_invTemporalKernel,s_maximumTime,s_numberOfVertices,m_sigma0)
            %kernelInv should be tridiagonal symmetric of size
            %s_maximumTime*n_numberOfVerticesxs_maximumTime*s_numberOfVertices
            if(s_maximumTime==1);
                t_correlations=inv(t_invSpatialKernel);
                t_transitions=zeros(s_numberOfVertices,s_numberOfVertices,s_maximumTime);
            else
                t_correlations=zeros(s_numberOfVertices,s_numberOfVertices,s_maximumTime);
                t_transitions=zeros(s_numberOfVertices,s_numberOfVertices,s_maximumTime);
                
                %Initialization
                t_correlations(:,:,s_maximumTime)=inv(KFonGSimulations.makepdagain(t_invSpatialKernel(:,:,s_maximumTime)));
                
                %Recursion
                for s_ind=s_maximumTime:-1:2
                    %Define Ct Dt-1 as in Paper
                    m_Ct=t_invTemporalKernel(:,:,s_ind-1);
                    m_Dtminone=t_invSpatialKernel(:,:,s_ind-1);
                    %Recursion for transitions
                    t_transitions(:,:,s_ind)=-t_correlations(:,:,s_ind)*m_Ct;
                    
                    %Recursion for correlations
                    t_correlations(:,:,s_ind-1)=inv(KFonGSimulations.makepdagain(m_Dtminone-m_Ct'*t_correlations(:,:,s_ind)*m_Ct));
                end
                %P1 picked as zeros
                
                t_transitions(:,:,1)=zeros(s_numberOfVertices,s_numberOfVertices); % SOS Choice is arbitary here
                
            end
        end
        function m_mat=makepdagain(m_mat)
            v_eig=eig(m_mat);
            s_minEig=min(v_eig);
            if(s_minEig<=0)
                m_mat=m_mat+(-s_minEig+eps)*eye(size(m_mat));
            end
        end
        function m_invExtendedKernel=createInvExtendedGraphKernel(t_invSpatialKernel,t_invTemporalKernel)
            s_maximumTime=size(t_invSpatialKernel,3);
            s_numberOfVertices=size(t_invTemporalKernel,1);
            m_invExtendedKernel=zeros(s_numberOfVertices*s_maximumTime);
            for s_ind=1:s_maximumTime
                m_invExtendedKernel((s_ind-1)*(s_numberOfVertices)+1:(s_ind)*(s_numberOfVertices),...
                    (s_ind-1)*(s_numberOfVertices)+1:(s_ind)*(s_numberOfVertices))=...
                    t_invSpatialKernel(:,:,s_ind);
                if s_ind<s_maximumTime
                    m_invExtendedKernel((s_ind)*(s_numberOfVertices)+1:(s_ind+1)*(s_numberOfVertices),...
                        (s_ind-1)*(s_numberOfVertices)+1:(s_ind)*(s_numberOfVertices))=...
                        t_invTemporalKernel(:,:,s_ind);
                    m_invExtendedKernel((s_ind-1)*(s_numberOfVertices)+1:(s_ind)*(s_numberOfVertices),...
                        (s_ind)*(s_numberOfVertices)+1:(s_ind+1)*(s_numberOfVertices))=...
                        t_invTemporalKernel(:,:,s_ind)';
                    
                end
            end
        end
        
        function t_invSpatialDiffusionKernel=createinvSpatialKernelMultDifKer(t_invdiffusionKernel,m_timeAdjacency,s_maximumTime)
            t_timeAuxMatrix=...
                repmat( diag(sum(m_timeAdjacency+m_timeAdjacency')),[1,1,s_maximumTime]);
            t_timeAuxMatrix(:,:,1)=t_timeAuxMatrix(:,:,1)-diag(sum(m_timeAdjacency));
            t_timeAuxMatrix(:,:,s_maximumTime)=t_timeAuxMatrix(:,:,s_maximumTime)-diag(sum(m_timeAdjacency'));
            t_invSpatialDiffusionKernel=t_invdiffusionKernel+t_timeAuxMatrix;
            
        end
        function t_invSpatialDiffusionKernel=createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime)
            t_timeAuxMatrix=...
                repmat( diag(sum(m_timeAdjacency+m_timeAdjacency')),[1,1,s_maximumTime]);
            t_timeAuxMatrix(:,:,1)=t_timeAuxMatrix(:,:,1)-diag(sum(m_timeAdjacency));
            t_timeAuxMatrix(:,:,s_maximumTime)=t_timeAuxMatrix(:,:,s_maximumTime)-diag(sum(m_timeAdjacency'));
            t_invSpatialDiffusionKernel=repmat(inv(m_diffusionKernel),[1,1,s_maximumTime]);
            t_invSpatialDiffusionKernel=t_invSpatialDiffusionKernel+t_timeAuxMatrix;
        end
    end
    
end
