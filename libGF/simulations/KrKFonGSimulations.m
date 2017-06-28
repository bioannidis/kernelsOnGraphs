

classdef KrKFonGSimulations < simFunctionSet
    
    properties
        
    end
    
    methods
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare perfomance of simple kr simple kf and combined kr
        % kf
        function F = compute_fig_1001(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=200;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            
            s_sigmaForDiffusion=3;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            
            
            %v_bandwidthPercentage=[0.01,0.1];
            
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %Obs model
            s_obsSigma=0;
            %Kr KF
            s_stateSigma=0.0003;
            s_pctOfTrainPhase=0.1;
            s_transWeight=0.028;
            %Multikernel
            v_sigmaForDiffusion=[1.2,1.3,1.8,1.9];
            s_numberOfKernels=size(v_sigmaForDiffusion,2);
            s_lambdaForMultiKernels=0.001;
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
            load('facebookDataConnectedGraph.mat');
            
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
         
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2,4];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            
            
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            
            
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
            %check expression again
            %t_invSpatialDiffusionKernel=KrKFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            %KKF part
            %             [t_correlations,t_transitions]=KrKFonGSimulations.kernelRegressionRecursion...
            %                 (t_invSpatialDiffusionKernel...
            %                 ,-t_timeAdjacencyAtDifferentTimes...
            %                 ,s_maximumTime,s_numberOfVertices,m_sigma0);
            % Correlation matrices for KrKF
            
            t_spatialDiffusionKernel=KrKFonGSimulations.createDiffusionKernelsFromTopologies(t_spaceAdjacencyAtDifferentTimes,s_sigmaForDiffusion);
            t_spatialCovariance=t_spatialDiffusionKernel;
            % Transition matrix for KrKf
            %m_transitions=s_transWeight*eye(s_numberOfVertices);
            %m_transitions=s_transWeight*(m_adjacency+diag(diag(m_adjacency)));
            m_transitions=randn(s_numberOfVertices);
            m_transitions=m_transitions+m_transitions';
            m_transitions=m_transitions*1/max(eig(m_transitions)+0.00001);
            t_transitionKrKF=...
                repmat(m_transitions,[1,1,s_maximumTime]);
            % Kernels for KrKF
            m_combinedKernel=m_diffusionKernel; % the combined kernel for kriging
            t_dictionaryOfKernels=zeros(s_numberOfKernels,s_numberOfVertices,s_numberOfVertices);
            for s_kernelInd=1:s_numberOfKernels
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_kernelInd),'m_laplacian',graph.getLaplacian);
                t_dictionaryOfKernels(s_kernelInd,:,:)=diffusionGraphKernel.generateKernelMatrix;
            end
            %initialize stateNoise somehow
            
            
            m_stateNoiseCovariance=s_stateSigma*KrKFonGSimulations.generateSPDmatrix(s_numberOfVertices);
            t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_maximumTime]);
            
            %% 3. generate synthetic signal
            v_bandwidthForSignal=5;
            v_stateNoiseMean=zeros(s_numberOfVertices,1);
            v_initialState=zeros(s_numberOfVertices,1);
            
            functionGenerator=BandlimitedCompStateSpaceCompGraphEvolvingFunctionGenerator...
                ('v_bandwidth',v_bandwidthForSignal,...
                't_adjacency',t_spaceAdjacencyAtDifferentTimes,...
                'm_transitions',m_transitions,...
                'm_stateNoiseCovariance',m_stateNoiseCovariance...
                ,'v_stateNoiseMean',v_stateNoiseMean,'v_initialState',v_initialState);
            
            m_graphFunction=functionGenerator.realization(s_monteCarloSimulations);
            
            
            %% 4.0 Estimate signal
            
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_krEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
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
                        [m_samples(v_timetIndicesForSamples,:),...
					m_positions(v_timetIndicesForSamples,:)]=sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
                    end
                    
                end
                
                %% 4.3 Kr estimate
                krigedKFonGFunctionEstimatorkr=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_maximumTime]);
                
                for s_timeInd=1:s_maximumTime
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    [m_estimateKR,~,~]=krigedKFonGFunctionEstimatorkr.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),t_stateNoiseCovariance,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    
                    t_krEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR;
                    
                    
                    
                end
                %% 4.4 KF estimate
                krigedKFonGFunctionEstimatorkf=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_maximumTime]);
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residual=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimatorkf.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),t_stateNoiseCovariance,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    
                    t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKF;
                    
                    
                    krigedKFonGFunctionEstimatorkf.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimatorkf.m_previousEstimate=m_estimateKF;
                    %                     if s_timeInd>1
                    %                         t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                    %                         t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                    %                         s_auxInd=s_auxInd+1;
                    %
                    %                     end
                    %                     if mod(s_timeInd,s_trainTimePeriod)==0
                    %
                    %                         t_stateNoiseCovariance=KrKFonGSimulations.reCalculateStateNoiseCov(t_qAuxForSignal,t_qAuxForMSE,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                    %                         s_auxInd=1;
                    %                     end
                    %
                    %
                    %                     m_estimateKFPrev=m_estimateKF;
                    %                     t_MSEKFPRev=t_MSEKF;
                end
                
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                l2MultiKernelKrigingCovEstimator=L2MultiKernelKrigingCovEstimator...
                    ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma);
                t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_maximumTime]);
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residual=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),t_stateNoiseCovariance,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
                    %                     if s_timeInd>1
                    %                         t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                    %                         t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                    %                         s_auxInd=s_auxInd+1;
                    %                         % save residual matrix
                    % %                         for s_monteCarloSimInd=1:s_monteCarloSimulations
                    % %                         t_residual(:,s_monteCarloSimInd,s_auxInd)=m_samplest(:,s_monteCarloSimInd)...
                    % %                         -m_estimateKF(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd);
                    % %                         end
                    %                     end
                    %                     if mod(s_timeInd,s_trainTimePeriod)==0
                    %                         % recalculate t_stateNoiseCorrelation
                    %                         %t_residualCov=KrKFonGSimulations.calculateResidualCov(t_residual,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                    %                         %normalize Cov?
                    %                         %t_residualCov=t_residualCov/1000;
                    %                         %m_theta=l2MultiKernelKrigingCovEstimator.estimateCoeffVector(t_residualCov,m_positionst);
                    %                         %t_stateNoiseCovariance=KrKFonGSimulations.reCalculateStateNoiseCov(t_qAuxForSignal,t_qAuxForMSE,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                    %                         s_auxInd=1;
                    %                     end
                    %
                    
                    %                     m_estimateKFPrev=m_estimateKF;
                    %                     t_MSEKFPRev=t_MSEKF;
                end
                
                
            end
            
            
            %% 9. measure difference
            
            
            m_relativeErrorKrKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKr=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOfKrKFErrors=zeros(s_maximumTime,1);
                v_normOfKrErrors=zeros(s_maximumTime,1);
                v_normOfKFErrors=zeros(s_maximumTime,1);
                v_normOfNotSampled=zeros(s_maximumTime,1);
                
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
                        %v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_notSampledPositions=v_allPositions;
                        
                        v_normOfKrKFErrors(s_timeInd)=v_normOfKrKFErrors(s_timeInd)+...
                            norm(t_krkfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrErrors(s_timeInd)=v_normOfKrErrors(s_timeInd)+...
                            norm(t_krEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKFErrors(s_timeInd)=v_normOfKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfNotSampled(s_timeInd)=v_normOfNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOfNotSampled(1:s_timeInd));
                    
                    m_relativeErrorKrKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrKFErrors(1:s_timeInd))/...
                        s_summedNorm;
                    m_relativeErrorKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKr(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrErrors(1:s_timeInd))/...
                        s_summedNorm;
                    myLegendKrKF{s_sampleInd}='KKrKF';
                    myLegendKF{s_sampleInd}='KF';
                    myLegendKr{s_sampleInd}='KKr';
                end
            end
            %normalize errors
            myLegend=[myLegendKr,myLegendKF,myLegendKrKF ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorKr,m_relativeErrorKF,m_relativeErrorKrKF]',...
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
        
        
        function F = compute_fig_1201(obj,niter)
            F = obj.load_F_structure(1001);
            F.ylimit=[0 1];
            %F.logy = 1;
            %F.xlimit=[0 100];
            F.styles = {'-.s','-.^','-.o'};
            F.colorset=[0 0 0;0 1 0;1 0 0];
            
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
            F.xlab='Time';
            
            %F.pos=[680 729 509 249];
            F.tit='';
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        %% Real data simulations
        % Data used: College MSg
        %  Goal: Compare perfomance of simple kr simple kf and combined kr
        % kf
        function F = compute_fig_1002(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=90;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            
            s_sigmaForDiffusion=1.5;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            
            
            %v_bandwidthPercentage=[0.01,0.1];
            
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %Obs model
            s_obsSigma=0;
            %Kr KF
            s_stateSigma=0.00016;
            s_pctOfTrainPhase=0.1;
            s_transWeight=0.028;
            %Multikernel
            v_sigmaForDiffusion=[1.2,1.3,1.8,1.9];
            s_numberOfKernels=size(v_sigmaForDiffusion,2);
            s_lambdaForMultiKernels=0.001;
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
            load('symCollegeMsgData.mat');
            
       
         
            
            s_numberOfVertices=size(t_adjReduced,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2,4];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            
            
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            
            
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_1redadj=repmat(t_adjReduced(:,:,1),[1,1,s_maximumTime/3]);
            t_2redadj=repmat(t_adjReduced(:,:,2),[1,1,s_maximumTime/3]);
            t_3redadj=repmat(t_adjReduced(:,:,3),[1,1,s_maximumTime/3]);
            
            t_spaceAdjacencyAtDifferentTimes=[permute(t_1redadj,[3,1,2]);permute(t_2redadj,[3,1,2]);permute(t_3redadj,[3,1,2])];
            t_spaceAdjacencyAtDifferentTimes=permute(t_spaceAdjacencyAtDifferentTimes,[2,3,1]);
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
           
            %check expression again
            %t_invSpatialDiffusionKernel=KrKFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            %KKF part
            %             [t_correlations,t_transitions]=KrKFonGSimulations.kernelRegressionRecursion...
            %                 (t_invSpatialDiffusionKernel...
            %                 ,-t_timeAdjacencyAtDifferentTimes...
            %                 ,s_maximumTime,s_numberOfVertices,m_sigma0);
            % Correlation matrices for KrKF
            
            t_spatialDiffusionKernel=KrKFonGSimulations.createDiffusionKernelsFromTopologies(t_spaceAdjacencyAtDifferentTimes,s_sigmaForDiffusion);
            t_spatialCovariance=t_spatialDiffusionKernel;
            % Transition matrix for KrKf
            %m_transitions=s_transWeight*eye(s_numberOfVertices);
            %m_transitions=s_transWeight*(m_adjacency+diag(diag(m_adjacency)));
            m_transitions=randn(s_numberOfVertices);
            m_transitions=m_transitions+m_transitions';
            m_transitions=m_transitions*1.004/max(eig(m_transitions));
            t_transitionKrKF=...
                repmat(m_transitions,[1,1,s_maximumTime]);
            % Kernels for KrKF
            %m_combinedKernel=m_diffusionKernel; % the combined kernel for kriging
            
            %initialize stateNoise somehow
            
            
            m_stateNoiseCovariance=s_stateSigma*KrKFonGSimulations.generateSPDmatrix(s_numberOfVertices);
            t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_monteCarloSimulations]);
            
            %% 3. generate synthetic signal
            v_bandwidthForSignal=1;
            v_stateNoiseMean=zeros(s_numberOfVertices,1);
            v_initialState=zeros(s_numberOfVertices,1);
            
            functionGenerator=BandlimitedCompStateSpaceCompGraphEvolvingFunctionGenerator...
                ('v_bandwidth',v_bandwidthForSignal,...
                't_adjacency',t_spaceAdjacencyAtDifferentTimes,...
                'm_transitions',m_transitions,...
                'm_stateNoiseCovariance',m_stateNoiseCovariance...
                ,'v_stateNoiseMean',v_stateNoiseMean,'v_initialState',v_initialState);
            
            m_graphFunction=functionGenerator.realization(s_monteCarloSimulations);
            
            
            %% 4.0 Estimate signal
            
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_krEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
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
                        [m_samples(v_timetIndicesForSamples,:),...
					m_positions(v_timetIndicesForSamples,:)]=sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
                    end
                    
                end
                
                %% 4.3 Kr estimate
                krigedKFonGFunctionEstimatorkr=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_monteCarloSimulations]);
                
                for s_timeInd=1:s_maximumTime
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    [m_estimateKR,~,~]=krigedKFonGFunctionEstimatorkr.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),t_stateNoiseCovariance,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    
                    t_krEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR;
                    
                    
                    
                end
                %% 4.4 KF estimate
                krigedKFonGFunctionEstimatorkf=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_monteCarloSimulations]);
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residual=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimatorkf.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),t_stateNoiseCovariance,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    
                    t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKF;
                    
                    
                    krigedKFonGFunctionEstimatorkf.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimatorkf.m_previousEstimate=m_estimateKF;
                    %                     if s_timeInd>1
                    %                         t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                    %                         t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                    %                         s_auxInd=s_auxInd+1;
                    %
                    %                     end
                    %                     if mod(s_timeInd,s_trainTimePeriod)==0
                    %
                    %                         t_stateNoiseCovariance=KrKFonGSimulations.reCalculateStateNoiseCov(t_qAuxForSignal,t_qAuxForMSE,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                    %                         s_auxInd=1;
                    %                     end
                    %
                    %
                    %                     m_estimateKFPrev=m_estimateKF;
                    %                     t_MSEKFPRev=t_MSEKF;
                end
                
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                
                t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_monteCarloSimulations]);
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residual=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),t_stateNoiseCovariance,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
                    %                     if s_timeInd>1
                    %                         t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                    %                         t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                    %                         s_auxInd=s_auxInd+1;
                    %                         % save residual matrix
                    % %                         for s_monteCarloSimInd=1:s_monteCarloSimulations
                    % %                         t_residual(:,s_monteCarloSimInd,s_auxInd)=m_samplest(:,s_monteCarloSimInd)...
                    % %                         -m_estimateKF(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd);
                    % %                         end
                    %                     end
                    %                     if mod(s_timeInd,s_trainTimePeriod)==0
                    %                         % recalculate t_stateNoiseCorrelation
                    %                         %t_residualCov=KrKFonGSimulations.calculateResidualCov(t_residual,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                    %                         %normalize Cov?
                    %                         %t_residualCov=t_residualCov/1000;
                    %                         %m_theta=l2MultiKernelKrigingCovEstimator.estimateCoeffVector(t_residualCov,m_positionst);
                    %                         %t_stateNoiseCovariance=KrKFonGSimulations.reCalculateStateNoiseCov(t_qAuxForSignal,t_qAuxForMSE,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                    %                         s_auxInd=1;
                    %                     end
                    %
                    
                    %                     m_estimateKFPrev=m_estimateKF;
                    %                     t_MSEKFPRev=t_MSEKF;
                end
                
                
            end
            
            
            %% 9. measure difference
            
            
            m_relativeErrorKrKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKr=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOfKrKFErrors=zeros(s_maximumTime,1);
                v_normOfKrErrors=zeros(s_maximumTime,1);
                v_normOfKFErrors=zeros(s_maximumTime,1);
                v_normOfNotSampled=zeros(s_maximumTime,1);
                
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
                        %v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_notSampledPositions=v_allPositions;
                        
                        v_normOfKrKFErrors(s_timeInd)=v_normOfKrKFErrors(s_timeInd)+...
                            norm(t_krkfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrErrors(s_timeInd)=v_normOfKrErrors(s_timeInd)+...
                            norm(t_krEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKFErrors(s_timeInd)=v_normOfKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfNotSampled(s_timeInd)=v_normOfNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOfNotSampled(1:s_timeInd));
                    
                    m_relativeErrorKrKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrKFErrors(1:s_timeInd))/...
                        s_summedNorm;
                    m_relativeErrorKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKr(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrErrors(1:s_timeInd))/...
                        s_summedNorm;
                    myLegendKrKF{s_sampleInd}='KKrKF';
                    myLegendKF{s_sampleInd}='KF';
                    myLegendKr{s_sampleInd}='KKr';
                end
            end
            %normalize errors
            myLegend=[myLegendKr,myLegendKF,myLegendKrKF ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorKr,m_relativeErrorKF,m_relativeErrorKrKF]',...
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
        
        
        
         %% Real data simulations
        % Data used: College MSg
        %  Goal: Compare perfomance of simple krkf and mkkrkf
        function F = compute_fig_1003(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=90;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            
            
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            
            
            %v_bandwidthPercentage=[0.01,0.1];
            
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %Obs model
            s_obsSigma=0;
            %Kr KF
            s_stateSigma=0.00016;
            s_pctOfTrainPhase=0.1;
            s_transWeight=0.028;
            %v_bandwidthPercentage=0.01;
            %v_sigma=ones(s_maximumTime,1)* sqrt((s_maximumTime)*v_numberOfSamples*s_mu)';
            %multi kernel
             s_pctOfTrainPhase=0.1;
            s_trainTime=s_pctOfTrainPhase*s_maximumTime;
            v_sigmaForDiffusion=[0.3,0.2,0.1,0.5,1.3,1.4,1.5,1.6];
            s_numberOfKernels=size(v_sigmaForDiffusion,2);
            s_lambdaForMultiKernels=1;
            
            %s_sigmaForDiffusion=1.5;
            s_sigmaForDiffusion=1.5;%mean(v_sigmaForDiffusion);
            %% 1. define graph
            tic
            
            v_propagationWeight=0.01; % weight of edges between the same node
            % in consecutive time instances
            % extend to vector case
            
            
            %loads [m_adjacency,m_temperatureTimeSeries]
            % the adjacency between the cities and the relevant time
            % series.
            load('symCollegeMsgData.mat');
            
       
         
            
            s_numberOfVertices=size(t_adjReduced,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2,4];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            
            
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            
            
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_1redadj=repmat(t_adjReduced(:,:,1),[1,1,s_maximumTime/3]);
            t_2redadj=repmat(t_adjReduced(:,:,2),[1,1,s_maximumTime/3]);
            t_3redadj=repmat(t_adjReduced(:,:,3),[1,1,s_maximumTime/3]);
            
            t_spaceAdjacencyAtDifferentTimes=[permute(t_1redadj,[3,1,2]);permute(t_2redadj,[3,1,2]);permute(t_3redadj,[3,1,2])];
            t_spaceAdjacencyAtDifferentTimes=permute(t_spaceAdjacencyAtDifferentTimes,[2,3,1]);
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
           
            %check expression again
            %t_invSpatialDiffusionKernel=KrKFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            %KKF part
            %             [t_correlations,t_transitions]=KrKFonGSimulations.kernelRegressionRecursion...
            %                 (t_invSpatialDiffusionKernel...
            %                 ,-t_timeAdjacencyAtDifferentTimes...
            %                 ,s_maximumTime,s_numberOfVertices,m_sigma0);
            % Correlation matrices for KrKF
            
            t_spatialDiffusionKernel=KrKFonGSimulations.createDiffusionKernelsFromTopologies(t_spaceAdjacencyAtDifferentTimes,s_sigmaForDiffusion);
            t_spatialCovariance=t_spatialDiffusionKernel;
            % Transition matrix for KrKf
            %m_transitions=s_transWeight*eye(s_numberOfVertices);
            %m_transitions=s_transWeight*(m_adjacency+diag(diag(m_adjacency)));
            m_transitions=randn(s_numberOfVertices);
            m_transitions=m_transitions+m_transitions';
            m_transitions=m_transitions*1.004/max(eig(m_transitions));
            t_transitionKrKF=...
                repmat(m_transitions,[1,1,s_maximumTime]);
              % Kernels for KrKF
            t_dictionaryOfKernels=zeros(s_numberOfKernels,s_numberOfVertices,s_numberOfVertices);
            v_thetaSpat=ones(s_numberOfKernels,1);
             graph=Graph('m_adjacency',t_spaceAdjacencyAtDifferentTimes(:,:,1));
            m_combinedKernel=zeros(s_numberOfVertices,s_numberOfVertices);
            for s_kernelInd=1:s_numberOfKernels
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_kernelInd),'m_laplacian',graph.getLaplacian);
                m_difker=diffusionGraphKernel.generateKernelMatrix;
                t_dictionaryOfKernels(s_kernelInd,:,:)=m_difker;
                m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));

            end
            % Kernels for KrKF
            %m_combinedKernel=m_diffusionKernel; % the combined kernel for kriging
            
            %initialize stateNoise somehow
            
            
            m_stateNoiseCovariance=s_stateSigma*KrKFonGSimulations.generateSPDmatrix(s_numberOfVertices);
            t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_monteCarloSimulations]);
            m_stateEvolutionKernel=m_stateNoiseCovariance;

            %% 3. generate synthetic signal
            v_bandwidthForSignal=1;
            v_stateNoiseMean=zeros(s_numberOfVertices,1);
            v_initialState=zeros(s_numberOfVertices,1);
            
            functionGenerator=BandlimitedCompStateSpaceCompGraphEvolvingFunctionGenerator...
                ('v_bandwidth',v_bandwidthForSignal,...
                't_adjacency',t_spaceAdjacencyAtDifferentTimes,...
                'm_transitions',m_transitions,...
                'm_stateNoiseCovariance',m_stateNoiseCovariance...
                ,'v_stateNoiseMean',v_stateNoiseMean,'v_initialState',v_initialState);
            
            m_graphFunction=functionGenerator.realization(s_monteCarloSimulations);
            
            
            %% 4.0 Estimate signal
            
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            t_mkrkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
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
                        [m_samples(v_timetIndicesForSamples,:),...
					m_positions(v_timetIndicesForSamples,:)]=sampler.sample(m_graphFunction(v_timetIndicesForSignals,:));
                    end
                    
                end
                
                %% 4.4 MKrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                l2MultiKernelKrigingCovEstimator=L2MultiKernelKrigingCovEstimator...
                    ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=0;
                t_residualSpat=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                t_residualState=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    %m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_spatialCovariance=m_combinedKernel;
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),m_stateEvolutionKernel,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_mkrkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
                    %% Multikernel
                    if s_timeInd>1
                        s_auxInd=s_auxInd+1;
                        t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                        t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                        
                        % save residual matrix
                        for s_monteCarloSimInd=1:s_monteCarloSimulations
                            t_residualSpat(:,s_monteCarloSimInd,s_auxInd)=m_samplest(:,s_monteCarloSimInd)...
                                -m_estimateKF(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd);
                        end
                         m_samplespt=m_samples((s_timeInd-2)*s_numberOfSamples+1:...
                        (s_timeInd-1)*s_numberOfSamples,:);
                        m_positionspt=m_positions((s_timeInd-2)*s_numberOfSamples+1:...
                        (s_timeInd-1)*s_numberOfSamples,:);
                        for s_monteCarloSimInd=1:s_monteCarloSimulations
                            t_residualState(:,s_monteCarloSimInd,s_auxInd)=(m_samplest(:,s_monteCarloSimInd)...
                                -m_estimateKR(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd))...
                            -(m_samplespt(:,s_monteCarloSimInd)...
                                -m_estimateKRPrev(m_positionspt(:,s_monteCarloSimInd),s_monteCarloSimInd));
                        end
                    end
                    if s_timeInd==s_trainTime
                        %calculate exact theta estimate
                        % recalculate t_stateNoiseCorrelation
                        t_residualSpatCov=KrKFonGSimulations.calculateResidualCov(t_residualSpat,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                        m_residualSpatMean=KrKFonGSimulations.calculateResidualMean(t_residualSpat,s_numberOfSamples,s_monteCarloSimulations);
                        t_residualStateCov=KrKFonGSimulations.calculateResidualCov(t_residualState,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                        m_residualStateMean=KrKFonGSimulations.calculateResidualMean(t_residualState,s_numberOfSamples,s_monteCarloSimulations);
                      
                        tic
                        v_thetaSpat=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorCVX(t_residualSpatCov,m_positionst);
%                         v_thetaState=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorCVX(t_residualStateCov,m_positionst);
                        timeCVX=toc
%                         tic
%                         v_thetaSpat=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGD(t_residualSpatCov,m_positionst);
%                         v_thetaState=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGD(t_residualStateCov,m_positionst);
%                         timeGD=toc
                        m_combinedKernel=zeros(s_numberOfVertices);
                        for s_kernelInd=1:s_numberOfKernels
                            m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                        end
%                         m_stateEvolutionKernel=zeros(s_numberOfVertices);
%                         for s_kernelInd=1:s_numberOfKernels
%                             m_stateEvolutionKernel=m_stateEvolutionKernel+v_thetaState(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
%                         end
                        s_auxInd=0;
                        t_residualSpat=zeros(s_numberOfSamples,s_monteCarloSimulations);
%                         t_residualState=zeros(s_numberOfSamples,s_monteCarloSimulations);
                    end
                    if s_timeInd>s_trainTime
                        %do a few gradient descent steps
                        % combine using formula
                        
                        %t_residualSpatCovRankOne=KrKFonGSimulations.calculateResidualCov(t_residualSpat,s_numberOfSamples,s_monteCarloSimulations,1);
                        
                        [t_residualSpatCov,m_residualSpatMean]=KrKFonGSimulations.incrementalCalcResCovMean...
                            (t_residualSpatCov,t_residualSpat,s_timeInd,m_residualSpatMean);
                        
                        %t_residualStateCovRankOne=KrKFonGSimulations.calculateResidualCov(t_residualState,s_numberOfSamples,s_monteCarloSimulations,1);
                        [t_residualStateCov,m_residualStateMean]=KrKFonGSimulations.incrementalCalcResCovMean...
                            (t_residualStateCov,t_residualState,s_timeInd,m_residualStateMean);
                        tic
                        v_thetaSpat=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGDWithInit(t_residualSpatCov,m_positionst,v_thetaSpat);
%                         v_thetaState=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGDWithInit(t_residualStateCov,m_positionst,v_thetaState);
                        timeGD=toc
                        s_timeInd
                        m_combinedKernel=zeros(s_numberOfVertices);
                        for s_kernelInd=1:s_numberOfKernels
                            m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                        end
%                         m_stateEvolutionKernel=zeros(s_numberOfVertices);
%                         for s_kernelInd=1:s_numberOfKernels
%                             m_stateEvolutionKernel=m_stateEvolutionKernel+v_thetaState(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
%                         end
                        s_auxInd=0;
                    end
                    
                    
                    m_estimateKFPrev=m_estimateKF;
                    t_MSEKFPRev=t_MSEKF;
                    m_estimateKRPrev=m_estimateKR;

                end


                
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residual=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),m_stateNoiseCovariance,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
                    %                     if s_timeInd>1
                    %                         t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                    %                         t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                    %                         s_auxInd=s_auxInd+1;
                    %                         % save residual matrix
                    % %                         for s_monteCarloSimInd=1:s_monteCarloSimulations
                    % %                         t_residual(:,s_monteCarloSimInd,s_auxInd)=m_samplest(:,s_monteCarloSimInd)...
                    % %                         -m_estimateKF(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd);
                    % %                         end
                    %                     end
                    %                     if mod(s_timeInd,s_trainTimePeriod)==0
                    %                         % recalculate t_stateNoiseCorrelation
                    %                         %t_residualCov=KrKFonGSimulations.calculateResidualCov(t_residual,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                    %                         %normalize Cov?
                    %                         %t_residualCov=t_residualCov/1000;
                    %                         %m_theta=l2MultiKernelKrigingCovEstimator.estimateCoeffVector(t_residualCov,m_positionst);
                    %                         %t_stateNoiseCovariance=KrKFonGSimulations.reCalculateStateNoiseCov(t_qAuxForSignal,t_qAuxForMSE,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                    %                         s_auxInd=1;
                    %                     end
                    %
                    
                    %                     m_estimateKFPrev=m_estimateKF;
                    %                     t_MSEKFPRev=t_MSEKF;
                end
                
                
            end
            
            
            %% 9. measure difference
            
            
            m_relativeErrorKrKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorMKrKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOfKrKFErrors=zeros(s_maximumTime,1);
                v_normOfMKrKFErrors=zeros(s_maximumTime,1);
                v_normOfNotSampled=zeros(s_maximumTime,1);
                
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
                        %v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_notSampledPositions=v_allPositions;
                        
                        v_normOfKrKFErrors(s_timeInd)=v_normOfKrKFErrors(s_timeInd)+...
                            norm(t_krkfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfMKrKFErrors(s_timeInd)=v_normOfMKrKFErrors(s_timeInd)+...
                            norm(t_mkrkfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                       
                        v_normOfNotSampled(s_timeInd)=v_normOfNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOfNotSampled(1:s_timeInd));
                    
                    m_relativeErrorKrKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrKFErrors(1:s_timeInd))/...
                        s_summedNorm;
                    m_relativeErrorMKrKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfMKrKFErrors(1:s_timeInd))/...
                        s_summedNorm;
                    myLegendKrKF{s_sampleInd}='KKrKF';
                    myLegendMKr{s_sampleInd}='MKKrKF';
                end
            end
            %normalize errors
            myLegend=[myLegendMKr,myLegendKrKF ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorMKrKF,m_relativeErrorKrKF]',...
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
        
              % using MLK with frobenious norm betweeen matrices and l2
        function F = compute_fig_1519(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=1000;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=2;
            s_mu=10^-7;
            
            s_sigmaForDiffusion=0.8;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.6:0.6:0.6);
            
            
            %v_bandwidthPercentage=[0.01,0.1];
            
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %Obs model
            s_obsSigma=0.01;
            %Kr KF
            s_stateSigma=10^-6;
            s_pctOfTrainPhase=0.3;
            s_transWeight=10^-6;
            %Multikernel
            v_sigmaForDiffusion=[2.1,0.7,0.8,1,1.1,1.2,1.3,1.5,1.8,1.9,2,2.2];
            s_numberOfKernels=size(v_sigmaForDiffusion,2);
            s_lambdaForMultiKernels=1;
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
            load('collegeMessagaT10.mat');
            %m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2,4];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_messagesTimeSeries,2);
            % data normalization
            v_mean = mean(m_messagesTimeSeries,2);
            v_std = std(m_messagesTimeSeries')';
            % 			m_temperatureTimeSeries = diag(1./v_std)*(m_temperatureTimeSeries...
            %                 - v_mean*ones(1,size(m_temperatureTimeSeries,2)));
            
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            
            
            m_signalTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_signalTimeSeries=m_messagesTimeSeries(s_vertInd,:);
                v_temperatureTimeSeriesSampledWhole=...
                    v_signalTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_signalTimeSeriesSampled(s_vertInd,:)=...
                    v_temperatureTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_signalTimeSeriesSampled=m_signalTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
            
            % 			graphGenerator = ExtendedGraphGenerator('t_spatialAdjacency',...
            % 				t_spaceAdjacencyAtDifferentTimes,'t_timeAdjacency',t_timeAdjacencyAtDifferentTimes);
            % 			graphT=graphGenerator.realization;
            %
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix; %generateKernelMatrixFromNormLaplacian;
            %check expression again
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            %KKF part
            
            % Correlation matrices for KrKF
            t_spatialCovariance=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            t_obsNoiseCovariace=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            t_spatialDiffusionKernel=KrKFonGSimulations.createDiffusionKernelsFromTopologies(t_spaceAdjacencyAtDifferentTimes,s_sigmaForDiffusion);
            t_spatialCovariance=t_spatialDiffusionKernel;
            % Transition matrix for KrKf
            t_transitionKrKF=...
                repmat(s_transWeight*eye(s_numberOfVertices),[1,1,s_maximumTime]);
            % Kernels for KrKF
            m_combinedKernel=m_diffusionKernel; % the combined kernel for kriging
            t_dictionaryOfKernels=zeros(s_numberOfKernels,s_numberOfVertices,s_numberOfVertices);
            for s_kernelInd=1:s_numberOfKernels
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_kernelInd),'m_laplacian',graph.getLaplacian);
                m_difker=diffusionGraphKernel.generateKernelMatrix;
                t_dictionaryOfKernels(s_kernelInd,:,:)=m_difker;
            end
            %initialize stateNoise somehow
            
            %m_stateEvolutionKernel=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            
            m_stateEvolutionKernel=s_stateSigma^2*eye(s_numberOfVertices);
            %m_stateEvolutionKernel=repmat(m_stateEvolutionKernel,[1,1,s_maximumTime]);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_signalTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %% 4.0 Estimate signal
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));         
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
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
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                l2MultiKernelKrigingCovEstimator=L2MultiKernelKrigingCovEstimator...
                    ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residualSpat=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                t_residualState=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    %m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_spatialCovariance=m_combinedKernel;
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),m_stateEvolutionKernel,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
%                     %% Multikernel
%                     if s_timeInd>1
%                         t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
%                         t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
%                         s_auxInd=s_auxInd+1;
%                         % save residual matrix
%                         for s_monteCarloSimInd=1:s_monteCarloSimulations
%                             t_residualSpat(:,s_monteCarloSimInd,s_auxInd)=m_samplest(:,s_monteCarloSimInd)...
%                                 -m_estimateKF(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd);
%                         end
%                          m_samplespt=m_samples((s_timeInd-2)*s_numberOfSamples+1:...
%                         (s_timeInd-1)*s_numberOfSamples,:);
%                         m_positionspt=m_positions((s_timeInd-2)*s_numberOfSamples+1:...
%                         (s_timeInd-1)*s_numberOfSamples,:);
%                         for s_monteCarloSimInd=1:s_monteCarloSimulations
%                             t_residualState(:,s_monteCarloSimInd,s_auxInd)=(m_samplest(:,s_monteCarloSimInd)...
%                                 -m_estimateKR(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd))...
%                             -(m_samplespt(:,s_monteCarloSimInd)...
%                                 -m_estimateKRPrev(m_positionspt(:,s_monteCarloSimInd),s_monteCarloSimInd));
%                         end
%                     end
%                     if mod(s_timeInd,s_trainTimePeriod)==0
%                         % recalculate t_stateNoiseCorrelation
%                         t_residualSpatCov=KrKFonGSimulations.calculateResidualCov(t_residualSpat,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
%                         
%                         t_residualStateCov=KrKFonGSimulations.calculateResidualCov(t_residualState,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
% 
%                         %normalize Cov?
%                         %v_theta1=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorCVX(t_residualCov,m_positionst);
%                         v_thetaSpat=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGD(t_residualSpatCov,m_positionst);
%                         v_thetaState=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGD(t_residualStateCov,m_positionst);
% 
%                         m_combinedKernel=zeros(s_numberOfVertices);
%                         for s_kernelInd=1:s_numberOfKernels
%                             m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
%                         end
%                         m_stateEvolutionKernel=zeros(s_numberOfVertices);
%                         for s_kernelInd=1:s_numberOfKernels
%                             m_stateEvolutionKernel=m_stateEvolutionKernel+v_thetaState(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
%                         end
%                         s_auxInd=1;
%                     end
%                     
%                     
%                     m_estimateKFPrev=m_estimateKF;
%                     t_MSEKFPRev=t_MSEKF;
%                     m_estimateKRPrev=m_estimateKR;

                end
                %% 5. KF estimate
%                 kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
%                     't_previousMinimumSquaredError',t_initialSigma0,...
%                     'm_previousEstimate',m_initialState);
%                 for s_timeInd=1:s_maximumTime
%                     time t indices
%                     v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
%                         (s_timeInd)*s_numberOfVertices;
%                     v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
%                         (s_timeInd)*s_numberOfSamples;
%                     
%                     samples and positions at time t
%                     m_samplest=m_samples(v_timetIndicesForSamples,:);
%                     m_positionst=m_positions(v_timetIndicesForSamples,:);
%                     estimate
%                     
%                     [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
%                         kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
%                         t_transitions(:,:,s_timeInd),...
%                         t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
%                     prepare KF for next iteration
%                     kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
%                     kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
%                         s_sampleInd);
%                     
%                 end
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
                        %%time t indices
                        v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                            (s_timeInd)*s_numberOfVertices;
                        v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                            (s_timeInd)*s_numberOfSamples;
                        
                        %%samples and positions at time t
                        
                        m_samplest=m_samples(v_timetIndicesForSamples,:);
                        m_positionst=m_positions(v_timetIndicesForSamples,:);
                        %%create take diagonals from extended graph
                        m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
                        grapht=Graph('m_adjacency',m_adjacency);
                        
                        %%bandlimited estimate
                        bandlimitedGraphFunctionEstimator= ...
                            BandlimitedGraphFunctionEstimator('m_laplacian'...
                            ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                %%method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                %%authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidth,2)
                    s_bandwidth=v_bandwidth(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
                % . LMS
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
            m_relativeErrorKrKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOfKFErrors=zeros(s_maximumTime,1);
                v_normOfKrKFErrors=zeros(s_maximumTime,1);
                v_normOfNotSampled=zeros(s_maximumTime,1);
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
                        v_normOfKFErrors(s_timeInd)=v_normOfKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrKFErrors(s_timeInd)=v_normOfKrKFErrors(s_timeInd)+...
                            norm(t_krkfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfNotSampled(s_timeInd)=v_normOfNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOfNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKrKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrKFErrors(1:s_timeInd))/...
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
                    myLegendKF{s_sampleInd}='KKF';
                    myLegendKRR{s_sampleInd}='KRR-IE';
                    myLegendKrKF{s_sampleInd}='KrKKF';
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKRR myLegendKF myLegendKrKF ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate...
                , m_relativeErrorKRR,m_relativeErrorKf,m_relativeErrorKrKF]',...
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
     
        
        function F = compute_fig_1202(obj,niter)
            F = obj.load_F_structure(1002);
            F.ylimit=[0 1];
            %F.logy = 1;
            F.xlimit=[0 90];
            F.styles = {'-.s','-.^','-.o'};
            F.colorset=[0 0 0;0 1 0;1 0 0];
            
            s_chunk=5;
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
            F.xlab='Time[day]';
            
            %F.pos=[680 729 509 249];
            F.tit='';
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        % using MLK with frobenious norm betweeen matrices and l2
        function F = compute_fig_1119(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=200;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            
            s_sigmaForDiffusion=1.9;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.3:0.3:0.3);
            
            
            %v_bandwidthPercentage=[0.01,0.1];
            
            s_stepLMS=0.7;
            s_muDLSR=1.4;
            s_betaDLSR=0.5;
            %Obs model
             s_obsSigma=0.001;
            %Kr KF
            s_stateSigma=0.00005;
            s_pctOfTrainPhase=0.1;
            s_transWeight=0.028;
            %Multikernel
            v_sigmaForDiffusion=[0.7,0.8,1,1.1,1.2,1.3,1.5,1.8,1.9,2,2.1,2.2];
            s_numberOfKernels=size(v_sigmaForDiffusion,2);
            s_lambdaForMultiKernels=1;
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
            load('facebookDataConnectedGraph.mat');
            
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
          
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[20];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            
           
            
            
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            
            
            
            
            % define adjacency in the space and in the time at each time
            % between locations
            t_spaceAdjacencyAtDifferentTimes=...
                repmat(m_adjacency,[1,1,s_maximumTime]);
          
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
           
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            %KKF part
            
            % Correlation matrices for KrKF
            % Transition matrix for KrKf
           m_transitions=s_transWeight*(m_adjacency+diag(diag(m_adjacency)));
            t_transitionKrKF=...
                repmat(m_transitions,[1,1,s_maximumTime]);
            % Kernels for KrKF
            m_combinedKernel=m_diffusionKernel; % the combined kernel for kriging
            t_dictionaryOfKernels=zeros(s_numberOfKernels,s_numberOfVertices,s_numberOfVertices);
            for s_kernelInd=1:s_numberOfKernels
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_kernelInd),'m_laplacian',graph.getLaplacian);
                m_difker=diffusionGraphKernel.generateKernelMatrix;
                t_dictionaryOfKernels(s_kernelInd,:,:)=m_difker;
            end
            %initialize stateNoise somehow
            
            
            m_stateNoiseCovariance=s_stateSigma*KrKFonGSimulations.generateSPDmatrix(s_numberOfVertices);
            t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_maximumTime]);
            
            %% 3. generate true signal
            
             v_bandwidthForSignal=20;
            v_stateNoiseMean=zeros(s_numberOfVertices,1);
            v_initialState=zeros(s_numberOfVertices,1);
            
            functionGenerator=BandlimitedCompStateSpaceCompGraphEvolvingFunctionGenerator...
                ('v_bandwidth',v_bandwidthForSignal,...
                't_adjacency',t_spaceAdjacencyAtDifferentTimes,...
                'm_transitions',m_transitions,...
                'm_stateNoiseCovariance',m_stateNoiseCovariance...
                ,'v_stateNoiseMean',v_stateNoiseMean,'v_initialState',v_initialState);
            
            m_graphFunction=functionGenerator.realization(s_monteCarloSimulations);
            
            %% 4.0 Estimate signal
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
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
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
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
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                l2MultiKernelKrigingCovEstimator=L2MultiKernelKrigingCovEstimator...
                    ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residual=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    %m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_spatialCovariance=m_combinedKernel;
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),t_stateNoiseCovariance,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
                   
                    
                    
                    m_estimateKFPrev=m_estimateKF;
                    t_MSEKFPRev=t_MSEKF;
                end
                %% 5. KF estimate
                %% 6. Kernel Ridge Regression
                
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
            m_relativeErrorKrKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOfKFErrors=zeros(s_maximumTime,1);
                v_normOfKrKFErrors=zeros(s_maximumTime,1);
                v_normOfNotSampled=zeros(s_maximumTime,1);
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
                        %v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_notSampledPositions=v_allPositions;
                        v_normOfKFErrors(s_timeInd)=v_normOfKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrKFErrors(s_timeInd)=v_normOfKrKFErrors(s_timeInd)+...
                            norm(t_krkfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfNotSampled(s_timeInd)=v_normOfNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOfNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKrKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrKFErrors(1:s_timeInd))/...
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
                    myLegendKF{s_sampleInd}='KKF';
                    myLegendKRR{s_sampleInd}='KRR-IE';
                    myLegendKrKF{s_sampleInd}='KrKKF';
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendLMS myLegendBan  myLegendKrKF ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate...
                , m_relativeErrorKrKF]',...
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
        
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare perfomance of simple kr simple kf and combined kr
        % kf
        function F = compute_fig_2001(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=200;
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
            
            
            %v_bandwidthPercentage=[0.01,0.1];
            
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %Obs model
            s_obsSigma=0.01;
            %Kr KF
            s_stateSigma=0.5;
            s_pctOfTrainPhase=0.1;
            s_transWeight=0.8;
            %Multikernel
            v_sigmaForDiffusion=[1.2,1.3,1.8,1.9];
            s_numberOfKernels=size(v_sigmaForDiffusion,2);
            s_lambdaForMultiKernels=0.001;
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
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            
            
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
            t_invSpatialDiffusionKernel=KrKFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            %KKF part
            [t_correlations,t_transitions]=KrKFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            % Correlation matrices for KrKF
            t_spatialCovariance=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            t_obsNoiseCovariace=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            t_spatialDiffusionKernel=KrKFonGSimulations.createDiffusionKernelsFromTopologies(t_spaceAdjacencyAtDifferentTimes,s_sigmaForDiffusion);
            t_spatialCovariance=t_spatialDiffusionKernel;
            % Transition matrix for KrKf
            t_transitionKrKF=...
                repmat(s_transWeight*eye(s_numberOfVertices),[1,1,s_maximumTime]);
            % Kernels for KrKF
            m_combinedKernel=m_diffusionKernel; % the combined kernel for kriging
            t_dictionaryOfKernels=zeros(s_numberOfKernels,s_numberOfVertices,s_numberOfVertices);
            for s_kernelInd=1:s_numberOfKernels
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_kernelInd),'m_laplacian',graph.getLaplacian);
                t_dictionaryOfKernels(s_kernelInd,:,:)=diffusionGraphKernel.generateKernelMatrix;
            end
            %initialize stateNoise somehow
            
            
            m_stateNoiseCovariance=s_stateSigma^2*eye(s_numberOfVertices);
            t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_maximumTime]);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %% 4.0 Estimate signal
            
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_krEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
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
                
                %% 4.3 Kr estimate
                krigedKFonGFunctionEstimatorkr=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_maximumTime]);
                
                for s_timeInd=1:s_maximumTime
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    [m_estimateKR,~,~]=krigedKFonGFunctionEstimatorkr.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),t_stateNoiseCovariance,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    
                    t_krEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR;
                    
                    
                    
                end
                %% 4.4 KF estimate
                krigedKFonGFunctionEstimatorkf=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_maximumTime]);
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residual=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimatorkf.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),t_stateNoiseCovariance,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    
                    t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKF;
                    
                    
                    krigedKFonGFunctionEstimatorkf.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimatorkf.m_previousEstimate=m_estimateKF;
                    if s_timeInd>1
                        t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                        t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                        s_auxInd=s_auxInd+1;
                        
                    end
                    if mod(s_timeInd,s_trainTimePeriod)==0
                        
                        %                         t_stateNoiseCovariance=KrKFonGSimulations.reCalculateStateNoiseCov(t_qAuxForSignal,t_qAuxForMSE,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                        s_auxInd=1;
                    end
                    
                    
                    m_estimateKFPrev=m_estimateKF;
                    t_MSEKFPRev=t_MSEKF;
                end
                
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                l2MultiKernelKrigingCovEstimator=L2MultiKernelKrigingCovEstimator...
                    ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma);
                t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_maximumTime]);
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residual=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),t_stateNoiseCovariance,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
                    if s_timeInd>1
                        t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                        t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                        s_auxInd=s_auxInd+1;
                        % save residual matrix
                        %                         for s_monteCarloSimInd=1:s_monteCarloSimulations
                        %                         t_residual(:,s_monteCarloSimInd,s_auxInd)=m_samplest(:,s_monteCarloSimInd)...
                        %                         -m_estimateKF(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd);
                        %                         end
                    end
                    if mod(s_timeInd,s_trainTimePeriod)==0
                        % recalculate t_stateNoiseCorrelation
                        %t_residualCov=KrKFonGSimulations.calculateResidualCov(t_residual,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                        %normalize Cov?
                        %t_residualCov=t_residualCov/1000;
                        %m_theta=l2MultiKernelKrigingCovEstimator.estimateCoeffVector(t_residualCov,m_positionst);
                        %t_stateNoiseCovariance=KrKFonGSimulations.reCalculateStateNoiseCov(t_qAuxForSignal,t_qAuxForMSE,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                        s_auxInd=1;
                    end
                    
                    
                    m_estimateKFPrev=m_estimateKF;
                    t_MSEKFPRev=t_MSEKF;
                end
                
                
            end
            
            
            %% 9. measure difference
            
            
            m_relativeErrorKrKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKr=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOfKrKFErrors=zeros(s_maximumTime,1);
                v_normOfKrErrors=zeros(s_maximumTime,1);
                v_normOfKFErrors=zeros(s_maximumTime,1);
                v_normOfNotSampled=zeros(s_maximumTime,1);
                
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
                        %v_notSampledPositions=setdiff(v_allPositions,m_positionst(:,s_mtind));
                        v_notSampledPositions=v_allPositions;
                        
                        v_normOfKrKFErrors(s_timeInd)=v_normOfKrKFErrors(s_timeInd)+...
                            norm(t_krkfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrErrors(s_timeInd)=v_normOfKrErrors(s_timeInd)+...
                            norm(t_krEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKFErrors(s_timeInd)=v_normOfKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfNotSampled(s_timeInd)=v_normOfNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOfNotSampled(1:s_timeInd));
                    
                    m_relativeErrorKrKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrKFErrors(1:s_timeInd))/...
                        s_summedNorm;
                    m_relativeErrorKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKr(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrErrors(1:s_timeInd))/...
                        s_summedNorm;
                    myLegendKrKF{s_sampleInd}='KKrKF';
                    myLegendKF{s_sampleInd}='KF';
                    myLegendKr{s_sampleInd}='KKr';
                end
            end
            %normalize errors
            myLegend=[myLegendKr,myLegendKF,myLegendKrKF ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorKr,m_relativeErrorKF,m_relativeErrorKrKF]',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            
            F.ylimit=[0 2];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_samplePercentage)];
            
            
        end
        
        
        %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: Compare perfomance of KRKF Kalman filter DLSR LMS Bandlimited and
        %  KRR agnostic up to time t as I
        %  on tracking the signal.
        function F = compute_fig_2018(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=200;
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
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %KrKKF
            s_obsSigma=0.01;
            s_stateSigma=0.00001;
            s_pctOfTrainPhase=0.1;
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
            v_bandwidth=5;
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
            
            %KrKKF
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            
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
            t_invSpatialDiffusionKernel=KrKFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            [t_correlations,t_transitions]=KrKFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            %KrKF part
            [t_correlations,t_transitions]=KrKFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            % Correlation matrices for KrKF
            
            t_spatialDiffusionKernel=KrKFonGSimulations.createDiffusionKernelsFromTopologies(t_spaceAdjacencyAtDifferentTimes,s_sigmaForDiffusion);
            t_spatialCovariance=t_spatialDiffusionKernel;
            
            %initialize stateNoise somehow
            
            t_stateNoiseCovariance=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            m_stateNoiseCovariance=s_stateSigma^2*eye(s_numberOfVertices);
            t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_maximumTime]);
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %%
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
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
                %KrKKF
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
                
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
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,eye(s_numberOfVertices),t_stateNoiseCovariance,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
                    if s_timeInd>1
                        t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                        t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                        s_auxInd=s_auxInd+1;
                    end
                    if mod(s_timeInd,s_trainTimePeriod)==0
                        % recalculate t_stateNoiseCorrelation
                        t_stateNoiseCovariance=KrKFonGSimulations.reCalculateStateNoiseCov(t_qAuxForSignal,t_qAuxForMSE,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                        s_auxInd=1;
                    end
                    
                    m_estimateKFPrev=m_estimateKF;
                    t_MSEKFPRev=t_MSEKF;
                end
                myLegendKrKF{s_sampleInd}='KKrKF';
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
                myLegendKRR{s_sampleInd}='KRR-IE';
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
                        'BL-IE';
                    
                    
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
                m_meanEstKrKF(s_vertInd,:)=mean(t_krkfEstimate((0:s_maximumTime-1)*...
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
            x=(1:109)';
            dif=setdiff(x,m_positions);
            s_vertexToPlot=dif(1);
            myLegend=[myLegandTrueSignal myLegendDLSR myLegendLMS myLegendBan  myLegendKrKF];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_temperatureTimeSeriesSampled(s_vertexToPlot,:);...
                m_meanEstDLSR(s_vertexToPlot,:);...
                m_meanEstLMS(s_vertexToPlot,:);m_meanEstBan(s_vertexToPlot,:);m_meanEstKrKF(s_vertexToPlot,:)],...
                'xlab','Time evolution','ylab','function value','leg',myLegend);
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_numberOfSamples)];
            
        end
        function F = compute_fig_2218(obj,niter)
            F = obj.load_F_structure(2018);
            %F.ylimit=[0 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            
            F.styles = {'-','.','--',':','-.'};
            F.colorset=[0 0 0;0 .7 0;0 0 .9 ;.5 .5 0 ;1 0 0];
            %F.pos=[680 729 509 249];
            %Initially: True signal KKF KRR-TA DLSR LMS BL-TA
            
            F.leg_pos='southeast';
            
            F.ylab='Temperature [F]';
            F.xlab='Time [hours]';
            %F.tit='Temperature tracking';
            %F.leg_pos = 'northeast';      % it can be 'northwest',
            %F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
         function F = compute_fig_2318(obj,niter)
            F = obj.load_F_structure(2218);
            %F.ylimit=[0 1];
            %F.logy = 1;
            %F.xlimit=[10 100];
            F.leg{5}='KKrKF';
            %F.tit='Temperature tracking';
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
        %  with Bandlimited model KRR at each time t and WangWangGuo paper LMS
        %  Kernel used: Diffusion Kernel in space
        
        
        function F = compute_fig_2019(obj,niter)
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
            
            s_stepLMS=0.6;
            s_muDLSR=1.6;
            s_betaDLSR=0.8;
            %Obs model
            s_obsSigma=0.01;
            v_bandwidthBL=[8,10];
            v_bandwidth=[14,18];
            %Kr KF
            s_stateSigma=0.003;
            s_pctOfTrainPhase=0.4;
            s_transWeight=0.1;
            %Multikernel
            v_sigmaForDiffusion=[1.2,1.3,1.8,1.9];
            s_numberOfKernels=size(v_sigmaForDiffusion,2);
            s_lambdaForMultiKernels=0.001;
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
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            
            
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
            t_invSpatialDiffusionKernel=KrKFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            %KKF part
            [t_correlations,t_transitions]=KrKFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            % Correlation matrices for KrKF
            t_spatialCovariance=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            t_obsNoiseCovariace=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            t_spatialDiffusionKernel=KrKFonGSimulations.createDiffusionKernelsFromTopologies(t_spaceAdjacencyAtDifferentTimes,s_sigmaForDiffusion);
            t_spatialCovariance=t_spatialDiffusionKernel;
            % Transition matrix for KrKf
            t_transitionKrKF=...
                repmat(s_transWeight*eye(s_numberOfVertices),[1,1,s_maximumTime]);
            % Kernels for KrKF
            m_combinedKernel=m_diffusionKernel; % the combined kernel for kriging
            t_dictionaryOfKernels=zeros(s_numberOfKernels,s_numberOfVertices,s_numberOfVertices);
            for s_kernelInd=1:s_numberOfKernels
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_kernelInd),'m_laplacian',graph.getLaplacian);
                t_dictionaryOfKernels(s_kernelInd,:,:)=diffusionGraphKernel.generateKernelMatrix;
            end
            %initialize stateNoise somehow
            
            %m_stateNoiseCovariance=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            
            m_stateNoiseCovariance=s_stateSigma^2*eye(s_numberOfVertices);
            %m_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_maximumTime]);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %% 4.0 Estimate signal
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
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
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
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
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                l2MultiKernelKrigingCovEstimator=L2MultiKernelKrigingCovEstimator...
                    ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma);
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residual=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),m_stateNoiseCovariance,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
                    if s_timeInd>1
                        t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                        t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                        s_auxInd=s_auxInd+1;
                        % save residual matrix
                        %                         for s_monteCarloSimInd=1:s_monteCarloSimulations
                        %                         t_residual(:,s_monteCarloSimInd,s_auxInd)=m_samplest(:,s_monteCarloSimInd)...
                        %                         -m_estimateKF(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd);
                        %                         end
                    end
                    if mod(s_timeInd,s_trainTimePeriod)==0
                        % recalculate t_stateNoiseCorrelation
                        %t_residualCov=KrKFonGSimulations.calculateResidualCov(t_residual,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                        %normalize Cov?
                        %t_residualCov=t_residualCov/1000;
                        %m_theta=l2MultiKernelKrigingCovEstimator.estimateCoeffVector(t_residualCov,m_positionst);
                        m_stateNoiseCovariance=KrKFonGSimulations.reCalculateStateNoiseCov(t_qAuxForSignal,t_qAuxForMSE,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                        s_auxInd=1;
                    end
                    
                    
                    m_estimateKFPrev=m_estimateKF;
                    t_MSEKFPRev=t_MSEKF;
                end
                %% 5. KF estimate
%                 kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
%                     't_previousMinimumSquaredError',t_initialSigma0,...
%                     'm_previousEstimate',m_initialState);
%                 for s_timeInd=1:s_maximumTime
%                     time t indices
%                     v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
%                         (s_timeInd)*s_numberOfVertices;
%                     v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
%                         (s_timeInd)*s_numberOfSamples;
%                     
%                     samples and positions at time t
%                     m_samplest=m_samples(v_timetIndicesForSamples,:);
%                     m_positionst=m_positions(v_timetIndicesForSamples,:);
%                     estimate
%                     
%                     [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
%                         kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
%                         t_transitions(:,:,s_timeInd),...
%                         t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
%                     prepare KF for next iteration
%                     kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
%                     kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
%                         s_sampleInd);
%                     
%                 end
                %% 6. Kernel Ridge Regression
                
%                 nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator...
%                     ('m_kernels',m_diffusionKernel,'s_lambda',s_mu);
%                 for s_timeInd=1:s_maximumTime
%                     %time t indices
%                     v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
%                         (s_timeInd)*s_numberOfVertices;
%                     v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
%                         (s_timeInd)*s_numberOfSamples;
%                     
%                     %samples and positions at time t
%                     m_samplest=m_samples(v_timetIndicesForSamples,:);
%                     m_positionst=m_positions(v_timetIndicesForSamples,:);
%                     %estimate
%                     
%                     [t_kRRestimate(v_timetIndicesForSignals,:,s_sampleInd)]=...
%                         nonParametricGraphFunctionEstimator.estimate...
%                         (m_samplest,m_positionst,s_mu);
%                     
%                 end
                %% 7. bandlimited estimate
                %bandwidth of the bandlimited signal
                
                myLegend={};
                
                
                for s_bandInd=1:size(v_bandwidth,2)
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
            m_relativeErrorKrKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOfKFErrors=zeros(s_maximumTime,1);
                v_normOfKrKFErrors=zeros(s_maximumTime,1);
                v_normOfNotSampled=zeros(s_maximumTime,1);
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
                        v_normOfKFErrors(s_timeInd)=v_normOfKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrKFErrors(s_timeInd)=v_normOfKrKFErrors(s_timeInd)+...
                            norm(t_krkfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfNotSampled(s_timeInd)=v_normOfNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOfNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKrKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrKFErrors(1:s_timeInd))/...
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
                         s_bandwidthBl=v_bandwidthBL(s_bandInd);
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' B=%g',s_bandwidth));
                        
                        myLegendLMS{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=strcat('LMS',...
                            sprintf(' B=%g',s_bandwidth))
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidth,2)+s_bandInd}=...
                            strcat('BL-IE, ',...
                            sprintf(' B=%g',s_bandwidthBl));
                    end
                    myLegendKF{s_sampleInd}='KKF';
                    myLegendKRR{s_sampleInd}='KRR-IE';
                    myLegendKrKF{s_sampleInd}='KKrKF';
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKrKF ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate...
                , m_relativeErrorKrKF]',...
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
        function F = compute_fig_2219(obj,niter)
            F = obj.load_F_structure(2019);
            F.ylimit=[0 1];
            %F.logy = 1;
            F.xlimit=[0 360];
            F.styles = {'-s','-o','--s','--o',':s',':o','-.d'};
            F.colorset=[0 0 0;0 .7 0;1 .5 0 ;.5 .5 0; .9 0 .9 ;0 0 1;1 0 0];
            F.leg{7}='KKrKF';
            s_chunk=30;
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
        % using MLK with frobenious norm betweeen matrices and l2
        function F = compute_fig_2519(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
             
            % maximum signal instances sampled
            s_maximumTime=360;
            
            % sample period: we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=24;
            
            % KKrKF parameters
            %regularization parameter
            s_mu=10^-7;
            s_sigmaForDiffusion=1.8;
            %Obs model
            s_obsSigma=0.01;
            %Kr KF
            s_stateSigma=0.05;
            s_pctOfTrainPhase=0.1;
            s_transWeight=0.4;
            %Multikernel
            v_sigmaForDiffusion=[2.1,0.2,0.7,0.8,1,1.1,1.2,1.3,1.6,1.8,1.9,2];
            s_numberOfKernels=size(v_sigmaForDiffusion,2);
            s_lambdaForMultiKernels=1;
            
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
          
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph          
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
        
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
         
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            
            
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
     
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
 
            %% generate transition, correlation matrices
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
          

            % Transition matrix for KrKf
            t_transitionKrKF=...
                repmat(s_transWeight*eye(s_numberOfVertices),[1,1,s_maximumTime]);
            % Kernels for KrKF
            t_dictionaryOfKernels=zeros(s_numberOfKernels,s_numberOfVertices,s_numberOfVertices);
            v_thetaSpat=ones(s_numberOfKernels,1);
            m_combinedKernel=zeros(s_numberOfVertices,s_numberOfVertices);
            for s_kernelInd=1:s_numberOfKernels
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_kernelInd),'m_laplacian',graph.getLaplacian);
                m_difker=diffusionGraphKernel.generateKernelMatrix;
                t_dictionaryOfKernels(s_kernelInd,:,:)=m_difker;
                m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));

            end
            %initialize stateNoise somehow
            
            %m_stateEvolutionKernel=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            
            m_stateEvolutionKernel=s_stateSigma^2*eye(s_numberOfVertices);
            %m_stateEvolutionKernel=repmat(m_stateEvolutionKernel,[1,1,s_maximumTime]);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %% 4.0 Estimate signal
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));         
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
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
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                l2MultiKernelKrigingCovEstimator=L2MultiKernelKrigingCovEstimator...
                    ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residualSpat=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                t_residualState=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    %m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_spatialCovariance=m_combinedKernel;
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),m_stateEvolutionKernel,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
                    %% Multikernel
                    if s_timeInd>1
                        t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                        t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                        s_auxInd=s_auxInd+1;
                        % save residual matrix
                        for s_monteCarloSimInd=1:s_monteCarloSimulations
                            t_residualSpat(:,s_monteCarloSimInd,s_auxInd)=m_samplest(:,s_monteCarloSimInd)...
                                -m_estimateKF(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd);
                        end
                         m_samplespt=m_samples((s_timeInd-2)*s_numberOfSamples+1:...
                        (s_timeInd-1)*s_numberOfSamples,:);
                        m_positionspt=m_positions((s_timeInd-2)*s_numberOfSamples+1:...
                        (s_timeInd-1)*s_numberOfSamples,:);
                        for s_monteCarloSimInd=1:s_monteCarloSimulations
                            t_residualState(:,s_monteCarloSimInd,s_auxInd)=(m_samplest(:,s_monteCarloSimInd)...
                                -m_estimateKR(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd))...
                            -(m_samplespt(:,s_monteCarloSimInd)...
                                -m_estimateKRPrev(m_positionspt(:,s_monteCarloSimInd),s_monteCarloSimInd));
                        end
                    end
                    if mod(s_timeInd,s_trainTimePeriod)==0
                        % recalculate t_stateNoiseCorrelation
                        t_residualSpatCov=KrKFonGSimulations.calculateResidualCov(t_residualSpat,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                        t_residualStateCov=KrKFonGSimulations.calculateResidualCov(t_residualState,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                     
                        %normalize Cov?
                        tic
                        v_thetaSpat=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorCVX(t_residualSpatCov,m_positionst);
                        v_thetaState=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorCVX(t_residualStateCov,m_positionst);
                        timeCVX=toc
%                         tic
%                         v_thetaSpat=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGD(t_residualSpatCov,m_positionst);
%                         v_thetaState=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGD(t_residualStateCov,m_positionst);
%                         timeGD=toc
                        m_combinedKernel=zeros(s_numberOfVertices);
                        for s_kernelInd=1:s_numberOfKernels
                            m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                        end
                        m_stateEvolutionKernel=zeros(s_numberOfVertices);
                        for s_kernelInd=1:s_numberOfKernels
                            m_stateEvolutionKernel=m_stateEvolutionKernel+v_thetaState(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                        end
                        s_auxInd=1;
                    end
                    
                    
                    m_estimateKFPrev=m_estimateKF;
                    t_MSEKFPRev=t_MSEKF;
                    m_estimateKRPrev=m_estimateKR;

                end

                % 6. Kernel Ridge Regression
                
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
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                %%method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                %%authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidthDLSR,2)
                    s_bandwidth=v_bandwidthDLSR(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
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
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidthDLSR,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKrKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidthLMS,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidthBL,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOfKFErrors=zeros(s_maximumTime,1);
                v_normOfKrKFErrors=zeros(s_maximumTime,1);
                v_normOfNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidthBL,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidthDLSR,2));
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
                        v_normOfKFErrors(s_timeInd)=v_normOfKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrKFErrors(s_timeInd)=v_normOfKrKFErrors(s_timeInd)+...
                            norm(t_krkfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfNotSampled(s_timeInd)=v_normOfNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOfNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKrKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrrErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidthBL,2)
                        
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
                        
                        s_bandwidth=v_bandwidthBL(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidthDLSR,2)+s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorLms(s_timeInd,(s_sampleInd-1)*size(v_bandwidthLMS,2)+s_bandInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd}=...
                            strcat('BL-IE, ',...
                            sprintf(' B=%g',s_bandwidth));
                        s_bandwidth=v_bandwidthDLSR(s_bandInd);
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidthDLSR,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' B=%g',s_bandwidth));
                        s_bandwidth=v_bandwidthLMS(s_bandInd);
                        myLegendLMS{(s_sampleInd-1)*size(v_bandwidthLMS,2)+s_bandInd}=strcat('LMS',...
                            sprintf(' B=%g',s_bandwidth));
                        
                    end
                    myLegendKRR{s_sampleInd}='KRR-IE';
                    myLegendKrKF{s_sampleInd}='KrKKF';
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKRR myLegendKrKF ];
            %myLegend=[myLegendKRR myLegendKrKF ];

            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate...
                , m_relativeErrorKRR,m_relativeErrorKrKF]',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
%             F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorKRR,m_relativeErrorKrKF]',...
%                 'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F.ylimit=[0 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_samplePercentage)];
            
        end
        
         % using MLK with frobenious norm betweeen matrices and l2
        function F = compute_fig_25191(obj,niter)
                %% 0. define parameters
            % maximum signal instances sampled
             
            % maximum signal instances sampled
            s_maximumTime=360;
            
            % sample period: we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=24;
            
            % KKrKF parameters
            %regularization parameter
            s_mu=10^-7;
            s_sigmaForDiffusion=1.8;
            %Obs model
            s_obsSigma=0.01;
            %Kr KF
            s_stateSigma=0.05;
            s_pctOfTrainPhase=0.1;
            s_transWeight=0.4;
            %Multikernel
            v_sigmaForDiffusion=[2.1,0.2,0.7,0.8,1,1.1,1.2,1.3,1.6,1.8,1.9,2];
            s_numberOfKernels=size(v_sigmaForDiffusion,2);
            s_lambdaForMultiKernels=1;
            
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
          
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph          
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
        
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
         
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            s_trainTime=round(s_pctOfTrainPhase*s_maximumTime);

            
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
     
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
 
            %% generate transition, correlation matrices
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
          

            % Transition matrix for KrKf
            t_transitionKrKF=...
                repmat(s_transWeight*eye(s_numberOfVertices),[1,1,s_maximumTime]);
            % Kernels for KrKF
            t_dictionaryOfKernels=zeros(s_numberOfKernels,s_numberOfVertices,s_numberOfVertices);
            v_thetaSpat=ones(s_numberOfKernels,1);
            m_combinedKernel=zeros(s_numberOfVertices,s_numberOfVertices);
            for s_kernelInd=1:s_numberOfKernels
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_kernelInd),'m_laplacian',graph.getLaplacian);
                m_difker=diffusionGraphKernel.generateKernelMatrix;
                t_dictionaryOfKernels(s_kernelInd,:,:)=m_difker;
                m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));

            end
 
            %initialize stateNoise somehow
            
            %m_stateEvolutionKernel=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            
            m_stateEvolutionKernel=s_stateSigma^2*eye(s_numberOfVertices);
            %m_stateEvolutionKernel=repmat(m_stateEvolutionKernel,[1,1,s_maximumTime]);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %% 4.0 Estimate signal
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));         
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
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
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                l2MultiKernelKrigingCovEstimator=L2MultiKernelKrigingCovEstimator...
                    ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=0;
                t_residualSpat=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                t_residualState=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    %m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_spatialCovariance=m_combinedKernel;
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),m_stateEvolutionKernel,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
                    %% Multikernel
                    if s_timeInd>1
                        s_auxInd=s_auxInd+1;
                        t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                        t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                        
                        % save residual matrix
                        for s_monteCarloSimInd=1:s_monteCarloSimulations
                            t_residualSpat(:,s_monteCarloSimInd,s_auxInd)=m_samplest(:,s_monteCarloSimInd)...
                                -m_estimateKF(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd);
                        end
                         m_samplespt=m_samples((s_timeInd-2)*s_numberOfSamples+1:...
                        (s_timeInd-1)*s_numberOfSamples,:);
                        m_positionspt=m_positions((s_timeInd-2)*s_numberOfSamples+1:...
                        (s_timeInd-1)*s_numberOfSamples,:);
                        for s_monteCarloSimInd=1:s_monteCarloSimulations
                            t_residualState(:,s_monteCarloSimInd,s_auxInd)=(m_samplest(:,s_monteCarloSimInd)...
                                -m_estimateKR(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd))...
                            -(m_samplespt(:,s_monteCarloSimInd)...
                                -m_estimateKRPrev(m_positionspt(:,s_monteCarloSimInd),s_monteCarloSimInd));
                        end
                    end
                    if s_timeInd==s_trainTime
                        %calculate exact theta estimate
                        % recalculate t_stateNoiseCorrelation
                        t_residualSpatCov=KrKFonGSimulations.calculateResidualCov(t_residualSpat,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                        m_residualSpatMean=KrKFonGSimulations.calculateResidualMean(t_residualSpat,s_numberOfSamples,s_monteCarloSimulations);
                        t_residualStateCov=KrKFonGSimulations.calculateResidualCov(t_residualState,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                        m_residualStateMean=KrKFonGSimulations.calculateResidualMean(t_residualState,s_numberOfSamples,s_monteCarloSimulations);
                      
                        tic
                        v_thetaSpat=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorCVX(t_residualSpatCov,m_positionst);
                        v_thetaState=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorCVX(t_residualStateCov,m_positionst);
                        timeCVX=toc
%                         tic
%                         v_thetaSpat=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGD(t_residualSpatCov,m_positionst);
%                         v_thetaState=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGD(t_residualStateCov,m_positionst);
%                         timeGD=toc
                        m_combinedKernel=zeros(s_numberOfVertices);
                        for s_kernelInd=1:s_numberOfKernels
                            m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                        end
                        m_stateEvolutionKernel=zeros(s_numberOfVertices);
                        for s_kernelInd=1:s_numberOfKernels
                            m_stateEvolutionKernel=m_stateEvolutionKernel+v_thetaState(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                        end
                        s_auxInd=0;
                        t_residualSpat=zeros(s_numberOfSamples,s_monteCarloSimulations);
                        t_residualState=zeros(s_numberOfSamples,s_monteCarloSimulations);
                    end
                    if s_timeInd>s_trainTime
                        %do a few gradient descent steps
                        % combine using formula
                        
                        %t_residualSpatCovRankOne=KrKFonGSimulations.calculateResidualCov(t_residualSpat,s_numberOfSamples,s_monteCarloSimulations,1);
                        
                        [t_residualSpatCov,m_residualSpatMean]=KrKFonGSimulations.incrementalCalcResCovMean...
                            (t_residualSpatCov,t_residualSpat,s_timeInd,m_residualSpatMean);
                        
                        %t_residualStateCovRankOne=KrKFonGSimulations.calculateResidualCov(t_residualState,s_numberOfSamples,s_monteCarloSimulations,1);
                        [t_residualStateCov,m_residualStateMean]=KrKFonGSimulations.incrementalCalcResCovMean...
                            (t_residualStateCov,t_residualState,s_timeInd,m_residualStateMean);
                        tic
                        v_thetaSpat=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGDWithInit(t_residualSpatCov,m_positionst,v_thetaSpat);
                        v_thetaState=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGDWithInit(t_residualStateCov,m_positionst,v_thetaState);
                        timeGD=toc
                        s_timeInd
                        m_combinedKernel=zeros(s_numberOfVertices);
                        for s_kernelInd=1:s_numberOfKernels
                            m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                        end
                        m_stateEvolutionKernel=zeros(s_numberOfVertices);
                        for s_kernelInd=1:s_numberOfKernels
                            m_stateEvolutionKernel=m_stateEvolutionKernel+v_thetaState(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                        end
                        s_auxInd=0;
                    end
                    
                    
                    m_estimateKFPrev=m_estimateKF;
                    t_MSEKFPRev=t_MSEKF;
                    m_estimateKRPrev=m_estimateKR;

                end

                % 6. Kernel Ridge Regression
                
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
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                %%method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                %%authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidthDLSR,2)
                    s_bandwidth=v_bandwidthDLSR(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
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
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidthDLSR,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKrKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidthLMS,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidthBL,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOfKFErrors=zeros(s_maximumTime,1);
                v_normOfKrKFErrors=zeros(s_maximumTime,1);
                v_normOfNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidthBL,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidthDLSR,2));
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
                        v_normOfKFErrors(s_timeInd)=v_normOfKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrKFErrors(s_timeInd)=v_normOfKrKFErrors(s_timeInd)+...
                            norm(t_krkfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfNotSampled(s_timeInd)=v_normOfNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOfNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKrKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrrErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidthBL,2)
                        
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
                        
                        s_bandwidth=v_bandwidthBL(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidthDLSR,2)+s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorLms(s_timeInd,(s_sampleInd-1)*size(v_bandwidthLMS,2)+s_bandInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd}=...
                            strcat('BL-IE, ',...
                            sprintf(' B=%g',s_bandwidth));
                        s_bandwidth=v_bandwidthDLSR(s_bandInd);
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidthDLSR,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' B=%g',s_bandwidth));
                        s_bandwidth=v_bandwidthLMS(s_bandInd);
                        myLegendLMS{(s_sampleInd-1)*size(v_bandwidthLMS,2)+s_bandInd}=strcat('LMS',...
                            sprintf(' B=%g',s_bandwidth));
                        
                    end
                    myLegendKRR{s_sampleInd}='KRR-IE';
                    myLegendKrKF{s_sampleInd}='KrKKF';
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKRR myLegendKrKF ];
            %myLegend=[myLegendKRR myLegendKrKF ];

            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate...
                , m_relativeErrorKRR,m_relativeErrorKrKF]',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
%             F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorKRR,m_relativeErrorKrKF]',...
%                 'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F.ylimit=[0 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_samplePercentage)];
            
        end
        
          % using MLK with mininization of trace betweeen matrices and l2
        function F = compute_fig_25192(obj,niter)
                %% 0. define parameters
            % maximum signal instances sampled
             
            % maximum signal instances sampled
            s_maximumTime=360;
            
            % sample period: we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=24;
            
            % KKrKF parameters
            %regularization parameter
            s_mu=10^-7;
            s_sigmaForDiffusion=1.8;
            %Obs model
            s_obsSigma=0.01;
            %Kr KF
            s_stateSigma=0.05;
            s_pctOfTrainPhase=0.30;
            s_transWeight=0.004;
            %Multikernel
            v_sigmaForDiffusion=[2.5,2.1,0.2,0.7,0.8,1,1.1,1.2,1.3,1.6,1.8,1.9,2];
            s_mean=1;
            s_std=2;
            v_sigmaForDiffusion= abs(s_mean+ s_std.*randn(100,1)');
            s_numberOfKernels=size(v_sigmaForDiffusion,2);
            s_lambdaForMultiKernels=10^-8;
            
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
          
            load('temperatureTimeSeriesData.mat');
            m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph          
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
        
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
         
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            s_trainTime=round(s_pctOfTrainPhase*s_maximumTime);

            
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
     
            %% 2. choise of Kernel must be positive definite
            % diffusion kernel
            graph=Graph('m_adjacency',m_adjacency);
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
 
            %% generate transition, correlation matrices
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
          

            % Transition matrix for KrKf
            t_transitionKrKF=...
                repmat(s_transWeight*eye(s_numberOfVertices),[1,1,s_maximumTime]);
            % Kernels for KrKF
            t_dictionaryOfKernels=zeros(s_numberOfKernels,s_numberOfVertices,s_numberOfVertices);
            t_dictionaryOfEigenValues=zeros(s_numberOfKernels,s_numberOfVertices,s_numberOfVertices);
            v_thetaSpat=ones(s_numberOfKernels,1);
            m_combinedKernel=zeros(s_numberOfVertices,s_numberOfVertices);
            [m_eigenvectorsAll,m_eigenvaluesAll]=KrKFonGSimulations.transformedDifEigValues(graph.getLaplacian,v_sigmaForDiffusion);
            for s_kernelInd=1:s_numberOfKernels
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_kernelInd),'m_laplacian',graph.getLaplacian);
                m_difker=diffusionGraphKernel.generateKernelMatrix;
                t_dictionaryOfKernels(s_kernelInd,:,:)=m_difker;
                m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                t_dictionaryOfEigenValues(s_kernelInd,:,:)=diag(m_eigenvaluesAll(:,s_kernelInd));
            end

         
            
            m_stateEvolutionKernel=s_stateSigma^2*eye(s_numberOfVertices);
            %m_stateEvolutionKernel=repmat(m_stateEvolutionKernel,[1,1,s_maximumTime]);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %% 4.0 Estimate signal
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));         
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
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
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                l2MultiKernelKrigingCorEstimator=L2MultiKernelKrigingCorEstimator...
                    ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
                 l2MultiKernelKrigingCorEstimatorEig=L2MultiKernelKrigingCorEstimator...
                    ('t_kernelDictionary',t_dictionaryOfEigenValues,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
                 l1MultiKernelKrigingCorEstimator=L1MultiKernelKrigingCorEstimator...
                    ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
                batchEmpiricalMeanCovEstimator=BatchEmpiricalMeanCovEstimator;
                % used for parameter estimation
                s_auxInd=0;
                t_approximSpat=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_residualState=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    %m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_spatialCovariance=m_combinedKernel;
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),m_stateEvolutionKernel,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
                    %% Multikernel
                    if s_trainTime>s_timeInd>1
                        s_auxInd=s_auxInd+1;
                        % save approximate matrix
                         t_approximSpat(:,:,s_auxInd)=m_estimateKR;
                            t_residualState(:,:,s_auxInd)=m_estimateKF...
                            -t_transitionKrKF(:,:,s_timeInd)*m_estimateKFPrev;

                    end
                    if s_timeInd==s_trainTime
                        %calculate exact theta estimate
                        t_approxSpatCor=batchEmpiricalMeanCovEstimator.calculateResidualCorBatch(t_approximSpat,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                        t_approxStateCor=batchEmpiricalMeanCovEstimator.calculateResidualCorBatch(t_residualState,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                        for s_monteCarloInd=1:s_monteCarloSimulations
                            t_transformedSpatCor(:,:,s_monteCarloInd)=m_eigenvectorsAll'*t_approxSpatCor(:,:,s_monteCarloInd)*m_eigenvectorsAll;
                            t_transformedStateCor(:,:,s_monteCarloInd)=m_eigenvectorsAll'*t_approxStateCor(:,:,s_monteCarloInd)*m_eigenvectorsAll;
                        end
                        
                        tic
                        v_thetaSpat=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorCVX(t_approxSpatCor);
                        v_thetaSpatl2eig=l2MultiKernelKrigingCorEstimatorEig.estimateCoeffVectorCVX(t_transformedSpatCor);
                        %v_thetaSpat=l1MultiKernelKrigingCorEstimator.estimateCoeffVectorCVX(t_approxSpatCor);
                        %v_thetaSpat=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorGD(t_transformedSpatCor,m_eigenvaluesAll);
                        %v_thetaSpat=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorNewton(t_transformedSpatCor,m_eigenvaluesAll);
                        %v_theta=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorGDWithInit(t_transformedSpatCor,m_eigenvaluesAll,v_thetaSpat);
                        v_thetaState=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorCVX(t_approxStateCor);
                        timeCVX=toc
%                         tic
%                         v_thetaSpat=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGD(t_residualSpatCov,m_positionst);
%                         v_thetaState=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGD(t_residualStateCov,m_positionst);
%                         timeGD=toc
                        m_combinedKernel=zeros(s_numberOfVertices);
                        for s_kernelInd=1:s_numberOfKernels
                            m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                        end
                        m_stateEvolutionKernel=zeros(s_numberOfVertices);
                        for s_kernelInd=1:s_numberOfKernels
                            m_stateEvolutionKernel=m_stateEvolutionKernel+v_thetaState(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                        end
                        m_stateEvolutionKernel=s_stateSigma^2*m_stateEvolutionKernel;
                        s_auxInd=0;
                        t_approximSpat=zeros(s_numberOfSamples,s_monteCarloSimulations);
                        t_residualState=zeros(s_numberOfSamples,s_monteCarloSimulations);
                    end
%                     if s_timeInd>s_trainTime
%                         do a few gradient descent steps
%                         combine using formula
%                         
%                         t_residualSpatCovRankOne=KrKFonGSimulations.calculateResidualCov(t_residualSpat,s_numberOfSamples,s_monteCarloSimulations,1);
%                         
%                         [t_approxSpatCor,m_residualSpatMean]=KrKFonGSimulations.incrementalCalcResCovMean...
%                             (t_approxSpatCor,t_approximSpat,s_timeInd,m_residualSpatMean);
%                         
%                         t_residualStateCovRankOne=KrKFonGSimulations.calculateResidualCov(t_residualState,s_numberOfSamples,s_monteCarloSimulations,1);
%                         [t_approxStateCor,m_residualStateMean]=KrKFonGSimulations.incrementalCalcResCovMean...
%                             (t_approxStateCor,t_residualState,s_timeInd,m_residualStateMean);
%                         tic
%                         v_thetaSpat=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorGDWithInit(t_approxSpatCor,m_positionst,v_thetaSpat);
%                         v_thetaState=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorGDWithInit(t_approxStateCor,m_positionst,v_thetaState);
%                         timeGD=toc
%                         s_timeInd
%                         m_combinedKernel=zeros(s_numberOfVertices);
%                         for s_kernelInd=1:s_numberOfKernels
%                             m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
%                         end
%                         m_stateEvolutionKernel=zeros(s_numberOfVertices);
%                         for s_kernelInd=1:s_numberOfKernels
%                             m_stateEvolutionKernel=m_stateEvolutionKernel+v_thetaState(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
%                         end
%                         s_auxInd=0;
%                     end
                    
                    
                    m_estimateKFPrev=m_estimateKF;
                    t_MSEKFPRev=t_MSEKF;
                    m_estimateKRPrev=m_estimateKR;

                end

                % 6. Kernel Ridge Regression
                
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
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                %%method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                %%authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidthDLSR,2)
                    s_bandwidth=v_bandwidthDLSR(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
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
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidthDLSR,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKrKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidthLMS,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidthBL,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOfKFErrors=zeros(s_maximumTime,1);
                v_normOfKrKFErrors=zeros(s_maximumTime,1);
                v_normOfNotSampled=zeros(s_maximumTime,1);
                v_normOfKrrErrors=zeros(s_maximumTime,1);
                m_normOfBLErrors=zeros(s_maximumTime,size(v_bandwidthBL,2));
                m_normOfDLSRErrors=zeros(s_maximumTime,size(v_bandwidthDLSR,2));
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
                        v_normOfKFErrors(s_timeInd)=v_normOfKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrKFErrors(s_timeInd)=v_normOfKrKFErrors(s_timeInd)+...
                            norm(t_krkfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfNotSampled(s_timeInd)=v_normOfNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOfNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKrKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKRR(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrrErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    for s_bandInd=1:size(v_bandwidthBL,2)
                        
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
                        
                        s_bandwidth=v_bandwidthBL(s_bandInd);
                        m_relativeErrorDistr(s_timeInd,(s_sampleInd-1)*size(v_bandwidthDLSR,2)+s_bandInd)=...
                            sum(m_normOfDLSRErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        m_relativeErrorLms(s_timeInd,(s_sampleInd-1)*size(v_bandwidthLMS,2)+s_bandInd)=...
                            sum(m_normOfLMSErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;
                        m_relativeErrorbandLimitedEstimate(s_timeInd,(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd)...
                            =sum(m_normOfBLErrors((1:s_timeInd),s_bandInd))/...
                            s_summedNorm;%s_timeInd*s_numberOfVertices;
                        myLegendBan{(s_sampleInd-1)*size(v_bandwidthBL,2)+s_bandInd}=...
                            strcat('BL-IE, ',...
                            sprintf(' B=%g',s_bandwidth));
                        s_bandwidth=v_bandwidthDLSR(s_bandInd);
                        myLegendDLSR{(s_sampleInd-1)*size(v_bandwidthDLSR,2)+s_bandInd}=strcat('DLSR',...
                            sprintf(' B=%g',s_bandwidth));
                        s_bandwidth=v_bandwidthLMS(s_bandInd);
                        myLegendLMS{(s_sampleInd-1)*size(v_bandwidthLMS,2)+s_bandInd}=strcat('LMS',...
                            sprintf(' B=%g',s_bandwidth));
                        
                    end
                    myLegendKRR{s_sampleInd}='KRR-IE';
                    myLegendKrKF{s_sampleInd}='KrKKF';
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKRR myLegendKrKF ];
            %myLegend=[myLegendKRR myLegendKrKF ];

            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate...
                , m_relativeErrorKRR,m_relativeErrorKrKF]',...
                'xlab','Time evolution','ylab','NMSE','leg',myLegend);
%             F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorKRR,m_relativeErrorKrKF]',...
%                 'xlab','Time evolution','ylab','NMSE','leg',myLegend);
            F.ylimit=[0 1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_samplePercentage)];
            
        end
        
        % using MLK with frobenious norm betweeen matrices and l1
          function F = compute_fig_2619(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=1000;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=2;
            s_mu=10^-7;
            
            s_sigmaForDiffusion=1.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.2:0.2:0.2);
            
            
            %v_bandwidthPercentage=[0.01,0.1];
            
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %Obs model
            s_obsSigma=0.01;
            %Kr KF
            s_stateSigma=0.05;
            s_pctOfTrainPhase=0.2;
            s_transWeight=0.4;
            %Multikernel
            v_sigmaParDiffusion=[2.1,0.7,0.8,1,1.1,1.2,1.3,1.5,1.8,1.9,2,2.2,10];
            s_numberOfKernels=size(v_sigmaParDiffusion,2);
            s_lambdaForMultiKernels=1;
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
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            
            
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
            t_invSpatialDiffusionKernel=KrKFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            %KKF part
            [t_correlations,t_transitions]=KrKFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            % Correlation matrices for KrKF
            t_spatialCovariance=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            t_obsNoiseCovariace=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            t_spatialDiffusionKernel=KrKFonGSimulations.createDiffusionKernelsFromTopologies(t_spaceAdjacencyAtDifferentTimes,s_sigmaForDiffusion);
            t_spatialCovariance=t_spatialDiffusionKernel;
            % Transition matrix for KrKf
            t_transitionKrKF=...
                repmat(s_transWeight*eye(s_numberOfVertices),[1,1,s_maximumTime]);
            % Kernels for KrKF
            m_combinedKernel=m_diffusionKernel; % the combined kernel for kriging
            t_dictionaryOfKernels=zeros(s_numberOfKernels,s_numberOfVertices,s_numberOfVertices);
            for s_kernelInd=1:s_numberOfKernels
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaParDiffusion(s_kernelInd),'m_laplacian',graph.getLaplacian);
                m_difker=diffusionGraphKernel.generateKernelMatrix;
                t_dictionaryOfKernels(s_kernelInd,:,:)=m_difker;
            end
            %initialize stateNoise somehow
            
            t_stateEvolutionKernel=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            
            m_stateEvolutionKernel=s_stateSigma^2*eye(s_numberOfVertices);
            t_stateEvolutionKernel=repmat(m_stateEvolutionKernel,[1,1,s_maximumTime]);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %% 4.0 Estimate signal
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));         
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
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
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                l1MultiKernelKrigingCovEstimator=L1MultiKernelKrigingCovEstimator...
                    ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residual=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    %m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_spatialCovariance=m_combinedKernel;
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),t_stateEvolutionKernel,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
                    if s_timeInd>1
                        t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                        t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                        s_auxInd=s_auxInd+1;
                        % save residual matrix
                        for s_monteCarloSimInd=1:s_monteCarloSimulations
                            t_residual(:,s_monteCarloSimInd,s_auxInd)=m_samplest(:,s_monteCarloSimInd)...
                                -m_estimateKF(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd);
                        end
                        m_samplespt=m_samples((s_timeInd-2)*s_numberOfSamples+1:...
                        (s_timeInd-1)*s_numberOfSamples,:);
                        m_positionspt=m_positions((s_timeInd-2)*s_numberOfSamples+1:...
                        (s_timeInd-1)*s_numberOfSamples,:);
                        for s_monteCarloSimInd=1:s_monteCarloSimulations
                            t_residual(:,s_monteCarloSimInd,s_auxInd)=(m_samplest(:,s_monteCarloSimInd)...
                                -m_estimateKR(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd))
                            -(m_samplespt(:,s_monteCarloSimInd)...
                                -m_estimateKRPrev(m_positionspt(:,s_monteCarloSimInd),s_monteCarloSimInd));
                        end
                        
                    end
                    if mod(s_timeInd,s_trainTimePeriod)==0
                        % recalculate t_stateNoiseCorrelation
                        t_residualCov=KrKFonGSimulations.calculateResidualCov(t_residual,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                        %normalize Cov?
                        t_residualCov=t_residualCov;
                        %v_theta1=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorCVX(t_residualCov,m_positionst);
                        v_theta=l1MultiKernelKrigingCovEstimator.estimateCoeffVectorCVX(t_residualCov,m_positionst);

                        m_combinedKernel=zeros(s_numberOfVertices);
                        for s_kernelInd=1:s_numberOfKernels
                            m_combinedKernel=m_combinedKernel+v_theta(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                        end
                        t_stateEvolutionKernel=KrKFonGSimulations.reCalculateStateNoiseCov(t_qAuxForSignal,t_qAuxForMSE,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                        s_auxInd=1;
                    end
                    
                    
                    m_estimateKFPrev=m_estimateKF;
                    t_MSEKFPRev=t_MSEKF;
                    m_estimateKRPrev=m_estimateKR;
                end
                %% 5. KF estimate
%                 kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
%                     't_previousMinimumSquaredError',t_initialSigma0,...
%                     'm_previousEstimate',m_initialState);
%                 for s_timeInd=1:s_maximumTime
%                     time t indices
%                     v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
%                         (s_timeInd)*s_numberOfVertices;
%                     v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
%                         (s_timeInd)*s_numberOfSamples;
%                     
%                     samples and positions at time t
%                     m_samplest=m_samples(v_timetIndicesForSamples,:);
%                     m_positionst=m_positions(v_timetIndicesForSamples,:);
%                     estimate
%                     
%                     [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
%                         kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
%                         t_transitions(:,:,s_timeInd),...
%                         t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
%                     prepare KF for next iteration
%                     kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
%                     kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
%                         s_sampleInd);
%                     
%                 end
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
                
%                 
%                 for s_bandInd=1:size(v_bandwidth,2)
%                     s_bandwidth=v_bandwidth(s_bandInd);
%                     for s_timeInd=1:s_maximumTime
%                         time t indices
%                         v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
%                             (s_timeInd)*s_numberOfVertices;
%                         v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
%                             (s_timeInd)*s_numberOfSamples;
%                         
%                         samples and positions at time t
%                         
%                         m_samplest=m_samples(v_timetIndicesForSamples,:);
%                         m_positionst=m_positions(v_timetIndicesForSamples,:);
%                         create take diagonals from extended graph
%                         m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
%                         grapht=Graph('m_adjacency',m_adjacency);
%                         
%                         bandlimited estimate
%                         bandlimitedGraphFunctionEstimator= ...
%                             BandlimitedGraphFunctionEstimator('m_laplacian'...
%                             ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
%                         t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
%                             bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
%                         
%                     end
%                     
%                     
%                 end
%                 
%                 % 8.DistributedFullTrackingAlgorithmEstimator
%                 method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
%                 authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
%                 
%                 
%                 for s_bandInd=1:size(v_bandwidth,2)
%                     s_bandwidth=v_bandwidth(s_bandInd);
%                     distributedFullTrackingAlgorithmEstimator=...
%                         DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
%                         's_bandwidth',s_bandwidth,'graph',graph);
%                     t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
%                         distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
%                     
%                     
%                 end
%                 % . LMS
%                 for s_bandInd=1:size(v_bandwidth,2)
%                     s_bandwidth=v_bandwidth(s_bandInd);
%                     m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
%                     grapht=Graph('m_adjacency',m_adjacency);
%                     lMSFullTrackingAlgorithmEstimator=...
%                         LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
%                         's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
%                     t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
%                         lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
%                     
%                     
%                 end
            end
            
            
            %% 9. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKrKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOfKFErrors=zeros(s_maximumTime,1);
                v_normOfKrKFErrors=zeros(s_maximumTime,1);
                v_normOfNotSampled=zeros(s_maximumTime,1);
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
                        v_normOfKFErrors(s_timeInd)=v_normOfKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrKFErrors(s_timeInd)=v_normOfKrKFErrors(s_timeInd)+...
                            norm(t_krkfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfNotSampled(s_timeInd)=v_normOfNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOfNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKrKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrKFErrors(1:s_timeInd))/...
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
                    myLegendKF{s_sampleInd}='KKF';
                    myLegendKRR{s_sampleInd}='KRR-IE';
                    myLegendKrKF{s_sampleInd}='KrKKF';
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKRR myLegendKF myLegendKrKF ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate...
                , m_relativeErrorKRR,m_relativeErrorKf,m_relativeErrorKrKF]',...
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
        
        % plot estimates path index and time 
        function F = compute_fig_2518(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=360;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=24;
            s_mu=10^-7;
            
            s_sigmaForDiffusion=1.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.4:0.4:0.4);
            
            
            %v_bandwidthPercentage=[0.01,0.1];
            
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %Obs model
            s_obsSigma=0.01;
            %Kr KF
            s_stateSigma=0.05;
            s_pctOfTrainPhase=0.1;
            s_transWeight=0.4;
            %Multikernel
            v_sigmaForDiffusion=[2.1,0.2,0.7,0.8,1,1.1,1.2,1.3,1.6,1.8,1.9,2];
            

            s_numberOfKernels=size(v_sigmaForDiffusion,2);
            s_lambdaForMultiKernels=1;
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
            v_bandwidthBL=[8,10];
            v_bandwidthLMS=[14,18];
            v_bandwidthDLSR=[10,12];
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
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            
            
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
%             t_invSpatialDiffusionKernel=KrKFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            %KKF part
%             [t_correlations,t_transitions]=KrKFonGSimulations.kernelRegressionRecursion...
%                 (t_invSpatialDiffusionKernel...
%                 ,-t_timeAdjacencyAtDifferentTimes...
%                 ,s_maximumTime,s_numberOfVertices,m_sigma0);
            % Correlation matrices for KrKF
            %t_spatialDiffusionKernel=KrKFonGSimulations.createDiffusionKernelsFromTopologies(t_spaceAdjacencyAtDifferentTimes,s_sigmaForDiffusion);
            %t_spatialCovariance=t_spatialDiffusionKernel;
            % Transition matrix for KrKf
            t_transitionKrKF=...
                repmat(s_transWeight*eye(s_numberOfVertices),[1,1,s_maximumTime]);
            % Kernels for KrKF
            t_dictionaryOfKernels=zeros(s_numberOfKernels,s_numberOfVertices,s_numberOfVertices);
            v_thetaSpat=ones(s_numberOfKernels,1);
            m_combinedKernel=zeros(s_numberOfVertices,s_numberOfVertices);
            for s_kernelInd=1:s_numberOfKernels
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_kernelInd),'m_laplacian',graph.getLaplacian);
                m_difker=diffusionGraphKernel.generateKernelMatrix;
                t_dictionaryOfKernels(s_kernelInd,:,:)=m_difker;
                m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));

            end
 
            %initialize stateNoise somehow
            
            %m_stateEvolutionKernel=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            
            m_stateEvolutionKernel=s_stateSigma^2*eye(s_numberOfVertices);
            %m_stateEvolutionKernel=repmat(m_stateEvolutionKernel,[1,1,s_maximumTime]);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %% 4.0 Estimate signal
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));         
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
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
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                l2MultiKernelKrigingCovEstimator=L2MultiKernelKrigingCovEstimator...
                    ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residualSpat=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                t_residualState=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    %m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_spatialCovariance=m_combinedKernel;
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),m_stateEvolutionKernel,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
                    %% Multikernel
                    if s_timeInd>1
                        t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                        t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                        s_auxInd=s_auxInd+1;
                        % save residual matrix
                        for s_monteCarloSimInd=1:s_monteCarloSimulations
                            t_residualSpat(:,s_monteCarloSimInd,s_auxInd)=m_samplest(:,s_monteCarloSimInd)...
                                -m_estimateKF(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd);
                        end
                         m_samplespt=m_samples((s_timeInd-2)*s_numberOfSamples+1:...
                        (s_timeInd-1)*s_numberOfSamples,:);
                        m_positionspt=m_positions((s_timeInd-2)*s_numberOfSamples+1:...
                        (s_timeInd-1)*s_numberOfSamples,:);
                        for s_monteCarloSimInd=1:s_monteCarloSimulations
                            t_residualState(:,s_monteCarloSimInd,s_auxInd)=(m_samplest(:,s_monteCarloSimInd)...
                                -m_estimateKR(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd))...
                            -(m_samplespt(:,s_monteCarloSimInd)...
                                -m_estimateKRPrev(m_positionspt(:,s_monteCarloSimInd),s_monteCarloSimInd));
                        end
                    end
                    if mod(s_timeInd,s_trainTimePeriod)==0
                        % recalculate t_stateNoiseCorrelation
                        t_residualSpatCov=KrKFonGSimulations.calculateResidualCov(t_residualSpat,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                        
                        t_residualStateCov=KrKFonGSimulations.calculateResidualCov(t_residualState,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);

                        %normalize Cov?
                        tic
                        v_thetaSpat=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorCVX(t_residualSpatCov,m_positionst);
                        v_thetaState=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorCVX(t_residualStateCov,m_positionst);
                        timeCVX=toc
%                         tic
%                         v_thetaSpat=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGD(t_residualSpatCov,m_positionst);
%                         v_thetaState=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGD(t_residualStateCov,m_positionst);
%                         timeGD=toc
                        m_combinedKernel=zeros(s_numberOfVertices);
                        for s_kernelInd=1:s_numberOfKernels
                            m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                        end
                        m_stateEvolutionKernel=zeros(s_numberOfVertices);
                        for s_kernelInd=1:s_numberOfKernels
                            m_stateEvolutionKernel=m_stateEvolutionKernel+v_thetaState(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                        end
                        s_auxInd=1;
                    end
                    
                    
                    m_estimateKFPrev=m_estimateKF;
                    t_MSEKFPRev=t_MSEKF;
                    m_estimateKRPrev=m_estimateKR;

                end

                % 6. Kernel Ridge Regression
                
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
                        t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
                            bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
                        
                    end
                    
                    
                end
                
                %% 8.DistributedFullTrackingAlgorithmEstimator
                %%method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
                %%authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
                
                
                for s_bandInd=1:size(v_bandwidthDLSR,2)
                    s_bandwidth=v_bandwidthDLSR(s_bandInd);
                    distributedFullTrackingAlgorithmEstimator=...
                        DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
                        's_bandwidth',s_bandwidth,'graph',graph);
                    t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
                        distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
                    
                    
                end
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
            % myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKRR myLegendKrKF ];
             
                     t_lmsEstimateReduced=squeeze(mean(t_lmsEstimate,2));
            t_bandLimitedEstimateReduced=squeeze(mean(t_bandLimitedEstimate,2));
            t_krkfEstimateReduced=mean(t_krkfEstimate,2);
            t_distrEstimateReduced=squeeze(mean(t_distrEstimate,2));
            t_kRRestimateReduced=mean(t_kRRestimate,2);
            m_krkfEstimate=reshape(t_krkfEstimateReduced,s_numberOfVertices,s_maximumTime);
            m_bandLimitedEstimate=reshape(t_bandLimitedEstimateReduced(:,1),s_numberOfVertices,s_maximumTime);
             m_lmsEstimate=reshape(t_lmsEstimateReduced(:,1),s_numberOfVertices,s_maximumTime);
             m_distrEstimate=reshape(t_distrEstimateReduced(:,1),s_numberOfVertices,s_maximumTime);
            m_kRREstimate=reshape(t_kRRestimateReduced,s_numberOfVertices,s_maximumTime);
            Ftruefunction = F_figure('X',(1:s_maximumTime),'Y',(1:s_numberOfVertices),'Z',reshape(m_graphFunction(:,1),s_numberOfVertices,s_maximumTime),...
                'xlab','Time [day]','ylab','Station','zlab','Temperature','leg','True');
            Fkrkf = F_figure('X',(1:s_maximumTime),'Y',(1:s_numberOfVertices),'Z',m_krkfEstimate,...
                'xlab','Time [day]','ylab','Station','zlab','Temperature','leg','KrKKF');
             Fband = F_figure('X',(1:s_maximumTime),'Y',(1:s_numberOfVertices),'Z',m_bandLimitedEstimate,...
                'xlab','Time [day]','ylab','Station','zlab','Temperature','leg','BL-IE');
             Flms = F_figure('X',(1:s_maximumTime),'Y',(1:s_numberOfVertices),'Z',m_lmsEstimate,...
                'xlab','Time [day]','ylab','Station','zlab','Temperature','leg','LMS');
             Fdlsr = F_figure('X',(1:s_maximumTime),'Y',(1:s_numberOfVertices),'Z',m_distrEstimate,...
                'xlab','Time [day]','ylab','Station','zlab','Temperature','leg','DLSR');
            FkRR = F_figure('X',(1:s_maximumTime),'Y',(1:s_numberOfVertices),'Z',m_kRREstimate,...
                'xlab','Time [day]','ylab','Station','zlab','Temperature','leg','KRR-IE');
            %normalize errors
            F=F_figure('multiplot_array',[Ftruefunction,Fkrkf,Fdlsr,Flms,Fband,FkRR]');
        
            %myLegend=[myLegendKRR myLegendKrKF ];

  
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_mu),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusion)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('weight of diagonal scaling =%g\n',v_propagationWeight)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_samplePercentage)];
            
        end
        
          function F = compute_fig_25181(obj,niter)
            F = obj.load_F_structure(2518);
            %F.multiplot_array(1).plot_type_3D='plot3';
            %F.multiplot_array(1).view_from_top = 0;
          end
        % plot temperature maps
        function F = compute_fig_2719(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=400;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=2;
            s_mu=10^-7;
            
            s_sigmaForDiffusion=1.8;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.6:0.6:0.6);
            
            
            %v_bandwidthPercentage=[0.01,0.1];
            
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %Obs model
            s_obsSigma=0.01;
            %Kr KF
            s_stateSigma=0.005;
            s_pctOfTrainPhase=0.3;
            s_transWeight=0.1;
            %Multikernel
            v_sigmaForDiffusion=[0.8,1.2,1.3,1.8,1.9,2,3,8,10];
            s_numberOfKernels=size(v_sigmaForDiffusion,2);
            s_lambdaForMultiKernels=0.0001;
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
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            
            
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
            t_invSpatialDiffusionKernel=KrKFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            %KKF part
            [t_correlations,t_transitions]=KrKFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            % Correlation matrices for KrKF
            t_spatialCovariance=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            t_obsNoiseCovariace=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            t_spatialDiffusionKernel=KrKFonGSimulations.createDiffusionKernelsFromTopologies(t_spaceAdjacencyAtDifferentTimes,s_sigmaForDiffusion);
            t_spatialCovariance=t_spatialDiffusionKernel;
            % Transition matrix for KrKf
            t_transitionKrKF=...
                repmat(s_transWeight*eye(s_numberOfVertices),[1,1,s_maximumTime]);
            % Kernels for KrKF
            m_combinedKernel=m_diffusionKernel; % the combined kernel for kriging
            t_dictionaryOfKernels=zeros(s_numberOfKernels,s_numberOfVertices,s_numberOfVertices);
            for s_kernelInd=1:s_numberOfKernels
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_kernelInd),'m_laplacian',graph.getLaplacian);
                m_difker=diffusionGraphKernel.generateKernelMatrix;
                t_dictionaryOfKernels(s_kernelInd,:,:)=m_difker/max(max(m_difker));
            end
            %initialize stateNoise somehow
            
            t_stateNoiseCovariance=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            
            m_stateNoiseCovariance=s_stateSigma^2*eye(s_numberOfVertices);
            t_stateNoiseCovariance=repmat(m_stateNoiseCovariance,[1,1,s_maximumTime]);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_temperatureTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %% 4.0 Estimate signal
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
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
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
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
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                eigDistMultiKernelKrigingCovEstimator=EigDistMultiKernelKrigingCovEstimator...
                    ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residual=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    %m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_spatialCovariance=m_combinedKernel;
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),t_stateNoiseCovariance,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
                    if s_timeInd>1
                        t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
                        t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
                        s_auxInd=s_auxInd+1;
                        % save residual matrix
                        for s_monteCarloSimInd=1:s_monteCarloSimulations
                            t_residual(:,s_monteCarloSimInd,s_auxInd)=m_samplest(:,s_monteCarloSimInd)...
                                -m_estimateKF(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd);
                        end
                    end
                    if mod(s_timeInd,s_trainTimePeriod)==0
                        % recalculate t_stateNoiseCorrelation
                        t_residualCov=KrKFonGSimulations.calculateResidualCov(t_residual,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                        %normalize Cov?
                        t_residualCov=t_residualCov;
                        v_theta=eigDistMultiKernelKrigingCovEstimator.estimateCoeffVector(t_residualCov,m_positionst);
                        m_combinedKernel=zeros(s_numberOfVertices);
                        for s_kernelInd=1:s_numberOfKernels
                            m_combinedKernel=m_combinedKernel+v_theta(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                        end
                        t_stateNoiseCovariance=KrKFonGSimulations.reCalculateStateNoiseCov(t_qAuxForSignal,t_qAuxForMSE,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                        s_auxInd=1;
                    end
                    
                    
                    m_estimateKFPrev=m_estimateKF;
                    t_MSEKFPRev=t_MSEKF;
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
            m_relativeErrorKrKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOfKFErrors=zeros(s_maximumTime,1);
                v_normOfKrKFErrors=zeros(s_maximumTime,1);
                v_normOfNotSampled=zeros(s_maximumTime,1);
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
                        v_normOfKFErrors(s_timeInd)=v_normOfKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrKFErrors(s_timeInd)=v_normOfKrKFErrors(s_timeInd)+...
                            norm(t_krkfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfNotSampled(s_timeInd)=v_normOfNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOfNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKrKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrKFErrors(1:s_timeInd))/...
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
                    myLegendKF{s_sampleInd}='KKF';
                    myLegendKRR{s_sampleInd}='KRR-IE';
                    myLegendKrKF{s_sampleInd}='KrKKF';
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKRR myLegendKF myLegendKrKF ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate...
                , m_relativeErrorKRR,m_relativeErrorKf,m_relativeErrorKrKF]',...
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
        
        %% Earthquake dataset
        
        % using MLK with frobenious norm betweeen matrices and l1
          function F = compute_fig_6619(obj,niter)
            %% 0. define parameters
            % maximum signal instances sampled
            
            s_maximumTime=1000;
            % period of sample we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            s_mu=10^-7;
            
            s_sigmaForDiffusion=0.2;
            s_monteCarloSimulations=niter;
            s_SNR=Inf;
            v_samplePercentage=(0.2:0.2:0.2);
            
            
            %v_bandwidthPercentage=[0.01,0.1];
            
            s_stepLMS=0.6;
            s_muDLSR=1.2;
            s_betaDLSR=0.5;
            %Obs model
            s_obsSigma=0.01;
            %Kr KF
            s_stateSigma=0.00005;
            s_pctOfTrainPhase=0.2;
            s_transWeight=0.4;
            %Multikernel
            v_sigmaForDiffusion=[1.2];
            s_numberOfKernels=size(v_sigmaForDiffusion,2);
            s_lambdaForMultiKernels=1;
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
            load('earthquakeTimeSeriesData.mat');
            m_adjacency=m_spatialAdjacency/max(max(m_spatialAdjacency));% normalize adjacency  so that the weights
            m_timeAdjacency=v_propagationWeight*eye(size(m_adjacency));
            % of m_adjacency  and
            % v_propagationWeight are similar.
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph
            
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
            v_bandwidth=[2,4];
            m_sigma=sqrt((1:s_maximumTime)'*v_numberOfSamples*s_mu)';
            %select a subset of measurements
            s_totalTimeSamples=size(m_magnitudesignals,2);
            % data normalization
            v_mean = mean(m_magnitudesignals,2);
            v_std = std(m_magnitudesignals')';
            % 			m_temperatureTimeSeries = diag(1./v_std)*(m_temperatureTimeSeries...
            %                 - v_mean*ones(1,size(m_temperatureTimeSeries,2)));
            
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            s_trainTimePeriod=round(s_pctOfTrainPhase*s_maximumTime);
            
            
            m_magnitudeTimeSeriesSampled=zeros(s_numberOfVertices,s_maximumTime);
            for s_vertInd=1:s_numberOfVertices
                v_magnitudeTimeSeries=m_magnitudesignals(s_vertInd,:);
                v_magnitudeTimeSeriesSampledWhole=...
                    v_magnitudeTimeSeries(1:s_samplePeriod:s_totalTimeSamples);
                m_magnitudeTimeSeriesSampled(s_vertInd,:)=...
                    v_magnitudeTimeSeriesSampledWhole(1:s_maximumTime);
            end
            m_magnitudeTimeSeriesSampled=m_magnitudeTimeSeriesSampled(:,1:s_maximumTime);
            
            
            
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
            t_invSpatialDiffusionKernel=KrKFonGSimulations.createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime);
            
            %% generate transition, correlation matrices
            m_sigma0=zeros(s_numberOfVertices); %TODO choose covariance of initial state.
            m_initialState=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialSigma0=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_ind=1:s_monteCarloSimulations
                t_sigma0(:,:,s_ind)=m_sigma0;
            end
            %KKF part
            [t_correlations,t_transitions]=KrKFonGSimulations.kernelRegressionRecursion...
                (t_invSpatialDiffusionKernel...
                ,-t_timeAdjacencyAtDifferentTimes...
                ,s_maximumTime,s_numberOfVertices,m_sigma0);
            % Correlation matrices for KrKF
            t_spatialCovariance=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            t_obsNoiseCovariace=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            t_spatialDiffusionKernel=KrKFonGSimulations.createDiffusionKernelsFromTopologies(t_spaceAdjacencyAtDifferentTimes,s_sigmaForDiffusion);
            t_spatialCovariance=t_spatialDiffusionKernel;
            % Transition matrix for KrKf
            t_transitionKrKF=...
                repmat(s_transWeight*eye(s_numberOfVertices),[1,1,s_maximumTime]);
            % Kernels for KrKF
            m_combinedKernel=zeros(s_numberOfVertices); % the combined kernel for kriging
            t_dictionaryOfKernels=zeros(s_numberOfKernels,s_numberOfVertices,s_numberOfVertices);
            for s_kernelInd=1:s_numberOfKernels
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusion(s_kernelInd),'m_laplacian',graph.getLaplacian);
                m_difker=diffusionGraphKernel.generateKernelMatrix;
                t_dictionaryOfKernels(s_kernelInd,:,:)=m_difker;
                 m_combinedKernel=m_combinedKernel+squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
            end
            %initialize stateNoise somehow
            
            
            
            m_stateEvolutionKernel=s_stateSigma^2*eye(s_numberOfVertices);
            
            %% 3. generate true signal
            
            m_graphFunction=reshape(m_magnitudeTimeSeriesSampled,[s_maximumTime*s_numberOfVertices,1]);
            
            m_graphFunction=repmat(m_graphFunction,1,s_monteCarloSimulations);
            
            
            %% 4.0 Estimate signal
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_krkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_bandLimitedEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));         
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidth,2));
            t_kRRestimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            
            
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);
                m_obsNoiseCovariance=s_obsSigma^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
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
                %% 4.5 KrKF estimate
                krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
                    't_previousMinimumSquaredError',t_initialSigma0,...
                    'm_previousEstimate',m_initialState);
                l1MultiKernelKrigingCovEstimator=L1MultiKernelKrigingCovEstimator...
                    ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
                % used for parameter estimation
                t_qAuxForSignal=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                t_qAuxForMSE=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
                s_auxInd=1;
                t_residual=zeros(s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
                for s_timeInd=1:s_maximumTime
                    %time t indices
                    v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
                        (s_timeInd)*s_numberOfVertices;
                    v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
                        (s_timeInd)*s_numberOfSamples;
                    %m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
                    m_spatialCovariance=m_combinedKernel;
                    m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
                    
                    %samples and positions at time t
                    m_samplest=m_samples(v_timetIndicesForSamples,:);
                    m_positionst=m_positions(v_timetIndicesForSamples,:);
                    %estimate
                    [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
                        (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),m_stateEvolutionKernel,m_spatialCovariance,m_obsNoiseCovariace);
                    
                    
                    
                    %prepare kf for next iter
                    
                    t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
                        m_estimateKR+m_estimateKF;
                    
                    krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
                    krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
%                     if s_timeInd>1
%                         t_qAuxForSignal(:,:,s_auxInd)=m_estimateKF-m_estimateKFPrev;
%                         t_qAuxForMSE(:,:,:,s_auxInd)=t_MSEKF-t_MSEKFPRev;
%                         s_auxInd=s_auxInd+1;
%                         save residual matrix
%                         for s_monteCarloSimInd=1:s_monteCarloSimulations
%                             t_residual(:,s_monteCarloSimInd,s_auxInd)=m_samplest(:,s_monteCarloSimInd)...
%                                 -m_estimateKF(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd);
%                         end
%                         m_samplespt=m_samples((s_timeInd-2)*s_numberOfSamples+1:...
%                         (s_timeInd-1)*s_numberOfSamples,:);
%                         m_positionspt=m_positions((s_timeInd-2)*s_numberOfSamples+1:...
%                         (s_timeInd-1)*s_numberOfSamples,:);
%                         for s_monteCarloSimInd=1:s_monteCarloSimulations
%                             t_residual(:,s_monteCarloSimInd,s_auxInd)=(m_samplest(:,s_monteCarloSimInd)...
%                                 -m_estimateKR(m_positionst(:,s_monteCarloSimInd),s_monteCarloSimInd))
%                             -(m_samplespt(:,s_monteCarloSimInd)...
%                                 -m_estimateKRPrev(m_positionspt(:,s_monteCarloSimInd),s_monteCarloSimInd));
%                         end
%                         
%                     end
%                     if mod(s_timeInd,s_trainTimePeriod)==0
%                         recalculate t_stateNoiseCorrelation
%                         t_residualCov=KrKFonGSimulations.calculateResidualCov(t_residual,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod);
%                         normalize Cov?
%                         t_residualCov=t_residualCov;
%                         v_theta1=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorCVX(t_residualCov,m_positionst);
%                         v_theta=l1MultiKernelKrigingCovEstimator.estimateCoeffVectorCVX(t_residualCov,m_positionst);
% 
%                         m_combinedKernel=zeros(s_numberOfVertices);
%                         for s_kernelInd=1:s_numberOfKernels
%                             m_combinedKernel=m_combinedKernel+v_theta(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
%                         end
%                         m_stateEvolutionKernel=KrKFonGSimulations.reCalculateStateNoiseCov(t_qAuxForSignal,t_qAuxForMSE,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
%                         s_auxInd=1;
%                     end
%                     
%                     
%                     m_estimateKFPrev=m_estimateKF;
%                     t_MSEKFPRev=t_MSEKF;
%                     m_estimateKRPrev=m_estimateKR;
                end
                %% 5. KF estimate
%                 kFOnGFunctionEstimator=KFOnGFunctionEstimator('s_maximumTime',s_maximumTime,...
%                     't_previousMinimumSquaredError',t_initialSigma0,...
%                     'm_previousEstimate',m_initialState);
%                 for s_timeInd=1:s_maximumTime
%                     time t indices
%                     v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
%                         (s_timeInd)*s_numberOfVertices;
%                     v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
%                         (s_timeInd)*s_numberOfSamples;
%                     
%                     samples and positions at time t
%                     m_samplest=m_samples(v_timetIndicesForSamples,:);
%                     m_positionst=m_positions(v_timetIndicesForSamples,:);
%                     estimate
%                     
%                     [t_kfEstimate(v_timetIndicesForSignals,:,s_sampleInd),t_newMSE]=...
%                         kFOnGFunctionEstimator.oneStepKF(m_samplest,m_positionst,...
%                         t_transitions(:,:,s_timeInd),...
%                         t_correlations(:,:,s_timeInd),m_sigma(s_sampleInd,s_timeInd));
%                     prepare KF for next iteration
%                     kFOnGFunctionEstimator.t_previousMinimumSquaredError=t_newMSE;
%                     kFOnGFunctionEstimator.m_previousEstimate=t_kfEstimate(v_timetIndicesForSignals,:,...
%                         s_sampleInd);
%                     
%                 end
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
                
%                 
%                 for s_bandInd=1:size(v_bandwidth,2)
%                     s_bandwidth=v_bandwidth(s_bandInd);
%                     for s_timeInd=1:s_maximumTime
%                         time t indices
%                         v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
%                             (s_timeInd)*s_numberOfVertices;
%                         v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
%                             (s_timeInd)*s_numberOfSamples;
%                         
%                         samples and positions at time t
%                         
%                         m_samplest=m_samples(v_timetIndicesForSamples,:);
%                         m_positionst=m_positions(v_timetIndicesForSamples,:);
%                         create take diagonals from extended graph
%                         m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
%                         grapht=Graph('m_adjacency',m_adjacency);
%                         
%                         bandlimited estimate
%                         bandlimitedGraphFunctionEstimator= ...
%                             BandlimitedGraphFunctionEstimator('m_laplacian'...
%                             ,grapht.getLaplacian,'s_bandwidth',s_bandwidth);
%                         t_bandLimitedEstimate(v_timetIndicesForSignals,:,s_sampleInd,s_bandInd)=...
%                             bandlimitedGraphFunctionEstimator.estimate(m_samplest,m_positionst);
%                         
%                     end
%                     
%                     
%                 end
%                 
%                 % 8.DistributedFullTrackingAlgorithmEstimator
%                 method from paper A distrubted Tracking Algorithm for Recostruction of Graph Signals
%                 authors Xiaohan Wang, Mengdi Wang, Yuantao Gu
%                 
%                 
%                 for s_bandInd=1:size(v_bandwidth,2)
%                     s_bandwidth=v_bandwidth(s_bandInd);
%                     distributedFullTrackingAlgorithmEstimator=...
%                         DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
%                         's_bandwidth',s_bandwidth,'graph',graph);
%                     t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
%                         distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
%                     
%                     
%                 end
%                 % . LMS
%                 for s_bandInd=1:size(v_bandwidth,2)
%                     s_bandwidth=v_bandwidth(s_bandInd);
%                     m_adjacency=t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd);
%                     grapht=Graph('m_adjacency',m_adjacency);
%                     lMSFullTrackingAlgorithmEstimator=...
%                         LMSFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
%                         's_bandwidth',s_bandwidth,'graph',grapht,'s_stepLMS',s_stepLMS);
%                     t_lmsEstimate(:,:,s_sampleInd,s_bandInd)=...
%                         lMSFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions,m_graphFunction);
%                     
%                     
%                 end
            end
            
            
            %% 9. measure difference
            
            m_relativeErrorDistr=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            m_relativeErrorKf=zeros(s_maximumTime,size(v_numberOfSamples,2));
            m_relativeErrorKrKF=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorKRR=zeros(s_maximumTime,size(v_numberOfSamples,2));
            
            m_relativeErrorLms=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            
            m_relativeErrorbandLimitedEstimate=zeros(s_maximumTime,size(v_numberOfSamples,2)*size(v_bandwidth,2));
            v_allPositions=(1:s_numberOfVertices)';
            for s_sampleInd=1:size(v_numberOfSamples,2)
                v_normOfKFErrors=zeros(s_maximumTime,1);
                v_normOfKrKFErrors=zeros(s_maximumTime,1);
                v_normOfNotSampled=zeros(s_maximumTime,1);
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
                        v_normOfKFErrors(s_timeInd)=v_normOfKFErrors(s_timeInd)+...
                            norm(t_kfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrKFErrors(s_timeInd)=v_normOfKrKFErrors(s_timeInd)+...
                            norm(t_krkfEstimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfKrrErrors(s_timeInd)=v_normOfKrrErrors(s_timeInd)+...
                            norm(t_kRRestimate(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices...
                            ,s_mtind,s_sampleInd)...
                            -m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                        v_normOfNotSampled(s_timeInd)=v_normOfNotSampled(s_timeInd)+...
                            norm(m_graphFunction(v_notSampledPositions+(s_timeInd-1)*s_numberOfVertices,s_mtind),'fro');
                    end
                    
                    s_summedNorm=sum(v_normOfNotSampled(1:s_timeInd));
                    m_relativeErrorKf(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKFErrors(1:s_timeInd))/...
                        s_summedNorm;%s_timeInd*s_numberOfVertices;
                    m_relativeErrorKrKF(s_timeInd, s_sampleInd)...
                        =sum(v_normOfKrKFErrors(1:s_timeInd))/...
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
                    myLegendKF{s_sampleInd}='KKF';
                    myLegendKRR{s_sampleInd}='KRR-IE';
                    myLegendKrKF{s_sampleInd}='KrKKF';
                    
                end
            end
            %normalize errors
            
            myLegend=[myLegendDLSR myLegendLMS myLegendBan myLegendKRR myLegendKF myLegendKrKF ];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_relativeErrorDistr...
                ,m_relativeErrorLms,m_relativeErrorbandLimitedEstimate...
                , m_relativeErrorKRR,m_relativeErrorKf,m_relativeErrorKrKF]',...
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
        
    end
    methods(Static)
        %% generate transformed eigenvalues
        function [m_eigenvectors,m_eigenvaluesAll]=transformedDifEigValues(m_laplacian,v_sigmaForDiffusion)
            [m_eigenvectors,m_eigenvalues] = eig(m_laplacian);%,size(obj.m_laplacian,1));
            v_eigenvalues=diag(m_eigenvalues);
            v_eigenvalues(v_eigenvalues == 0) = eps;
            m_eigenvaluesAll=zeros(size(m_laplacian,1),size(v_sigmaForDiffusion,2));
            for s_ind=1:size(v_sigmaForDiffusion,2);
                m_eigenvaluesAll(:,s_ind)=1./(exp(v_sigmaForDiffusion(s_ind)^2*v_eigenvalues/2));
            end
                
        end
        
        %% calculate residual covariance
        function t_residualCov=calculateResidualCov(t_residual,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod)
            t_auxCovForSignal=zeros(s_numberOfSamples,s_numberOfSamples,s_monteCarloSimulations);
            for s_realizationCounter=1:s_monteCarloSimulations
                m_residual=squeeze(t_residual(:,s_monteCarloSimulations,:));
                v_meanqAuxForSignal=mean(m_residual,2);
                m_meanqAuxForSignal=repmat(v_meanqAuxForSignal,[1,size(m_residual,2)]);
                t_auxCovForSignal(:,:,s_realizationCounter)=(1/s_trainTimePeriod)*...
                    (m_residual-m_meanqAuxForSignal)*(m_residual-m_meanqAuxForSignal)';
            end
            t_residualCov=t_auxCovForSignal;
        end
        function m_residualSpatMean=calculateResidualMean(t_residual,s_numberOfSamples,s_monteCarloSimulations)
            m_auxMeanForSignal=zeros(s_numberOfSamples,s_monteCarloSimulations);
            for s_realizationCounter=1:s_monteCarloSimulations
                m_residual=squeeze(t_residual(:,s_monteCarloSimulations,:));
                m_auxMeanForSignal(:,s_realizationCounter)=mean(m_residual,2);
            end
            m_residualSpatMean=m_auxMeanForSignal;
        
        end
        function [t_residualSpatCov,m_residualSpatMean]=incrementalCalcResCovMean...
                (t_residualSpatCov,t_residualSpat,s_timeInd,m_residualSpatMean)
            for s_realizationCounter=1:size(m_residualSpatMean,2)
                v_vt=(1/s_timeInd)*(t_residualSpat(:,s_realizationCounter)-m_residualSpatMean(:,s_realizationCounter));
                m_residualSpatMean(:,s_realizationCounter)=m_residualSpatMean(:,s_realizationCounter)+v_vt;
                t_residualSpatCov(:,:,s_realizationCounter)=((s_timeInd-2)/(s_timeInd-1))*t_residualSpatCov(:,:,s_realizationCounter)...
                    +s_timeInd*(v_vt*v_vt');
            end
        end
        %% calculate state covariance
        function m_stateNoiseCovariance=reCalculateStateNoiseCov(t_qAuxForSignal,t_qAuxForMSE,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod)
            t_meanqAuxForMSE=mean(t_qAuxForMSE,4);
            t_auxCovForSignal=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_realizationCounter=1:s_monteCarloSimulations
                m_qAuxForSignal=squeeze(t_qAuxForSignal(:,s_monteCarloSimulations,:));
                v_meanqAuxForSignal=mean(m_qAuxForSignal,2);
                m_meanqAuxForSignal=repmat(v_meanqAuxForSignal,[1,s_trainTimePeriod]);
                t_auxCovForSignal(:,:,s_realizationCounter)=(1/s_trainTimePeriod)*...
                    (m_qAuxForSignal-m_meanqAuxForSignal)*(m_qAuxForSignal-m_meanqAuxForSignal)';
            end
            t_stateNoiseCovariance=t_auxCovForSignal+t_meanqAuxForMSE;
            m_stateNoiseCovariance=mean(t_stateNoiseCovariance,3);
        end
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
                t_correlations(:,:,s_maximumTime)=inv(KrKFonGSimulations.makepdagain(t_invSpatialKernel(:,:,s_maximumTime)));
                
                %Recursion
                for s_ind=s_maximumTime:-1:2
                    %Define Ct Dt-1 as in Paper
                    m_Ct=t_invTemporalKernel(:,:,s_ind-1);
                    m_Dtminone=t_invSpatialKernel(:,:,s_ind-1);
                    %Recursion for transitions
                    t_transitions(:,:,s_ind)=-t_correlations(:,:,s_ind)*m_Ct;
                    
                    %Recursion for correlations
                    t_correlations(:,:,s_ind-1)=inv(KrKFonGSimulations.makepdagain(m_Dtminone-m_Ct'*t_correlations(:,:,s_ind)*m_Ct));
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
        function t_spatialDiffusionKernel=createDiffusionKernelsFromTopologies(t_spaceAdjacencyAtDifferentTimes,s_sigmaForDiffusion)
            s_maximumTime=size(t_spaceAdjacencyAtDifferentTimes,3);
            t_spatialDiffusionKernel=zeros(size(t_spaceAdjacencyAtDifferentTimes));
            for s_timeInd=1:s_maximumTime
                graph=Graph('m_adjacency',t_spaceAdjacencyAtDifferentTimes(:,:,s_timeInd));
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusion,'m_laplacian',graph.getLaplacian);
                t_spatialDiffusionKernel(:,:,s_timeInd)=diffusionGraphKernel.generateKernelMatrix;
            end
        end
        function t_invSpatialDiffusionKernel=createinvSpatialKernelSingleDifKer(m_diffusionKernel,m_timeAdjacency,s_maximumTime)
            t_timeAuxMatrix=...
                repmat( diag(sum(m_timeAdjacency+m_timeAdjacency')),[1,1,s_maximumTime]);
            t_timeAuxMatrix(:,:,1)=t_timeAuxMatrix(:,:,1)-diag(sum(m_timeAdjacency));
            t_timeAuxMatrix(:,:,s_maximumTime)=t_timeAuxMatrix(:,:,s_maximumTime)-diag(sum(m_timeAdjacency'));
            t_invSpatialDiffusionKernel=repmat(inv(m_diffusionKernel),[1,1,s_maximumTime]);
            t_invSpatialDiffusionKernel=t_invSpatialDiffusionKernel+t_timeAuxMatrix;
        end
        function A = generateSPDmatrix(n)
            % Generate a dense n x n symmetric, positive definite matrix
            
            A = rand(n,n); % generate a random n x n matrix
            
            % construct a symmetric matrix using either
            A = 0.5*(A+A');% OR A = A*A';
            % The first is significantly faster: O(n^2) compared to O(n^3)
            
            % since A(i,j) < 1 by construction and a symmetric diagonally dominant matrix
            %   is symmetric positive definite, which can be ensured by adding nI
            A = A + n*eye(n);
            
        end
    end
    
end
