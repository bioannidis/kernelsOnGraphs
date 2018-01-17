%
%
%  FIGURES FOR THE PAPER Joint Topology ID Journal
%
%

classdef BlindTopologySignalIDTVSimulations < simFunctionSet
    
    properties
        
    end
    
    methods
        %% Synthetic
        % Data used: Synthetic graph
        % Goal: methods comparison NMSE
        function F = compute_fig_11001(obj,niter)
            
            %%0. define parameters
            s_lambda1 =0.4;
            s_lambda2 = 0.1;
            s_maxIterSEM = 40;
            s_maxIterJoint=10;
            s_maxIterSEMLongRun=40;
            s_rho = 1e4;% rho = 10;
            s_tolSEM = 1e-5;
            b_display= 0;
            s_mu=1e4;
            s_tolJoint=1e-6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf; % no noise real data
            s_layers=200;
            v_bandwidth=10*ones(1,s_layers);
            v_samplepct=0.8*ones(1,s_layers);
            %%1. Simulation
            %create graph
            L=4;                % level of kronecker graph
            s=[1 0.1 0.7;
                0.3 0.1 0.5;
                0 1 0.1];
            graphGenerator = KroneckerGraphGenerator('s_kronNum',L,'m_seedInit',s);
            graph = graphGenerator.realization;
            m_adjacency=graph.m_adjacency;
            %create noisy graph
            s_mean=0;
            s_std=0.1;
            m_randMat = s_mean + s_std.*randn(size(m_adjacency));
            m_randMat=(m_randMat+m_randMat')./2;
            m_randMat=m_randMat-diag(diag(m_randMat));
            m_noisyAdjacency=m_adjacency+m_randMat;
            noisyGraph=Graph('m_adjacency',m_noisyAdjacency);
            s_numberOfVertices=size(m_adjacency,1);
            %create graph function
            m_graphFunction=zeros(s_numberOfVertices,s_layers);
            for s_bandInd=1:s_layers
                functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',v_bandwidth(s_bandInd));
                m_graphFunction(:,s_bandInd)=functionGeneratorBL.realization(1);
            end
            for s_layerInd=1:s_layers
                s_numberOfSamples=round(s_numberOfVertices*v_samplepct(s_layerInd));
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                [c_samples{s_layerInd},...
                    c_positions{s_layerInd}]...
                    = sampler.sample(m_graphFunction(:,s_layerInd));
            end
            %estimate functions
            %sem approach
            sEMGraphGraphFunctionEstimator=SEMGraphGraphFunctionEstimator('s_lambda1',s_lambda1,'s_lambda2',s_lambda2,...
                        's_maxIterSEMInit',s_maxIterSEM,'s_maxIterSEMLongRun',s_maxIterSEMLongRun,...
                        'm_graphFunctionInitialize',m_graphFunctionWithMisses,'s_numberOfVertices',s_numberOfVertices,...
                        's_maxIterJoint',s_maxIterJoint,'s_tolJoint',s_tolJoint,'s_rho',s_rho,'s_tolSEM',s_tolSEM,'s_mu',s_mu);
            %SEMGraphGraphFunctionEstimator('s_lambda1',s_lambda1,'s_lambda2',s_lambda2,'s_maxIterSEMInit',s_maxIterSEM,...
            %    's_numberOfVertices',s_numberOfVertices,'s_maxIterJoint',s_maxIterJoint,'s_tolJoint',s_tolJoint,'s_rho',s_rho,'s_tolSEM',s_tolSEM,'s_mu',s_mu);
            [m_adjacencyEstimate,m_LSEMestimate] =sEMGraphGraphFunctionEstimator.estimate(c_samples,c_positions);
            
            %bl approach
            m_blEstimate=zeros(s_numberOfVertices,s_layers);
            for s_layerInd=1:s_layers
                blestimator = BandlimitedGraphFunctionEstimator('m_laplacian',noisyGraph.getLaplacian,'s_bandwidth',v_bandwidth(s_layerInd));
                m_blEstimate(:,s_layerInd)=blestimator.estimate(c_samples{s_layerInd},c_positions{s_layerInd});
            end
            
            %Multikernel approach
            s_muMulti = 1e-4;
            m_multikernEstimate=zeros(s_numberOfVertices,s_layers);
            sigmaCell = { %sqrt(0.1), sqrt(.15), sqrt(.2), sqrt(.25), sqrt(.3), (linspace(sqrt(0.1), sqrt(.3), 10)),
                sqrt(linspace(0.01, .8, 100)) };
            for i = 1:length(sigmaCell)
                kG = LaplacianKernel('m_laplacian',noisyGraph.getLaplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaCell{i}));
                m_kernel{i} = kG.getKernelMatrix();
            end
            multiKernelEstimator = MkrGraphFunctionEstimator('s_regularizationParameter',s_muMulti, 'ch_type', 'kernel superposition');
            multiKernelEstimator = multiKernelEstimator.replicate('m_kernel', m_kernel, [], {});
            for i = 1 : length(multiKernelEstimator)
                if length(sigmaCell{i}) == 1
                    multiKernelEstimator(i).s_sigma = (sigmaCell{i});
                end
                multiKernelEstimator(i).c_replicatedVerticallyAlong = {'legendString'};
                %estimator = [estimator; est(i).replicate('ch_type', {'RKHS superposition','kernel superposition'}, [], {}) ];
            end
            for s_layerInd=1:s_layers
                m_multikernEstimate(:,s_layerInd)=multiKernelEstimator.estimate(c_samples{s_layerInd},c_positions{s_layerInd});
            end
            %%2. Compare error
            s_NMSE=BlindTopologySignalIDSimulations.estimateNormalizedMeanSquaredError(m_LSEMestimate,m_graphFunction)
            %s_NMSEUn=BlindTopologySignalIDSimulations.estimateNormalizedMeanSquaredErrorUnsampled(m_graphFunctionEstimate,m_graphFunction,c_positions)
            s_NMSEBL=BlindTopologySignalIDSimulations.estimateNormalizedMeanSquaredError(m_blEstimate,m_graphFunction)
            s_NMSEMultiKernel=BlindTopologySignalIDSimulations.estimateNormalizedMeanSquaredError(m_multikernEstimate,m_graphFunction)
            
            figure(2); imagesc(m_adjacencyEstimate);
            
            F = F_figure('X',v_sampleSetSize,'Y',m_meanSquaredError','xlab','Number of observed vertices (S)','ylab','NMSE','leg',{strcat('Bandlimited  ',sprintf(' W=%g',s_bandwidth1)),strcat('Bandlimited',sprintf(' W=%g',s_bandwidth2)),'Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)'});
            
        end
        % Data used: Synthetic graph
        % Goal: methods comparison NMSE vary sampling size
        function F = compute_fig_11002(obj,niter)
            
            %%0. define parameters
            s_lambda1 =0.5;
            s_lambda2 = 0.1;
            s_maxIterSEM = 40;
            s_maxIterJoint=10;
            s_rho = 1e4;% rho = 10;
            s_tolSEM = 1e-5;
            b_display= 0;
            s_mu=1e4;
            s_tolJoint=1e-6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf; % no noise real data
            s_layers=100;
            s_bandwidth=10;
            s_bandwidth1=s_bandwidth+2;
            v_bandwidth=s_bandwidth*ones(1,s_layers);
            v_samplepct=0.4:0.05:1;%ones(1,s_layers);
            s_muMulti = 1e-9;
            
            s_differentSampleSetSizenum=size(v_samplepct,2);
            s_kernelnum=100;
            v_sigma=linspace(0.01, 2, s_kernelnum);
            %%1. Simulation
            %create graph
            L=4;                % level of kronecker graph
            s=[1 0.1 0.7;
                0.3 0.1 0.5;
                0 1 0.1];
            graphGenerator = KroneckerGraphGenerator('s_kronNum',L,'m_seedInit',s);
            graph = graphGenerator.realization;
            m_adjacency=graph.m_adjacency;
            
            %support is known exactly so only noise in the true edges
            %m_noisyAdjacency=m_adjacency+m_randMat.*m_adjacency;
            %support not known noise creates new edges.
            s_numberOfVertices=size(m_adjacency,1);
            
            %create graph function
            m_graphFunction=zeros(s_numberOfVertices,s_layers);
            for s_bandInd=1:s_layers
                functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',v_bandwidth(s_bandInd));
                m_graphFunction(:,s_bandInd)=functionGeneratorBL.realization(1);
            end
            
            %different montecarlo iterations
            m_NMSE=zeros(s_differentSampleSetSizenum,s_monteCarloSimulations);
            m_NMSEBL1=zeros(s_differentSampleSetSizenum,s_monteCarloSimulations);
            m_NMSEBL2=zeros(s_differentSampleSetSizenum,s_monteCarloSimulations);
            m_NMSEMultiKernel=zeros(s_differentSampleSetSizenum,s_monteCarloSimulations);
            m_NMSEMultiKernelTrue=zeros(s_differentSampleSetSizenum,s_monteCarloSimulations);
            
            for s_monteCarloSimulationsInd=1:s_monteCarloSimulations
                v_NMSE=zeros(s_differentSampleSetSizenum,1);
                v_NMSEBL1=zeros(s_differentSampleSetSizenum,1);
                v_NMSEBL2=zeros(s_differentSampleSetSizenum,1);
                v_NMSEMultiKernel=zeros(s_differentSampleSetSizenum,1);
                v_NMSEMultiKernelTrue=zeros(s_differentSampleSetSizenum,1);
                %create noisy graph
                s_mean=0;
                s_std=0.23;
                m_randMat = s_mean + s_std.*randn(size(m_adjacency));
                m_randMat=(m_randMat+m_randMat')./2;
                m_randMat=m_randMat-diag(diag(m_randMat));
                m_noisyAdjacency=m_adjacency+abs(m_randMat);
                noisyGraph=Graph('m_adjacency',m_noisyAdjacency);
                %sample differnt sizes
                
                for s_sampleSizeInd=1:s_differentSampleSetSizenum
                    v_samplepctpersize=v_samplepct(s_sampleSizeInd)*ones(1,s_layers);
                    for s_layerInd=1:s_layers
                        s_numberOfSamples=round(s_numberOfVertices*v_samplepctpersize(s_layerInd));
                        sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                        [c_samples{s_layerInd},...
                            c_positions{s_layerInd}]...
                            = sampler.sample(m_graphFunction(:,s_layerInd));
                    end
                    %estimate functions
                    %sem approach
                    toc
                    sEMGraphGraphFunctionEstimator=SEMGraphGraphFunctionEstimator('s_lambda1',s_lambda1,'s_lambda2',s_lambda2,'s_maxIterSEMInit',s_maxIterSEM,...
                        's_numberOfVertices',s_numberOfVertices,'s_maxIterJoint',s_maxIterJoint,'s_tolJoint',s_tolJoint,'s_rho',s_rho,'s_tolSEM',s_tolSEM,'s_mu',s_mu);
                    tic
                    [m_adjacencyEstimate,m_LSEMestimate] =sEMGraphGraphFunctionEstimator.estimate(c_samples,c_positions);
                    timeLSEM=toc
                    legendSEM='LSEM';
                    %bl approach
                    m_blEstimate1=zeros(s_numberOfVertices,s_layers);
                    for s_layerInd=1:s_layers
                        blestimator = BandlimitedGraphFunctionEstimator('m_laplacian',noisyGraph.getLaplacian,'s_bandwidth',s_bandwidth);
                        m_blEstimate1(:,s_layerInd)=blestimator.estimate(c_samples{s_layerInd},c_positions{s_layerInd});
                    end
                    legendBL{1}=strcat('BL B=',num2str(s_bandwidth));
                    m_blEstimate2=zeros(s_numberOfVertices,s_layers);
                    for s_layerInd=1:s_layers
                        blestimator = BandlimitedGraphFunctionEstimator('m_laplacian',noisyGraph.getLaplacian,'s_bandwidth',s_bandwidth1);
                        m_blEstimate2(:,s_layerInd)=blestimator.estimate(c_samples{s_layerInd},c_positions{s_layerInd});
                    end
                    legendBL{2}=strcat('BL B=',num2str(s_bandwidth1));
                    %Multikernel approach
                    tic
                    m_multikernEstimate=zeros(s_numberOfVertices,s_layers);
                    sigmaCell = { %sqrt(0.1), sqrt(.15), sqrt(.2), sqrt(.25), sqrt(.3), (linspace(sqrt(0.1), sqrt(.3), 10)),
                        sqrt(v_sigma) };
                    for i = 1:length(sigmaCell)
                        kG = LaplacianKernel('m_laplacian',noisyGraph.getLaplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaCell{i}));
                        m_kernel{i} = kG.getKernelMatrix();
                    end
                    legendMK='Multi-kernel';
                    multiKernelEstimator = MkrGraphFunctionEstimator('s_regularizationParameter',s_muMulti, 'ch_type', 'kernel superposition');
                    multiKernelEstimator = multiKernelEstimator.replicate('m_kernel', m_kernel, [], {});
                    for i = 1 : length(multiKernelEstimator)
                        if length(sigmaCell{i}) == 1
                            multiKernelEstimator(i).s_sigma = (sigmaCell{i});
                        end
                        multiKernelEstimator(i).c_replicatedVerticallyAlong = {'legendString'};
                        %estimator = [estimator; est(i).replicate('ch_type', {'RKHS superposition','kernel superposition'}, [], {}) ];
                    end
                    for s_layerInd=1:s_layers
                        m_multikernEstimate(:,s_layerInd)=multiKernelEstimator.estimate(c_samples{s_layerInd},c_positions{s_layerInd});
                    end
                    timeMKL=toc
                    m_multikernTrueEstimate=zeros(s_numberOfVertices,s_layers);
                    sigmaCell = { %sqrt(0.1), sqrt(.15), sqrt(.2), sqrt(.25), sqrt(.3), (linspace(sqrt(0.1), sqrt(.3), 10)),
                        sqrt(v_sigma) };
                    for i = 1:length(sigmaCell)
                        kG = LaplacianKernel('m_laplacian',graph.getLaplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaCell{i}));
                        m_kernel{i} = kG.getKernelMatrix();
                    end
                    legendBas='Baseline';
                    multiKerneltrueEstimator = MkrGraphFunctionEstimator('s_regularizationParameter',s_muMulti, 'ch_type', 'kernel superposition');
                    multiKerneltrueEstimator = multiKerneltrueEstimator.replicate('m_kernel', m_kernel, [], {});
                    for i = 1 : length(multiKerneltrueEstimator)
                        if length(sigmaCell{i}) == 1
                            v(i).s_sigma = (sigmaCell{i});
                        end
                        multiKerneltrueEstimator(i).c_replicatedVerticallyAlong = {'legendString'};
                        %estimator = [estimator; est(i).replicate('ch_type', {'RKHS superposition','kernel superposition'}, [], {}) ];
                    end
                    for s_layerInd=1:s_layers
                        m_multikernTrueEstimate(:,s_layerInd)=multiKerneltrueEstimator.estimate(c_samples{s_layerInd},c_positions{s_layerInd});
                    end
                    %%2. Compare error
                    v_NMSE(s_sampleSizeInd)=BlindTopologySignalIDSimulations.estimateNormalizedMeanSquaredError(m_LSEMestimate,m_graphFunction);
                    %s_NMSEUn=BlindTopologySignalIDSimulations.estimateNormalizedMeanSquaredErrorUnsampled(m_graphFunctionEstimate,m_graphFunction,c_positions)
                    v_NMSEBL1(s_sampleSizeInd)=BlindTopologySignalIDSimulations.estimateNormalizedMeanSquaredError(m_blEstimate1,m_graphFunction);
                    v_NMSEBL2(s_sampleSizeInd)=BlindTopologySignalIDSimulations.estimateNormalizedMeanSquaredError(m_blEstimate2,m_graphFunction);
                    v_NMSEMultiKernel(s_sampleSizeInd)=BlindTopologySignalIDSimulations.estimateNormalizedMeanSquaredError(m_multikernEstimate,m_graphFunction);
                    v_NMSEMultiKernelTrue(s_sampleSizeInd)=BlindTopologySignalIDSimulations.estimateNormalizedMeanSquaredError(m_multikernTrueEstimate,m_graphFunction);
                end
                m_NMSE(:,s_monteCarloSimulationsInd)=v_NMSE;
                m_NMSEBL1(:,s_monteCarloSimulationsInd)=v_NMSEBL1;
                m_NMSEBL2(:,s_monteCarloSimulationsInd)=v_NMSEBL2;
                m_NMSEMultiKernel(:,s_monteCarloSimulationsInd)=v_NMSEMultiKernel;
                m_NMSEMultiKernelTrue(:,s_monteCarloSimulationsInd)=v_NMSEMultiKernelTrue;
                
            end
            v_NMSE=mean(m_NMSE,2);
            v_NMSEBL1=mean(m_NMSEBL1,2);
            v_NMSEBL2=mean(m_NMSEBL2,2);
            v_NMSEMultiKernel=mean(m_NMSEMultiKernel,2);
            v_NMSEMultiKernelTrue=mean(m_NMSEMultiKernelTrue,2);
            
            m_meanSquaredError=[v_NMSEBL1,v_NMSEBL2,v_NMSEMultiKernel,v_NMSE,v_NMSEMultiKernelTrue]';
            legend=[legendBL,legendMK,legendSEM,legendBas]';
            figure(2); imagesc(m_adjacencyEstimate);
            F = F_figure('X',v_samplepct*s_numberOfVertices,'Y',m_meanSquaredError,'xlab','sample size (M)','ylab','NMSE','leg',legend);
            %F.logy=1;
        end
        % Data used: Synthetic graph
        % Goal: compare graph reconstruction
        function F = compute_fig_11003(obj,niter)
            
            %%0. define parameters
             %s_lambda1 =0.5;
            %s_lambda2 = 0.1;
            %s_mu=1e4;
            s_lambda1 =10^2;
            s_lambda2 = 10^0;
            s_maxIterSEMInit = 40;
            s_maxIterJoint=40;
            s_rho = 1e4;% rho = 10;
            s_tolSEM = 1e-5;
            b_display= 0;
            s_mu=10^10;
            s_tolJoint=1e-6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf; % no noise real data
            s_layers=100;
            s_bandwidth=10;
            s_bandwidth1=s_bandwidth+2;
            s_maxIterSEMLongRun=40;
            v_bandwidth=s_bandwidth*ones(1,s_layers);
            v_samplepct=0.6:0.05:1;
            v_thressholds=10.^(-3:0.1:-0.5);%0.01:0.02:0.6;
            s_differentSampleSetSizenum=size(v_samplepct,2);
            %%1. Simulation
            %create graph
            L=4;                % level of kronecker graph
            s=[0.6 0.1 0.7;
                0.3 0.1 0.5;
                0 1 0.1];
          
            
            %different montecarlo iterations
            m_NMSE=zeros(s_differentSampleSetSizenum,s_monteCarloSimulations);
            m_NMSELSEM=zeros(s_differentSampleSetSizenum,s_monteCarloSimulations);
            m_edgePredErrorLSEM=zeros(s_differentSampleSetSizenum,s_monteCarloSimulations);
            m_edgePredError=zeros(s_differentSampleSetSizenum,s_monteCarloSimulations);
            
            for s_monteCarloSimulationsInd=1:s_monteCarloSimulations
                graphGenerator = KroneckerGraphGenerator('s_kronNum',L,'m_seedInit',s);
                graph = graphGenerator.realization;
                m_adjacency=graph.m_adjacency;
                %create noisy graph
                %support is known exactly so only noise in the true edges
                %m_noisyAdjacency=m_adjacency+m_randMat.*m_adjacency;
                %support not known noise creates new edges.
                s_numberOfVertices=size(m_adjacency,1);
                colormap('gray');
                %create graph function
                m_graphFunction=zeros(s_numberOfVertices,s_layers);
                for s_bandInd=1:s_layers
                    functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',v_bandwidth(s_bandInd));
                    m_graphFunction(:,s_bandInd)=functionGeneratorBL.realization(1);
                end
                v_NMSE=zeros(s_differentSampleSetSizenum,1);
                v_NMSELSEM=zeros(s_differentSampleSetSizenum,1);
                v_edgePredError=zeros(s_differentSampleSetSizenum,1);
                v_edgePredErrorLSEM=zeros(s_differentSampleSetSizenum,1);
                %sample differnt sizes
                
                for s_sampleSizeInd=1:s_differentSampleSetSizenum
                    v_samplepctpersize=v_samplepct(s_sampleSizeInd)*ones(1,s_layers);
                    m_graphFunctionWithMisses=zeros(size(m_graphFunction));
                    for s_layerInd=1:s_layers
                        s_numberOfSamples=round(s_numberOfVertices*v_samplepctpersize(s_layerInd));
                        sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                        [c_samples{s_layerInd},...
                            c_positions{s_layerInd}]...
                            = sampler.sample(m_graphFunction(:,s_layerInd));
                        m_S=BlindTopologySignalIDSimulations.createSamplingMatrix(c_positions{s_layerInd},s_numberOfVertices);
                        m_graphFunctionWithMisses(:,s_layerInd)=m_S'*c_samples{s_layerInd};
                    end
                    %estimate functions
                    %sem approach
                    sEMGraphGraphFunctionEstimator=SEMGraphGraphFunctionEstimator('s_lambda1',s_lambda1,'s_lambda2',s_lambda2,...
                        's_maxIterSEMInit',s_maxIterSEMInit,'s_maxIterSEMLongRun',s_maxIterSEMLongRun,...
                        'm_graphFunctionInitialize',m_graphFunctionWithMisses,'s_numberOfVertices',s_numberOfVertices,...
                        's_maxIterJoint',s_maxIterJoint,'s_tolJoint',s_tolJoint,'s_rho',s_rho,'s_tolSEM',s_tolSEM,'s_mu',s_mu);
                    [m_adjacencyEstimateJISG,m_LSEMestimate] =sEMGraphGraphFunctionEstimator.estimate(c_samples,c_positions);
                    legendJISG={'JISG'};
                    lSEMGraphEstimator=LSEMGraphEstimator('s_lambda1',s_lambda1,'s_lambda2',s_lambda2,'s_maxIter',30,'s_rho',s_rho,'s_tol',s_tolSEM);
                    m_adjacencyEstimatSEM=lSEMGraphEstimator.estimate(m_graphFunction);
                    legendSEM={'LSEM'};
                    colormap('gray');
                    m_adjacencyEstimateJISG=m_adjacencyEstimateJISG./max(max(m_adjacencyEstimateJISG));
                    figure(2); imagesc(m_adjacencyEstimateJISG);
                    
                    m_adjacencyEstimatSEM=m_adjacencyEstimatSEM./max(max(m_adjacencyEstimatSEM));
%                     colormap('gray');
%                     figure(3); imagesc(m_adjacencyEstimatSEM);
                    [m_adjacencyEstimate,s_edgePredEr]=BlindTopologySignalIDSimulations.reconstructAdj(m_adjacencyEstimateJISG,m_adjacency,v_thressholds);
                    
                    [As,s_edgePredErLSEM]=BlindTopologySignalIDSimulations.reconstructAdj(m_adjacencyEstimatSEM,m_adjacency,v_thressholds);
                    %%2. Compare error
                    v_edgePredError(s_sampleSizeInd)=s_edgePredEr;
                    v_edgePredErrorLSEM(s_sampleSizeInd)=s_edgePredErLSEM;
                    v_NMSE(s_sampleSizeInd)=BlindTopologySignalIDSimulations.measureMSE(m_adjacencyEstimateJISG,m_adjacency);
                    v_NMSELSEM(s_sampleSizeInd)=BlindTopologySignalIDSimulations.measureMSE(m_adjacencyEstimatSEM,m_adjacency);
                end
                m_NMSE(:,s_monteCarloSimulationsInd)=v_NMSE;
                m_NMSELSEM(:,s_monteCarloSimulationsInd)=v_NMSELSEM;
                m_edgePredError(:,s_monteCarloSimulationsInd)=v_edgePredError;
                m_edgePredErrorLSEM(:,s_monteCarloSimulationsInd)=v_edgePredErrorLSEM;
                
            end
            v_NMSE=mean(m_NMSE,2);
            v_NMSELSEM=mean(m_NMSELSEM,2);
            v_edgePredError=mean(m_edgePredError,2).*(100/(s_numberOfVertices*(s_numberOfVertices-1)));
            v_edgePredErrorLSEM=mean(m_edgePredErrorLSEM,2).*(100/(s_numberOfVertices*(s_numberOfVertices-1)));
            m_meanSquaredError=[v_edgePredErrorLSEM,v_edgePredError]';
            save('libGF/simulations/BlindTopologySignalIDSimulations_data/results');
            
            legend=[legendSEM,legendJISG]';
            figure(4); imagesc(m_adjacencyEstimate);
            F=F_figure('X',v_samplepct*s_numberOfVertices,'Y',m_meanSquaredError,'xlab','sample size (M)','ylab','EIER','leg',legend);
            %sF.logy=1;
        end
        % Data used: Synthetic graph BL signal
        % Goal: optimize parameters
        function F = compute_fig_11004(obj,niter)
            
            %%0. define parameters
            v_lambda1 =10.^(-8:0.5:-2);
            v_lambda2 = 10.^(-9:0.5:-2);
            s_maxIterSEM = 40;
            s_maxIterJoint=40;
            s_rho = 1e3;% rho = 10;
            v_rho=10.^4%;(3:1:5);
            s_tolSEM = 1e-4;
            b_display= 0;
            s_mu=10;
            s_tolJoint=1e-6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf; % no noise real data
            s_layers=100;
            s_bandwidth=10;
            s_bandwidth1=s_bandwidth+2;
            v_bandwidth=s_bandwidth*ones(1,s_layers);
            v_samplepct=0.8%:0.1:1;
            v_thressholds=10.^(-3:0.1:-0.5);%0.01:0.02:0.6;
            s_differentSampleSetSizenum=size(v_samplepct,2);
            %%1. Simulation
            %create graph
            L=4;                % level of kronecker graph
            s=[1 0.1 0.7;
                0.3 0.1 0.5;
                0 1 0.1];
            graphGenerator = KroneckerGraphGenerator('s_kronNum',L,'m_seedInit',s);
            graph = graphGenerator.realization;
            m_adjacency=graph.m_adjacency;
            %create noisy graph
            %support is known exactly so only noise in the true edges
            %m_noisyAdjacency=m_adjacency+m_randMat.*m_adjacency;
            %support not known noise creates new edges.
            s_numberOfVertices=size(m_adjacency,1);
            
            %create graph function
            m_graphFunction=zeros(s_numberOfVertices,s_layers);
            for s_bandInd=1:s_layers
                functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',v_bandwidth(s_bandInd));
                m_graphFunction(:,s_bandInd)=functionGeneratorBL.realization(1);
            end
            
            %different montecarlo iterations
            s_it=1;
            v_NMSEpr=zeros(s_differentSampleSetSizenum,size(v_lambda1,2),size(v_lambda2,2),size(v_rho,2));
            v_edgePredErrorpr=zeros(s_differentSampleSetSizenum,size(v_lambda1,2),size(v_lambda2,2),size(v_rho,2));
            for s_monteCarloSimulationsInd=1:s_monteCarloSimulations
                v_NMSE=zeros(s_differentSampleSetSizenum,size(v_lambda1,2),size(v_lambda2,2),size(v_rho,2));
                v_edgePredError=zeros(s_differentSampleSetSizenum,size(v_lambda1,2),size(v_lambda2,2),size(v_rho,2));
                %sample differnt sizes
                for s_lambda1Ind=1:size(v_lambda1,2)
                    s_lambda1=v_lambda1(s_lambda1Ind);
                    for s_lambda2Ind=1:size(v_lambda2,2)
                        s_lambda2=v_lambda2(s_lambda2Ind);
                        for s_rhoInd=1:size(v_rho,2)
                            s_rho=v_rho(s_rhoInd);
                        for s_sampleSizeInd=1:s_differentSampleSetSizenum
                            v_samplepctpersize=v_samplepct(s_sampleSizeInd)*ones(1,s_layers);
                            m_graphFunctionWithMisses=zeros(size(m_graphFunction));
                            for s_layerInd=1:s_layers
                                s_numberOfSamples=round(s_numberOfVertices*v_samplepctpersize(s_layerInd));
                                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                                [c_samples{s_layerInd},...
                                    c_positions{s_layerInd}]...
                                    = sampler.sample(m_graphFunction(:,s_layerInd));
                                m_S=BlindTopologySignalIDSimulations.createSamplingMatrix(c_positions{s_layerInd},s_numberOfVertices);
                                m_graphFunctionWithMisses(:,s_layerInd)=m_S'*c_samples{s_layerInd};
                            end
                            %estimate functions
                            %sem approach
                            sEMGraphGraphFunctionEstimator=SEMGraphGraphFunctionEstimator('s_lambda1',s_lambda1,'s_lambda2',s_lambda2,'s_maxIterSEM',s_maxIterSEM,...
                                's_numberOfVertices',s_numberOfVertices,'s_maxIterJoint',s_maxIterJoint,'s_tolJoint',s_tolJoint,'s_rho',s_rho,'s_tolSEM',s_tolSEM,'s_mu',s_mu);
                            [m_adjacencyEstimateJISG,m_LSEMestimate] =sEMGraphGraphFunctionEstimator.estimate(c_samples,c_positions);
                            legendJISG={'JISG'};
                            
                            m_adjacencyEstimateJISG=m_adjacencyEstimateJISG./max(max(m_adjacencyEstimateJISG));
                            figure(2); imagesc(m_adjacencyEstimateJISG); 
                            [As,s_edgePredEr]=BlindTopologySignalIDSimulations.reconstructAdj(m_adjacencyEstimateJISG,m_adjacency,v_thressholds);
                            %%2. Compare error
                            v_error{s_it,1}={v_samplepct(s_sampleSizeInd),s_lambda1,s_lambda2,s_rho};
                            v_error{s_it,2}=s_edgePredEr;
                            v_error{s_it,3}=BlindTopologySignalIDSimulations.measureMSE(m_adjacencyEstimateJISG,m_adjacency);
                            s_it=s_it+1;
                            end
                        end
                    end
                end
            
            end
%             m_meanSquaredError=[v_NMSELSEM,v_NMSE]';
%             legend=[legendSEM,legendJISG]';
            save('partuning');
%             figure(2); imagesc(m_adjacencyEstimate);
            F = [];%F_figure('X',v_samplepct*s_numberOfVertices,'Y',m_meanSquaredError,'xlab','sample size (M)','ylab','NMSE','leg',legend);
            %F.logy=1;
        end
        % Data used: Synthetic graph SEM signal
        % Goal: optimize parameters
        function F = compute_fig_11005(obj,niter)
            
            %%0. define parameters
            v_lambda1 =10.^(-9:0.2:-7);
            v_lambda2 = 10.^(-9:0.2:-7);
            s_maxIterSEM = 40;
            s_maxIterJoint=40;
            s_rho = 1e4;% rho = 10;
            s_tolSEM = 1e-4;
            b_display= 0;
            v_mu=10;%(0.1:0.5:6);
            s_tolJoint=1e-6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf; % no noise real data
            s_layers=100;
            s_bandwidth=10;
            s_bandwidth1=s_bandwidth+2;
            v_bandwidth=s_bandwidth*ones(1,s_layers);
            v_samplepct=0.8%:0.1:1;
            v_thressholds=10.^(-3:0.1:-0.5);%0.01:0.02:0.6;
            s_var=0.01;
            s_differentSampleSetSizenum=size(v_samplepct,2);
            %%1. Simulation
            %create graph
            L=4;                % level of kronecker graph
            s=[1 0.1 0.7;
                0.3 0.1 0.5;
                0 1 0.1];
            
            graphGenerator = KroneckerGraphGenerator('s_kronNum',L,'m_seedInit',s);
            graph = graphGenerator.realization;
            m_adjacency=graph.m_adjacency;
            %create noisy graph
            %support is known exactly so only noise in the true edges
            %m_noisyAdjacency=m_adjacency+m_randMat.*m_adjacency;
            %support not known noise creates new edges.
            s_numberOfVertices=size(m_adjacency,1);
            
            %create graph function
            m_graphFunction=zeros(s_numberOfVertices,s_layers);
            for s_bandInd=1:s_layers
                functionGeneratorBL = SEMGraphFunctionGenerator('graph',graph,'s_var',s_var);
                m_graphFunction(:,s_bandInd)=functionGeneratorBL.realization(1);
            end
            
            %different montecarlo iterations
            s_it=1;
            v_NMSEpr=zeros(s_differentSampleSetSizenum,size(v_lambda1,2),size(v_lambda2,2),size(v_mu,2));
            v_edgePredErrorpr=zeros(s_differentSampleSetSizenum,size(v_lambda1,2),size(v_lambda2,2),size(v_mu,2));
            for s_monteCarloSimulationsInd=1:s_monteCarloSimulations
                v_NMSE=zeros(s_differentSampleSetSizenum,size(v_lambda1,2),size(v_lambda2,2),size(v_mu,2));
                v_edgePredError=zeros(s_differentSampleSetSizenum,size(v_lambda1,2),size(v_lambda2,2),size(v_mu,2));
                %sample differnt sizes
                for s_lambda1Ind=1:size(v_lambda1,2)
                    s_lambda1=v_lambda1(s_lambda1Ind);
                    for s_lambda2Ind=1:size(v_lambda2,2)
                        s_lambda2=v_lambda2(s_lambda2Ind);
                        for s_muInd=1:size(v_mu,2)
                            s_mu=v_mu(s_muInd);
                        for s_sampleSizeInd=1:s_differentSampleSetSizenum
                            v_samplepctpersize=v_samplepct(s_sampleSizeInd)*ones(1,s_layers);
                            m_graphFunctionWithMisses=zeros(size(m_graphFunction));
                            for s_layerInd=1:s_layers
                                s_numberOfSamples=round(s_numberOfVertices*v_samplepctpersize(s_layerInd));
                                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                                [c_samples{s_layerInd},...
                                    c_positions{s_layerInd}]...
                                    = sampler.sample(m_graphFunction(:,s_layerInd));
                                m_S=BlindTopologySignalIDSimulations.createSamplingMatrix(c_positions{s_layerInd},s_numberOfVertices);
                                m_graphFunctionWithMisses(:,s_layerInd)=m_S'*c_samples{s_layerInd};
                            end
                            %estimate functions
                            %sem approach
                            sEMGraphGraphFunctionEstimator=SEMGraphGraphFunctionEstimator('s_lambda1',s_lambda1,'s_lambda2',s_lambda2,'s_maxIterSEM',s_maxIterSEM,...
                                's_numberOfVertices',s_numberOfVertices,'s_maxIterJoint',s_maxIterJoint,'s_tolJoint',s_tolJoint,'s_rho',s_rho,'s_tolSEM',s_tolSEM,'s_mu',s_mu);
                            [m_adjacencyEstimateJISG,m_LSEMestimate] =sEMGraphGraphFunctionEstimator.estimate(c_samples,c_positions);
                            legendJISG={'JISG'};
                            m_adjacencyEstimateJISG=m_adjacencyEstimateJISG./max(max(m_adjacencyEstimateJISG));
                            figure(2); imagesc(m_adjacencyEstimateJISG);
                            

                            [As,s_edgePredEr]=BlindTopologySignalIDSimulations.reconstructAdj(m_adjacencyEstimateJISG,m_adjacency,v_thressholds);
                            %%2. Compare error
                            v_error{s_it,1}={v_samplepct(s_sampleSizeInd),s_lambda1,s_lambda2,s_mu};
                            v_error{s_it,2}=s_edgePredEr;
                            v_error{s_it,3}=BlindTopologySignalIDSimulations.measureMSE(m_adjacencyEstimateJISG,m_adjacency);
                            s_it=s_it+1;
                            end
                        end
                    end
                end
            
            end
%             m_meanSquaredError=[v_NMSELSEM,v_NMSE]';
%             legend=[legendSEM,legendJISG]';
            save('partuning');
             figure(2); imagesc(m_adjacencyEstimateJISG);
            F = [];%F_figure('X',v_samplepct*s_numberOfVertices,'Y',m_meanSquaredError,'xlab','sample size (M)','ylab','NMSE','leg',legend);
            %F.logy=1;
        end
        %% Real
        % Data used: Real gene signal
        % Goal: optimize parameters
        function F = compute_fig_11006(obj,niter)
            
            %%0. define parameters
            v_lambda1 =10.^(-6:1:8);
            v_lambda2 = 10.^(-6:1:8);
            s_maxIterSEM = 40;
            s_maxIterJoint=40;
            s_rho = 1e4;% rho = 10;
            s_tolSEM = 1e-4;
            b_display= 0;
            v_mu=10;%(0.1:0.5:6);
            s_tolJoint=1e-6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf; % no noise real data
            s_layers=100;
            s_bandwidth=10;
            s_bandwidth1=s_bandwidth+2;
            v_bandwidth=s_bandwidth*ones(1,s_layers);
            v_samplepct=0.8%:0.1:1;
            v_thressholds=10.^(-3:0.1:-0.5);%0.01:0.02:0.6;
            s_var=0.01;
            s_differentSampleSetSizenum=size(v_samplepct,2);
            %%1. Simulation
            %create graph
            L=4;                % level of kronecker graph
            s=[1 0.1 0.7;
                0.3 0.1 0.5;
                0 1 0.1];
            
            graphGenerator = KroneckerGraphGenerator('s_kronNum',L,'m_seedInit',s);
            graph = graphGenerator.realization;
            m_adjacency=graph.m_adjacency;
            %create noisy graph
            %support is known exactly so only noise in the true edges
            %m_noisyAdjacency=m_adjacency+m_randMat.*m_adjacency;
            %support not known noise creates new edges.
            s_numberOfVertices=size(m_adjacency,1);
            
            %create graph function
            m_graphFunction=zeros(s_numberOfVertices,s_layers);
            for s_bandInd=1:s_layers
                functionGeneratorBL = SEMGraphFunctionGenerator('graph',graph,'s_var',s_var);
                m_graphFunction(:,s_bandInd)=functionGeneratorBL.realization(1);
            end
            
            %different montecarlo iterations
            s_it=1;
            v_NMSEpr=zeros(s_differentSampleSetSizenum,size(v_lambda1,2),size(v_lambda2,2),size(v_mu,2));
            v_edgePredErrorpr=zeros(s_differentSampleSetSizenum,size(v_lambda1,2),size(v_lambda2,2),size(v_mu,2));
            for s_monteCarloSimulationsInd=1:s_monteCarloSimulations
                v_NMSE=zeros(s_differentSampleSetSizenum,size(v_lambda1,2),size(v_lambda2,2),size(v_mu,2));
                v_edgePredError=zeros(s_differentSampleSetSizenum,size(v_lambda1,2),size(v_lambda2,2),size(v_mu,2));
                %sample differnt sizes
                for s_lambda1Ind=1:size(v_lambda1,2)
                    s_lambda1=v_lambda1(s_lambda1Ind);
                    for s_lambda2Ind=1:size(v_lambda2,2)
                        s_lambda2=v_lambda2(s_lambda2Ind);
                        for s_muInd=1:size(v_mu,2)
                            s_mu=v_mu(s_muInd);
                        for s_sampleSizeInd=1:s_differentSampleSetSizenum
                            v_samplepctpersize=v_samplepct(s_sampleSizeInd)*ones(1,s_layers);
                            m_graphFunctionWithMisses=zeros(size(m_graphFunction));
                            for s_layerInd=1:s_layers
                                s_numberOfSamples=round(s_numberOfVertices*v_samplepctpersize(s_layerInd));
                                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                                [c_samples{s_layerInd},...
                                    c_positions{s_layerInd}]...
                                    = sampler.sample(m_graphFunction(:,s_layerInd));
                                m_S=BlindTopologySignalIDSimulations.createSamplingMatrix(c_positions{s_layerInd},s_numberOfVertices);
                                m_graphFunctionWithMisses(:,s_layerInd)=m_S'*c_samples{s_layerInd};
                            end
                            %estimate functions
                            %sem approach
                            sEMGraphGraphFunctionEstimator=SEMGraphGraphFunctionEstimator('s_lambda1',s_lambda1,'s_lambda2',s_lambda2,'s_maxIterSEM',s_maxIterSEM,...
                                's_numberOfVertices',s_numberOfVertices,'s_maxIterJoint',s_maxIterJoint,'s_tolJoint',s_tolJoint,'s_rho',s_rho,'s_tolSEM',s_tolSEM,'s_mu',s_mu);
                            [m_adjacencyEstimateJISG,m_LSEMestimate] =sEMGraphGraphFunctionEstimator.estimate(c_samples,c_positions);
                            legendJISG={'JISG'};
                            m_adjacencyEstimateJISG=m_adjacencyEstimateJISG./max(max(m_adjacencyEstimateJISG));
                            figure(2); imagesc(m_adjacencyEstimateJISG);
                            

                            [As,s_edgePredEr]=BlindTopologySignalIDSimulations.reconstructAdj(m_adjacencyEstimateJISG,m_adjacency,v_thressholds);
                            %%2. Compare error
                            v_error{s_it,1}={v_samplepct(s_sampleSizeInd),s_lambda1,s_lambda2,s_mu};
                            v_error{s_it,2}=s_edgePredEr;
                            v_error{s_it,3}=BlindTopologySignalIDSimulations.measureMSE(m_adjacencyEstimateJISG,m_adjacency);
                            s_it=s_it+1;
                            end
                        end
                    end
                end
            
            end
%             m_meanSquaredError=[v_NMSELSEM,v_NMSE]';
%             legend=[legendSEM,legendJISG]';
            save('partuning');
             figure(2); imagesc(m_adjacencyEstimateJISG);
            F = [];%F_figure('X',v_samplepct*s_numberOfVertices,'Y',m_meanSquaredError,'xlab','sample size (M)','ylab','NMSE','leg',legend);
            %F.logy=1;
        end
            % Data used: Real gene signal
        % Goal: compare graph reconstruction
        function F = compute_fig_11007(obj,niter)
            
            %%0. define parameters
            s_lambda1 =10^-2;
            s_lambda2 = 10^-4;
            s_maxIterSEMInit = 40;
            s_maxIterJoint=40;
            s_rho = 1e4;% rho = 10;
            s_tolSEM = 1e-5;
            b_display= 0;
            s_mu=10^8;
            s_tolJoint=1e-6;
            s_monteCarloSimulations=niter;
            s_SNR=Inf; % no noise real data
            s_layers=100;
            s_bandwidth=10;
            s_bandwidth1=s_bandwidth+2;
            s_maxIterSEMLongRun=40;
            v_samplepct=0.8;%:0.05:1;
            v_thressholds=10.^(-3:0.1:-0.5);%0.01:0.02:0.6;
            s_differentSampleSetSizenum=size(v_samplepct,2);
            %%1. Simulation
            %create graph
            L=4;                % level of kronecker graph
            s=[0.6 0.1 0.7;
                0.3 0.1 0.5;
                0 1 0.1];
          
            
            %different montecarlo iterations
            m_NMSE=zeros(s_differentSampleSetSizenum,s_monteCarloSimulations);
            m_NMSELSEM=zeros(s_differentSampleSetSizenum,s_monteCarloSimulations);
            m_edgePredErrorLSEM=zeros(s_differentSampleSetSizenum,s_monteCarloSimulations);
            m_edgePredError=zeros(s_differentSampleSetSizenum,s_monteCarloSimulations);
            
            for s_monteCarloSimulationsInd=1:s_monteCarloSimulations
                m_graphFunction = readGeneNetworkDataset;
                s_numberOfVertices=size(m_graphFunction,1);
                s_layers=size(m_graphFunction,2);

                colormap('gray');
                %create graph function
              
                v_NMSE=zeros(s_differentSampleSetSizenum,1);
                v_NMSELSEM=zeros(s_differentSampleSetSizenum,1);
                v_edgePredError=zeros(s_differentSampleSetSizenum,1);
                v_edgePredErrorLSEM=zeros(s_differentSampleSetSizenum,1);
                %sample differnt sizes
                
                for s_sampleSizeInd=1:s_differentSampleSetSizenum
                    v_samplepctpersize=v_samplepct(s_sampleSizeInd)*ones(1,s_layers);
                    m_graphFunctionWithMisses=zeros(size(m_graphFunction));
                    for s_layerInd=1:s_layers
                        s_numberOfSamples=round(s_numberOfVertices*v_samplepctpersize(s_layerInd));
                        sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                        [c_samples{s_layerInd},...
                            c_positions{s_layerInd}]...
                            = sampler.sample(m_graphFunction(:,s_layerInd));
                        m_S=BlindTopologySignalIDSimulations.createSamplingMatrix(c_positions{s_layerInd},s_numberOfVertices);
                        m_graphFunctionWithMisses(:,s_layerInd)=m_S'*c_samples{s_layerInd};
                    end
                    %estimate functions
                    %sem approach
                    sEMGraphGraphFunctionEstimator=SEMGraphGraphFunctionEstimator('s_lambda1',s_lambda1,'s_lambda2',s_lambda2,...
                        's_maxIterSEMInit',s_maxIterSEMInit,'s_maxIterSEMLongRun',s_maxIterSEMLongRun,...
                        'm_graphFunctionInitialize',m_graphFunctionWithMisses,'s_numberOfVertices',s_numberOfVertices,...
                        's_maxIterJoint',s_maxIterJoint,'s_tolJoint',s_tolJoint,'s_rho',s_rho,'s_tolSEM',s_tolSEM,'s_mu',s_mu);
                    [m_adjacencyEstimateJISG,m_LSEMestimate] =sEMGraphGraphFunctionEstimator.estimate(c_samples,c_positions);
                    legendJISG={'JISG'};
                    lSEMGraphEstimator=LSEMGraphEstimator('s_lambda1',s_lambda1,'s_lambda2',s_lambda2,'s_maxIter',s_maxIterSEMInit,'s_rho',s_rho,'s_tol',s_tolSEM);
                    m_adjacencyEstimatSEM=lSEMGraphEstimator.estimate(m_graphFunction);
                    legendSEM={'LSEM'};
                    colormap('gray');
                   % m_adjacencyEstimateJISG=m_adjacencyEstimateJISG./max(max(m_adjacencyEstimateJISG));
                    figure(1); imagesc(m_adjacencyEstimateJISG);
                    
                    %m_adjacencyEstimatSEM=m_adjacencyEstimatSEM./max(max(m_adjacencyEstimatSEM));
                    colormap('gray');
                    figure(2); imagesc(m_adjacencyEstimatSEM);
                    
                 
                end
     
                
            end
            nmse=BlindTopologySignalIDSimulations.estimateNormalizedMeanSquaredError(m_LSEMestimate,m_graphFunction)
            save('libGF/simulations/BlindTopologySignalIDSimulations_data/resultsGene');
            
            F = [];%F_figure('X',v_samplepct*s_numberOfVertices,'Y',m_meanSquaredError,'xlab','sample size (M)','ylab','NMSE','leg',legend);
            %F.logy=1;
        end
       
        
        %% Real
        %Data used: recommender system
        %Goal: compare NMSE
        function F=compute_fig_12001(obj,niter)
            %%0. define parameters
            s_lambda1 =0.5;
            s_lambda2 = 0.1;
            s_maxIter = 40;
            s_rho = 1e4;% rho = 10;
            s_tol = 1e-5;
            b_display= 0;
            [R_train,R_test,c_Um,c_Up,c_rUm,c_S,c_y]=readMovieLensDatasetYanning;
            %%1. Simulation
            lSEMGraphEstimator=LSEMGraphEstimator('s_lambda1',s_lambda1,'s_lambda2',s_lambda2,'s_maxIter',s_maxIter,'s_rho',s_rho,'s_tol',s_tol,'b_display',b_display);
            m_adjacency=lSEMGraphEstimator.estimate(R_train);
            
            %%2. Compare error
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% TV methods
             %% Real data simulations
        %  Data used: Temperature Time Series in places across continental
        %  USA
        %  Goal: tracking
        function F = compute_fig_21001(obj,niter)
                %% 0. define parameters
            % maximum signal instances sampled
             
            % maximum signal instances sampled
            s_maximumTime=150;
            
            % sample period: we have total 8759 time instances (hours
            % throught a year 8760) so if we want to sample per day we
            % pick period 24 if we want to sample per month average time of
            % hours per month is 720 week 144
            s_samplePeriod=1;
            
            % KKrKF parameters
            %regularization parameter
            s_muKrKF=10^-7;
            s_sigmaForDiffusionKrKF=2.2;
            %Obs model
            s_obsSigmaKrKF=0.01;
            %Kr KF
            s_stateSigmaKrKF=0.000005;
            s_pctOfTrainPhaseKrKF=0.30;
            s_transWeightKrKF=0.000001;
            %Multikernel
            s_meanspatioKrKF=2;
            s_stdspatioKrKF=0.5;
            s_numberspatioOfKernelsKrKF=40;
            v_sigmaForDiffusionKrKF= abs(s_meanspatioKrKF+ s_stdspatioKrKF.*randn(s_numberspatioOfKernelsKrKF,1)');
             s_meanstnKrKF=10^-4;
            s_stdstnKrKF=10^-5;
            s_numberstnOfKernelsKrKF=40;
            v_sigmaForstn= abs(s_meanstnKrKF+ s_stdstnKrKF.*randn(s_numberstnOfKernelsKrKF,1)');
            %v_sigmaForDiffusion=[1.4,1.6,1.8,2,2.2];

            s_numberspatioOfKernelsKrKF=size(v_sigmaForDiffusionKrKF,2);
            s_lambdaForMultiKernelsKrKF=10^2;
            s_stepSizeCov=0.999;

            s_monteCarloSimulations=niter;
            s_SNR=Inf; % no noise real data
            
            %sample size percentage
            v_samplePercentage=(0.4:0.4:0.4);
            % LMS step size
            s_stepLMS=1.8;
            %DLSR parameters
            s_muDLSR=1.5;
            s_betaDLSR=0.4;
            
            %bandwidth of bandlimited approaches
            v_bandwidthBL=[8];
            v_bandwidthLMS=[40];
            v_bandwidthDLSR=[10];
            %% JISG
            s_lambda1JISG =10^-2;
            s_lambda2JISG = 1;
            s_maxIterSEMInit = 40;
            s_maxIterJoint=40;
            s_rho = 1e5;% rho = 10;
            s_tolSEM = 1e-5;
            b_display= 0;
            s_muJISG=10^2;
            s_tolJoint=1e-6;
            s_SNR=Inf; % no noise real data
            s_maxIterSEMLongRun=40;
            v_thressholds=10.^(-3:0.1:-0.5);%0.01:0.02:0.6;
           
            %% 1. define graph
            tic
          
            load('temperatureTimeSeriesData.mat');
            %m_adjacency=m_adjacency/max(max(m_adjacency));  % normalize adjacency  
            
            s_numberOfVertices=size(m_adjacency,1);  % size of the graph          
            
            m_initialStateJSIG=zeros(s_numberOfVertices,s_monteCarloSimulations); % mean of initial state
            t_initialErrorCovJISG=eye(s_numberOfVertices);
            t_initialErrorCovJISG=repmat(t_initialErrorCovJISG,[1,1,s_monteCarloSimulations]);
            m_adjtempInitJISG=eye(s_numberOfVertices);
            v_numberOfSamples=...                              % must extend to support vector cases
                round(s_numberOfVertices*v_samplePercentage);
        
            %select a subset of measurements
            s_totalTimeSamples=size(m_temperatureTimeSeries,2);
         
            
            s_timeSamples= round(s_totalTimeSamples/s_samplePeriod);
            s_maximumTime=min([s_timeSamples,s_maximumTime,s_totalTimeSamples]);
            s_trainTimePeriod=round(s_pctOfTrainPhaseKrKF*s_maximumTime);
            s_trainTime=round(s_pctOfTrainPhaseKrKF*s_maximumTime);

            
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
            
            diffusionGraphKernel=DiffusionGraphKernel('s_sigma',s_sigmaForDiffusionKrKF,'m_laplacian',graph.getLaplacian);
            m_diffusionKernel=diffusionGraphKernel.generateKernelMatrix;
 
            % generate transition, correlation matrices
           

            % Transition matrix for KrKf
            t_transitionKrKF=...
                repmat(s_transWeightKrKF*eye(s_numberOfVertices),[1,1,s_maximumTime]);
            % Kernels for KrKF
            t_dictionaryOfKernels=zeros(s_numberspatioOfKernelsKrKF,s_numberOfVertices,s_numberOfVertices);
            t_dictionaryOfEigenValues=zeros(s_numberspatioOfKernelsKrKF,s_numberOfVertices,s_numberOfVertices);
            v_thetaSpat=ones(s_numberspatioOfKernelsKrKF,1);
            m_combinedKernel=zeros(s_numberOfVertices,s_numberOfVertices);
            m_combinedKernelEig=zeros(s_numberOfVertices,s_numberOfVertices);
            [m_eigenvectorsAll,m_eigenvaluesAll]=KrKFonGSimulations.transformedDifEigValues(graph.getLaplacian,v_sigmaForDiffusionKrKF);
            for s_kernelInd=1:s_numberspatioOfKernelsKrKF
                diffusionGraphKernel=DiffusionGraphKernel('s_sigma',v_sigmaForDiffusionKrKF(s_kernelInd),'m_laplacian',graph.getLaplacian);
                m_difker=diffusionGraphKernel.generateKernelMatrix;
                t_dictionaryOfKernels(s_kernelInd,:,:)=m_difker;
                %t_dictionaryOfKernels(s_kernelInd,:,:)=squeeze(t_dictionaryOfKernels(s_kernelInd,:,:))+eye(s_numberOfVertices);
                m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
                t_dictionaryOfEigenValues(s_kernelInd,:,:)=diag(m_eigenvaluesAll(:,s_kernelInd));
                %t_dictionaryOfEigenValues(s_kernelInd,:,:)=squeeze(t_dictionaryOfEigenValues(s_kernelInd,:,:))+eye(s_numberOfVertices);
                m_combinedKernelEig=m_combinedKernelEig+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfEigenValues(s_kernelInd,:,:));
            end

         
            
            m_stateEvolutionKernel=s_stateSigmaKrKF^2*eye(s_numberOfVertices);
            %m_stateEvolutionKernel=repmat(m_stateEvolutionKernel,[1,1,s_maximumTime]);
            
            %% 3. generate true signal
            
            m_graphFunction=m_temperatureTimeSeriesSampled;
            
            t_graphFunction=repmat(m_graphFunction,1,1,s_monteCarloSimulations);
            
            
            %% 4.0 Estimate signal
            t_kfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_mkrkfEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2));
            t_krkfEstimate=zeros(s_numberOfVertices,s_maximumTime,s_monteCarloSimulations);
            t_distrEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));         
            t_lmsEstimate=zeros(s_numberOfVertices*s_maximumTime,s_monteCarloSimulations...
                ,size(v_numberOfSamples,2),size(v_bandwidthBL,2));
           
            c_samples=cell(size(v_numberOfSamples,2),1);
            c_positions=cell(size(v_numberOfSamples,2),1); %contains the samples for each sampling size
           
            for s_sampleInd=1:size(v_numberOfSamples,2)
                %% 4. generate observations
                s_numberOfSamples=v_numberOfSamples(s_sampleInd);         
                m_obsNoiseCovariance=s_obsSigmaKrKF^2*eye(s_numberOfSamples);
                t_obsNoiseCovariace=repmat(m_obsNoiseCovariance,[1,1,s_maximumTime]);
                sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
                t_samples=zeros(s_numberOfSamples,s_maximumTime,s_monteCarloSimulations);
                t_positions=zeros(s_numberOfSamples,s_maximumTime,s_monteCarloSimulations);
%                [m_samples(1:s_numberOfSamples,:),...
%                    m_positions(1:s_numberOfSamples,:)]...
%                    = sampler.sample(m_graphFunction(1:s_numberOfVertices,:));
                t_graphFunctionInit=zeros(size(t_graphFunction));
                for s_timeInd=1:s_maximumTime
            [t_samples(:,s_timeInd,:),t_positions(:,s_timeInd,:)]...
                    = sampler.sample(squeeze(t_graphFunction(:,s_timeInd,:)));
                    for s_niter=1:s_monteCarloSimulations
                        t_graphFunctionInit(t_positions(:,s_timeInd,s_niter),s_timeInd,s_niter)=t_samples(:,s_timeInd,s_niter);
                    end
                    
                end
    
                sEMGraphGraphFunctionEstimator=SVARGraphGraphFunctionEstimator('s_lambda10',s_lambda1JISG,'s_lambda11',s_lambda1JISG,'s_lambda20',s_lambda2JISG,'s_lambda21',s_lambda2JISG,...
                        's_maxIterSEMInit',s_maxIterSEMInit,'s_maxIterSEMLongRun',s_maxIterSEMLongRun,...
                        't_graphFunctionInitialize',t_graphFunctionInit,'s_numberOfVertices',s_numberOfVertices,'s_boolInit',1,...
                        's_maxIterJoint',s_maxIterJoint,'s_maximumTime',s_maximumTime,'s_tolJoint',s_tolJoint,'s_rho',s_rho,'s_tolSEM',s_tolSEM,'s_mu',s_muJISG,...
                    't_initialErrorCov',t_initialErrorCovJISG,'m_initialState',m_initialStateJSIG,'m_adjInit',m_adjacency,'m_adjtempInit',m_adjtempInitJISG);
                
                [m_adjacency,m_timeAdjacency,t_graphFunctionEstimate]=sEMGraphGraphFunctionEstimator.estimate(t_samples,t_positions);
                %% Other approaches
                
%                 %% 4.5 MKrKF estimate
%                 krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
%                     't_previousMinimumSquaredError',t_initialSigma0,...
%                     'm_previousEstimate',m_initialState);
%                 l2MultiKernelKrigingCorEstimator=L2MultiKernelKrigingCorEstimator...
%                     ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
%                  l2MultiKernelKrigingCorEstimatorEig=L2MultiKernelKrigingCorEstimator...
%                     ('t_kernelDictionary',t_dictionaryOfEigenValues,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
%                  l1MultiKernelKrigingCorEstimator=L1MultiKernelKrigingCorEstimator...
%                     ('t_kernelDictionary',t_dictionaryOfKernels,'s_lambda',s_lambdaForMultiKernels,'s_obsNoiseVar',s_obsSigma^2);
%                 batchEmpiricalMeanCovEstimator=BatchEmpiricalMeanCovEstimator;
%                 onlineEmpiricalMeanCovEstimator=OnlineEmpiricalMeanCovEstimator('s_stepSize',s_stepSizeCov);
%                 % used for parameter estimation
%                 s_auxInd=0;
%                 t_approximSpat=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod); 
%                 t_residualState=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
%                 diffusionGraphKernel=DiffusionGraphKernel('s_sigma',mean(v_sigmaForDiffusion),'m_laplacian',graph.getLaplacian);
%                 m_combinedKernel=diffusionGraphKernel.generateKernelMatrix;
%                 for s_timeInd=1:s_maximumTime
%                     %time t indices
%                     v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
%                         (s_timeInd)*s_numberOfVertices;
%                     v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
%                         (s_timeInd)*s_numberOfSamples;
%                     %m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
%                     m_spatialCovariance=m_combinedKernel;
%                     m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
%                     
%                     %samples and positions at time t
%                     m_samplest=m_samples(v_timetIndicesForSamples,:);
%                     m_positionst=m_positions(v_timetIndicesForSamples,:);
%                     %estimate
%                     [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
%                         (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),m_stateEvolutionKernel,m_spatialCovariance,m_obsNoiseCovariace);
%                     
%                     
%                     
%                     %prepare kf for next iter
%                     
%                     t_mkrkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
%                         m_estimateKR+m_estimateKF;
%                     
%                     krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
%                     krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
%                     %% Multikernel
%                     if s_timeInd>1&&s_timeInd<s_trainTimePeriod
%                         s_auxInd=s_auxInd+1;
%                         % save approximate matrix
%                          t_approximSpat(:,:,s_auxInd)=m_estimateKR;
%                             t_residualState(:,:,s_auxInd)=m_estimateKF...
%                             -t_transitionKrKF(:,:,s_timeInd)*m_estimateKFPrev;
% 
%                     end
%                     if s_timeInd==s_trainTime
%                         %calculate exact theta estimate
%                         t_approxSpatCor=batchEmpiricalMeanCovEstimator.calculateResidualCorBatch(t_approximSpat,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
%                         t_approxStateCor=batchEmpiricalMeanCovEstimator.calculateResidualCorBatch(t_residualState,s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
%                         for s_monteCarloInd=1:s_monteCarloSimulations
%                             t_transformedSpatCor(:,:,s_monteCarloInd)=m_eigenvectorsAll'*t_approxSpatCor(:,:,s_monteCarloInd)*m_eigenvectorsAll;
%                             t_transformedStateCor(:,:,s_monteCarloInd)=m_eigenvectorsAll'*t_approxStateCor(:,:,s_monteCarloInd)*m_eigenvectorsAll;
%                         end
%                         t01=trace(squeeze(t_transformedSpatCor(:,:,1))*inv(m_combinedKernelEig))+s_lambdaForMultiKernels*norm(v_thetaSpat)^2
%                         t02=trace(squeeze(t_approxSpatCor(:,:,1))*inv(m_combinedKernel))+s_lambdaForMultiKernels*norm(v_thetaSpat)^2
%                         tic
%                         %v_thetaSpat=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorCVX(t_approxSpatCor);
%                         %t1=trace(squeeze(t_approxSpatCor(:,:,1))*inv(v_thetaSpat(1)*squeeze(t_dictionaryOfKernels(1,:,:))+v_thetaSpat(2)*squeeze(t_dictionaryOfKernels(2,:,:))))+s_lambdaForMultiKernels*norm(v_thetaSpat)^2;
%                         %v_thetaSpatl2eig=l2MultiKernelKrigingCorEstimatorEig.estimateCoeffVectorCVX(t_transformedSpatCor);
%                         %v_thetaSpat=l1MultiKernelKrigingCorEstimator.estimateCoeffVectorCVX(t_approxSpatCor);
%                         %v_thetaSpat=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorGD(t_transformedSpatCor,m_eigenvaluesAll);
%                         v_thetaSpat=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorNewton(t_transformedSpatCor,m_eigenvaluesAll);
%                         %v_thetaSpatAN=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorOnlyDiagNewton(t_transformedSpatCor,m_eigenvaluesAll);
%                         %t2=trace(squeeze(t_approxSpatCor(:,:,1))*inv(v_thetaSpatN(1)*squeeze(t_dictionaryOfKernels(1,:,:))+v_thetaSpatN(2)*squeeze(t_dictionaryOfKernels(2,:,:))))+s_lambdaForMultiKernels*norm(v_thetaSpatN)^2;
%                         %v_theta=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorGDWithInit(t_transformedSpatCor,m_eigenvaluesAll,v_thetaSpat);
%                         v_thetaState=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorCVX(t_approxStateCor);
%                         timeCVX=toc
% %                         tic
% %                         v_thetaSpat=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGD(t_residualSpatCov,m_positionst);
% %                         v_thetaState=l2MultiKernelKrigingCovEstimator.estimateCoeffVectorGD(t_residualStateCov,m_positionst);
% %                         timeGD=toc
%                         m_combinedKernel=zeros(s_numberOfVertices);
%                         for s_kernelInd=1:s_numberspatioOfKernels
%                             m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
%                         end
%                         m_stateEvolutionKernel=zeros(s_numberOfVertices);
%                         for s_kernelInd=1:s_numberspatioOfKernels
%                             m_stateEvolutionKernel=m_stateEvolutionKernel+v_thetaState(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
%                         end
%                         m_stateEvolutionKernel=s_stateSigma^2*m_stateEvolutionKernel;
%                         s_auxInd=0;
%                         t_approximSpat=zeros(s_numberOfSamples,s_monteCarloSimulations);
%                         t_residualState=zeros(s_numberOfSamples,s_monteCarloSimulations);
%                     end
%                     if s_timeInd>s_trainTime
% %                         do a few gradient descent steps
% %                         combine using formula
%                         t_approxSpatCor=onlineEmpiricalMeanCovEstimator.incrementalCalcResCor...
%                          (t_approxSpatCor,m_estimateKR,s_timeInd);
%                         m_estimateStateNoise=m_estimateKF...
%                             -t_transitionKrKF(:,:,s_timeInd)*m_estimateKFPrev;
%                         t_approxStateCor=onlineEmpiricalMeanCovEstimator.incrementalCalcResCor...
%                             (t_approxStateCor,m_estimateStateNoise,s_timeInd);
%                         tic
%                         for s_monteCarloInd=1:s_monteCarloSimulations
%                             t_transformedSpatCor(:,:,s_monteCarloInd)=m_eigenvectorsAll'*t_approxSpatCor(:,:,s_monteCarloInd)*m_eigenvectorsAll;
%                             t_transformedStateCor(:,:,s_monteCarloInd)=m_eigenvectorsAll'*t_approxStateCor(:,:,s_monteCarloInd)*m_eigenvectorsAll;
%                         end
%                         v_thetaSpat=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorGDWithInit(t_transformedSpatCor,m_eigenvaluesAll,v_thetaSpat);
%                         %v_thetaState=l2MultiKernelKrigingCorEstimator.estimateCoeffVectorGDWithInit(t_approxStateCor,m_positionst,v_thetaState);
%                         timeGD=toc
%                         s_timeInd
%                         m_combinedKernel=zeros(s_numberOfVertices);
%                         for s_kernelInd=1:s_numberspatioOfKernels
%                             m_combinedKernel=m_combinedKernel+v_thetaSpat(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
%                         end
% %                         m_stateEvolutionKernel=zeros(s_numberOfVertices);
% %                         for s_kernelInd=1:s_numberOfKernels
% %                             m_stateEvolutionKernel=m_stateEvolutionKernel+v_thetaState(s_kernelInd)*squeeze(t_dictionaryOfKernels(s_kernelInd,:,:));
% %                         end
%                         s_auxInd=0;
%                     end
%                     
%                     
%                     m_estimateKFPrev=m_estimateKF;
%                     t_MSEKFPRev=t_MSEKF;
%                     m_estimateKRPrev=m_estimateKR;
% 
%                 end
%                 %% 4.6 KKrKF estimate
%                 krigedKFonGFunctionEstimator=KrigedKFonGFunctionEstimator('s_maximumTime',s_maximumTime,...
%                     't_previousMinimumSquaredError',t_initialSigma0,...
%                     'm_previousEstimate',m_initialState);
%                 batchEmpiricalMeanCovEstimator=BatchEmpiricalMeanCovEstimator;
%                 onlineEmpiricalMeanCovEstimator=OnlineEmpiricalMeanCovEstimator('s_stepSize',s_stepSizeCov);
%                 % used for parameter estimation
%                 s_auxInd=0;
%                 t_approximSpat=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
%                 t_residualState=zeros(s_numberOfVertices,s_monteCarloSimulations,s_trainTimePeriod);
%                 m_combinedKernel=m_diffusionKernel;
%                 m_stateEvolutionKernel=s_stateSigma^2*eye(s_numberOfVertices);
%                 for s_timeInd=1:s_maximumTime
%                     %time t indices
%                     v_timetIndicesForSignals=(s_timeInd-1)*s_numberOfVertices+1:...
%                         (s_timeInd)*s_numberOfVertices;
%                     v_timetIndicesForSamples=(s_timeInd-1)*s_numberOfSamples+1:...
%                         (s_timeInd)*s_numberOfSamples;
%                     %m_spatialCovariance=t_spatialCovariance(:,:,s_timeInd);
%                     m_spatialCovariance=m_diffusionKernel;
%                     m_obsNoiseCovariace=t_obsNoiseCovariace(:,:,s_timeInd);
%                     
%                     %samples and positions at time t
%                     m_samplest=m_samples(v_timetIndicesForSamples,:);
%                     m_positionst=m_positions(v_timetIndicesForSamples,:);
%                     %estimate
%                     [m_estimateKR,m_estimateKF,t_MSEKF]=krigedKFonGFunctionEstimator.estimate...
%                         (m_samplest,m_positionst,squeeze(t_transitionKrKF(:,:,s_timeInd)),m_stateEvolutionKernel,m_spatialCovariance,m_obsNoiseCovariace);
%                     
%                     
%                     
%                     %prepare kf for next iter
%                     
%                     t_krkfEstimate(v_timetIndicesForSignals,:,s_sampleInd)=...
%                         m_estimateKR+m_estimateKF;
%                     
%                     krigedKFonGFunctionEstimator.t_previousMinimumSquaredError=t_MSEKF;
%                     krigedKFonGFunctionEstimator.m_previousEstimate=m_estimateKF;
%                  
%                 end
% 
%                 %% 4.7 DLSR                
%                 
%                 for s_bandInd=1:size(v_bandwidthDLSR,2)
%                     s_bandwidth=v_bandwidthDLSR(s_bandInd);
%                     distributedFullTrackingAlgorithmEstimator=...
%                         DistributedFullTrackingAlgorithmEstimator('s_maximumTime',s_maximumTime,...
%                         's_bandwidth',s_bandwidth,'graph',graph,'s_mu',s_muDLSR,'s_beta',s_betaDLSR);
%                     t_distrEstimate(:,:,s_sampleInd,s_bandInd)=...
%                         distributedFullTrackingAlgorithmEstimator.estimate(m_samples,m_positions);
%                     
%                     
%                 end
%                 %% 4.8 LMS
%                 for s_bandInd=1:size(v_bandwidthLMS,2)
%                     s_bandwidth=v_bandwidthLMS(s_bandInd);
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
%                 
%                 
                
                
            end
            
            
            %% 5. measure difference
            
%           for s_vertInd=1:s_numberOfVertices
                
%                 
%                 m_meanEstKF(s_vertInd,:)=mean(t_kfEstimate((0:s_maximumTime-1)*...
%                     s_numberOfVertices+s_vertInd,1,:),2)';
%                 m_meanEstKrKF(s_vertInd,:)=mean(t_krkfEstimate((0:s_maximumTime-1)*...
%                     s_numberOfVertices+s_vertInd,1,:),2)';
%                 m_meanEstMKrKF(s_vertInd,:)=mean(t_mkrkfEstimate((0:s_maximumTime-1)*...
%                     s_numberOfVertices+s_vertInd,1,:),2)';
%                 m_meanEstDLSR(s_vertInd,:)=mean(t_distrEstimate((0:s_maximumTime-1)*...
%                     s_numberOfVertices+s_vertInd,:,1,1),2)';
%                 m_meanEstLMS(s_vertInd,:)=mean(t_lmsEstimate((0:s_maximumTime-1)*...
%                     s_numberOfVertices+s_vertInd,:,1,1),2)';
                            
%           end
            %m_meanEstSmooth=mean(t_graphFunctionEstimate,3);    
            m_meanEstSmooth=squeeze(t_graphFunctionEstimate(:,:,1));    
            %Greece node 43
           %v_vertexToPlot=setdiff((1:s_numberOfVertices),m_positionst);

            s_vertexToPlot=1;%v_vertexToPlot(1);
             if(ismember(s_vertexToPlot,t_positions))
                 warning('vertex to plot sampled')
             end
                        myLegendDLSR{1}=strcat('DLSR',...
                            sprintf(' B=%g',v_bandwidthDLSR));
                        myLegendLMS{1}=strcat('LMS',...
                            sprintf(' B=%g',v_bandwidthLMS));
                    myLegendMKrKF{1}='MKrKF';
                    myLegendKrKF{1}='KKrKF';
                    myLegendJSIG='JSIG';
            myLegandTrueSignal{1}='True temperature';
            myLegend=[myLegandTrueSignal... myLegendDLSR myLegendLMS myLegendKrKF myLegendMKrKF 
                myLegendJSIG];
            F = F_figure('X',(1:s_maximumTime),'Y',[m_temperatureTimeSeriesSampled(s_vertexToPlot,:);...
                ...m_meanEstDLSR(s_vertexToPlot,:);
                ...m_meanEstLMS(s_vertexToPlot,:);m_meanEstKrKF(s_vertexToPlot,:);m_meanEstMKrKF(s_vertexToPlot,:)
                m_meanEstSmooth(s_vertexToPlot,:);],...
                'xlab','Time','ylab','Temperature[F]','leg',myLegend);
           F.caption=[	sprintf('regularization parameter mu=%g\n',s_muKrKF),...
                sprintf(' diffusion parameter sigma=%g\n',s_sigmaForDiffusionKrKF)...
                sprintf('s_stateSigma =%g\n',s_stateSigmaKrKF)...
                sprintf('s_transWeight =%g\n',s_transWeightKrKF)...
                sprintf('mu for DLSR =%g\n',s_muDLSR)...
                sprintf('beta for DLSR =%g\n',s_betaDLSR)...
                sprintf('step LMS =%g\n',s_stepLMS)...
                sprintf('sampling size =%g\n',v_samplePercentage)];
            
        end
       
        
    end
    
    methods(Static)
        %calculates the best thresshold that gives the minimum edge
        %identification  error.
        function s_error=measureMSE(m_estimatedAdja,m_trueAdj)
            s_error=norm(m_estimatedAdja-m_trueAdj,'fro');
        end
        %calculates the best thresshold that gives the minimum edge
        %identification  error.
        function [As,er]=reconstructAdj(m_estimatedAdja,m_trueSupport,v_thressholds)
            % A_hat: estimated adjacency
            % S: true support of the actual adjacency
            % tau1: vector of candidate thresholds for deciding zeros
            % As: estimated support matrix
            
            N=size(m_trueSupport,1);
            
            
            for ii=1:length(v_thressholds)
                tau=v_thressholds(ii);
                As=zeros(N,N);
                As(find(abs(m_estimatedAdja)>tau))=1;
                err(ii)=nnz(As-m_trueSupport);
            end
            [er, id]=min(err);
            er;%=(1/((N-1)*N))*er;
            %tau1(id)
            As=zeros(N,N);
            As(find(abs(m_estimatedAdja)>v_thressholds(id)))=1;
        end
        %creates sampling matrices from indicator vector
        function m_S=createSamplingMatrix(v_ratedIndex,Nm)
            m_S=zeros(size(v_ratedIndex,1),Nm);
            for s_ratIndexNm=1:size(v_ratedIndex,1)
                m_S(s_ratIndexNm,v_ratedIndex(s_ratIndexNm))=1;
            end
        end
        %estimates the normalized mean squared error
        function res = estimateNormalizedMeanSquaredError(m_est,m_observed)
            res=0;
            for i=1:size(m_est,2)
                % 				if (norm(m_est(:,i)-m_observed(:,i))^2/norm(m_observed(:,i))^2)>1
                %                     huge error normalize exclude this monte carlo
                %                     for bandlimited approaches
                %                     res=res+1;
                %                 else
                res=res+norm(m_est(:,i)-m_observed(:,i))^2/norm(m_observed(:,i))^2;
                %                 end
            end
            res=(1/size(m_est,2))*res;
        end
        function res = estimateNormalizedMeanSquaredErrorUnsampled(m_est,m_observed,c_positions)
            res=0;
            for i=1:size(m_est,2)
                
                v_ind=setdiff((1:size(m_est,1)),c_positions{i});
                % 				if (norm(m_est(:,i)-m_observed(:,i))^2/norm(m_observed(:,i))^2)>1
                %                     huge error normalize exclude this monte carlo
                %                     for bandlimited approaches
                %                     res=res+1;
                %                 else
                res=res+norm(m_est(v_ind,i)-m_observed(v_ind,i))^2/norm(m_observed(v_ind,i))^2;
                %                 end
            end
            res=(1/size(m_est,2))*res;
        end
        %estimates the root mean squared error over the known values
        function res = estimateMeanSquaredError(m_est,m_observed)
            res=0;
            for i=1:size(m_est,2)
                res=res+norm(m_est(:,i)-m_observed(:,i))^2;
            end
            res=(1/size(m_est,2))*res;
        end
        function [recall, precision]=test_rec(R_est,Up,Um,n_prob,Nu,N)
            
            D=length(N);
            
            
            k=1;
            flag=zeros(n_prob,D);
            for i=1:Nu
                id_p=Up{i};
                um_temp=Um{i};
                for j=1:length(id_p)
                    
                    id_temp=randperm(length(um_temp),1000);
                    cid=um_temp(id_temp);
                    
                    r_est=sort(R_est(i,cid),'descend');
                    for d=1:D
                        trd=r_est(N(d));
                        flag(k,d)=(trd<R_est(i,id_p(j)));
                        k=k+1;
                    end
                end
            end
            
            recall=sum(flag,1)/n_prob;
            
            precision=recall./N;
        end
    end
    
end
