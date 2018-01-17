%
%
%  FIGURES FOR THE PAPER Joint Topology ID ICASSP
%
%

classdef BlindTopologySignalIDSimulations < simFunctionSet
    
    properties
        
    end
    
    methods
        %% Synthetic
        % Data used: Synthetic graph
        % Goal: methods comparison NMSE
        function F = compute_fig_1001(obj,niter)
            
            %%0. define parameters
            s_lambda1 =0.4;
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
            sEMGraphGraphFunctionEstimator=SEMGraphGraphFunctionEstimator('s_lambda1',s_lambda1,'s_lambda2',s_lambda2,'s_maxIterSEM',s_maxIterSEM,...
                's_numberOfVertices',s_numberOfVertices,'s_maxIterJoint',s_maxIterJoint,'s_tolJoint',s_tolJoint,'s_rho',s_rho,'s_tolSEM',s_tolSEM,'s_mu',s_mu);
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
        function F = compute_fig_1002(obj,niter)
            
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
                    sEMGraphGraphFunctionEstimator=SEMGraphGraphFunctionEstimator('s_lambda1',s_lambda1,'s_lambda2',s_lambda2,'s_maxIterSEM',s_maxIterSEM,...
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
        function F = compute_fig_1003(obj,niter)
            
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
        function F = compute_fig_1004(obj,niter)
            
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
        function F = compute_fig_1005(obj,niter)
            
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
        function F = compute_fig_1006(obj,niter)
            
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
        function F = compute_fig_1007(obj,niter)
            
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
        function F=compute_fig_2001(obj,niter)
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
