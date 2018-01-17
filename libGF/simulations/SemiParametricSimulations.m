%
%  FIGURES FOR THE PAPER ON SemiParametric
%
%

classdef SemiParametricSimulations < simFunctionSet
	
	properties
		
	end
	
	methods
		%% Real data simulations 
		%  Data used: Swiss temperature
		% Goal: methods comparison NMSE
		function F = compute_fig_1001(obj,niter)
			
			%0. define parameters
			s_sigma=1.3;
			s_numberOfClusters=5;
			s_lambda=10^-5;
			s_monteCarloSimulations=niter;%100;
			s_bandwidth1=10;
			s_bandwidth2=20;
			s_SNR=1000;
			s_beta=0.02;
			s_alpha=0.005;
			s_niter=10;
			s_epsilon=0.2;
			s_functionTypes=5;
			v_sampleSetSize=(0.1:0.1:1);
			
			m_meanSquaredError=zeros(size(v_sampleSetSize,2),s_functionTypes);
			% define graph
			[Ho,Mo,Alto,Hn,Mn,Altn] = readTemperatureDataset;
			%tic
			graphGenerator = GraphLearningSmoothSignalGraphGenerator('m_observed',Ho,'s_niter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta,'m_missingValuesIndicator',[]);
			%toc
			
			% tic
			% graphGenerator=SmoothSignalGraphGenerator('m_observed',Ho,'s_maxIter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta);
			% toc
			
			graph = graphGenerator.realization;
			%graph1 = graphGenerator1.realization;
			%L1=graph.getLaplacian
			%L2=graph1.getLaplacian
			v_sampleSetSize=round(v_sampleSetSize*graph.getNumberOfVertices);
			
			
			m_basis= SemiParametricSimulations.parametricPartForTempData(graph.getLaplacian,Alto,s_numberOfClusters);
			
			%functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth);
			%functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
			%signal
			v_realSignal=Hn(:,3);
			functionGenerator=RealDataGraphFunctionGenerator('graph',graph,'v_realSignal',v_realSignal,'s_normalize',1);
			% define bandlimited function estimator
			%m_laplacianEigenvectors=(graph.getLaplacianEigenvectors);
			bandlimitedGraphFunctionEstimator1 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth1);
			bandlimitedGraphFunctionEstimator2 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth2);
			
			% define Kernel function
			diffusionGraphKernel = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigma);
			%define non-parametric estimator
			nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix);
			
			%define semi-parametric estimator
			semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			
			
			% Simulation
			for s_sampleSetIndex=1:size(v_sampleSetSize,2)
				
				
				s_numberOfSamples=v_sampleSetSize(s_sampleSetIndex);
				%sample
				sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
				m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
				[m_samples,m_positions] = sampler.sample(m_graphFunction);
				%estimate
				m_graphFunctionEstimateBL1 = bandlimitedGraphFunctionEstimator1.estimate(m_samples,m_positions);
				m_graphFunctionEstimateBL2= bandlimitedGraphFunctionEstimator2.estimate(m_samples,m_positions);
				m_graphFunctionEstimateNP=nonParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
				m_graphFunctionEstimateSP=semiParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
				m_graphFunctionEpsilonInsesitiveEstimateSP=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samples,m_positions,s_lambda,s_epsilon);
				
				% Performance assessment
				m_meanSquaredError(s_sampleSetIndex,1) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL1,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,2) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL2,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,3) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateNP,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,4) = SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSP,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,5) = SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEpsilonInsesitiveEstimateSP,m_graphFunction);
				
			end
			%m_meanSquaredError(m_meanSquaredError>1)=1;
			%save('real.mat');
			
			F = F_figure('X',v_sampleSetSize,'Y',m_meanSquaredError','xlab','Number of observed vertices (S)','ylab','NMSE','leg',{strcat('Bandlimited  ',sprintf(' W=%g',s_bandwidth1)),strcat('Bandlimited',sprintf(' W=%g',s_bandwidth2)),'Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)'});
		end
		%Goal: count only error on unseen data
		%     generalization capabibilities 
		function F = compute_fig_1101(obj,niter)
			
			%0. define parameters
			
			s_sigma=2.0;
			s_numberOfClusters=5;
			s_lambda=10^-6;
			s_monteCarloSimulations=niter;%100;
			s_bandwidth1=30;
			s_bandwidth2=35;
			s_SNR=1000;
			s_beta=0.002;
			s_alpha=0.00005;
			s_niter=10;
			s_epsilon=0;
			s_functionTypes=6;
			v_sampleSetSize=(0.1:0.1:1);
		
			m_meanSquaredError=zeros(size(v_sampleSetSize,2),s_functionTypes);
			% define graph
			[Ho,Mo,Alto,Hn,Mn,Altn] = readTemperatureDataset;
			%tic
			m_constraintLaplacian=zeros(size(Ho,1));
			%m_constraintLaplacian(4:15,1)=1;
			graphGenerator = GraphLearningSmoothSignalGraphGenerator('m_observed',Ho,'s_niter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta,'m_constraintLaplacian',m_constraintLaplacian);
			%toc
			
			% tic
			% graphGenerator=SmoothSignalGraphGenerator('m_observed',Ho,'s_maxIter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta);
			% toc
			
			graph = graphGenerator.realization;
			%graph1 = graphGenerator1.realization;
			%L1=graph.getLaplacian
			%L2=graph1.getLaplacian
			v_sampleSetSize=round(v_sampleSetSize*graph.getNumberOfVertices);
			
			
			m_basis= SemiParametricSimulations.parametricPartForTempData(graph.getLaplacian,Alto,s_numberOfClusters);
			
			%functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth);
			%functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
			%signal
			v_realSignal=Hn(:,3);
			functionGenerator=RealDataGraphFunctionGenerator('graph',graph,'v_realSignal',v_realSignal,'s_normalize',1);
			% define bandlimited function estimator
			%m_laplacianEigenvectors=(graph.getLaplacianEigenvectors);
			bandlimitedGraphFunctionEstimator1 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth1);
			bandlimitedGraphFunctionEstimator2 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth2);
			
			% define Kernel function
			diffusionGraphKernel = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigma);
			%define non-parametric estimator
			nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'s_lambda',0.01);
			nonParametricGraphFunctionEstimator.s_regularizationParameter=[10^-10,10^-9,10^-8,10^-7,10^-6,10^-5,10^-4,10^-3,10^-2,10^-1];
			%define semi-parametric estimator
			semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			%define parametric estimator=
			parametricGraphFunctionEstimator=ParametricGraphFunctionEstimator('m_basis',m_basis);
			
			% Simulation
			%dont sample all the vertices no point
			for s_sampleSetIndex=1:(size(v_sampleSetSize,2)-1)
				
				
				s_numberOfSamples=v_sampleSetSize(s_sampleSetIndex);
				%sample
				sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
				m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
				[m_samples,m_positions] = sampler.sample(m_graphFunction);
				
				%estimate
				m_graphFunctionEstimateBL1 = bandlimitedGraphFunctionEstimator1.estimate(m_samples,m_positions);
				m_graphFunctionEstimateBL2= bandlimitedGraphFunctionEstimator2.estimate(m_samples,m_positions);
				m_graphFunctionEstimateNP=nonParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
				m_graphFunctionEstimateSP=semiParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
				m_graphFunctionEpsilonInsesitiveEstimateSP=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samples,m_positions,s_lambda,s_epsilon);
				m_graphFunctionEstimateP=parametricGraphFunctionEstimator.estimate(m_samples,m_positions);
				% Performance assessment
				m_indicator=SemiParametricSimulations.createIndicatorMatrix(m_graphFunction,m_positions);
				m_meanSquaredError(s_sampleSetIndex,1) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_indicator.*m_graphFunctionEstimateBL1,m_indicator.*m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,2) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_indicator.*m_graphFunctionEstimateBL2,m_indicator.*m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,3) = SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_indicator.*m_graphFunctionEstimateP,m_indicator.*m_graphFunction);

				m_meanSquaredError(s_sampleSetIndex,4) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_indicator.*m_graphFunctionEstimateNP,m_indicator.*m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,5) = SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_indicator.*m_graphFunctionEstimateSP,m_indicator.*m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,6) = SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_indicator.*m_graphFunctionEpsilonInsesitiveEstimateSP,m_indicator.*m_graphFunction);
			end
			%m_meanSquaredError(m_meanSquaredError>1)=1;
			%save('real.mat');
			
			F = F_figure('X',v_sampleSetSize,'Y',m_meanSquaredError','xlab','Number of sampled vertices (S)','ylab','NMSE','leg',{strcat('Bandlimited  ',sprintf(' W=%g',s_bandwidth1)),strcat('Bandlimited',sprintf(' W=%g',s_bandwidth2)),'Parametric','Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)'});
			F.ylimit=[0 1];
			F.xlimit=[10 80];
			F.styles = {'-','--','-o','-x','--^','-*'};
			F.pos=[680 729 509 249];
		end
		%Goal: add another month as outlier to the data
		%      outlier detection l1 norm and measuere error on unseen data
		function F = compute_fig_1201(obj,niter)
			
			%0. define parameters
			s_sigma=0.72;
			s_numberOfClusters=5;
			s_lambda=10^-5;
			s_monteCarloSimulations=niter;%100;
			s_bandwidth1=10;
			s_bandwidth2=20;
			s_SNR=1000;
			s_beta=0.02;
			s_alpha=0.00005;
			s_niter=10;
			s_epsilon=0.2;
			s_functionTypes=5;
			v_sampleSetSize=(0.1:0.1:1);
		
			m_meanSquaredError=zeros(size(v_sampleSetSize,2),s_functionTypes);
			% define graph
			[Ho,Mo,Alto,Hn,Mn,Altn] = readTemperatureDataset;
			%tic
			m_constraintLaplacian=zeros(size(Ho,1));
			%m_constraintLaplacian(4:15,1)=1;
			graphGenerator = GraphLearningSmoothSignalGraphGenerator('m_observed',Ho,'s_niter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta,'m_constraintLaplacian',m_constraintLaplacian);
			%toc
			
			% tic
			% graphGenerator=SmoothSignalGraphGenerator('m_observed',Ho,'s_maxIter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta);
			% toc
			
			graph = graphGenerator.realization;
			%graph1 = graphGenerator1.realization;
			%L1=graph.getLaplacian
			%L2=graph1.getLaplacian
			v_sampleSetSize=round(v_sampleSetSize*graph.getNumberOfVertices);
			
			
			m_basis= SemiParametricSimulations.parametricPartForTempData(graph.getLaplacian,Alto,s_numberOfClusters);
			
			%functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth);
			%functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
			%signal
			v_realSignal=Hn(:,1);
			v_outlierSignal=Hn(:,8);
			functionGenerator=RealDataGraphFunctionGenerator('graph',graph,'v_realSignal',v_realSignal,'s_normalize',1);
			outlyingFunctionGenerator=RealDataGraphFunctionGenerator('graph',graph,'v_realSignal',v_outlierSignal,'s_normalize',1);
			% define bandlimited function estimator
			%m_laplacianEigenvectors=(graph.getLaplacianEigenvectors);
			bandlimitedGraphFunctionEstimator1 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth1);
			bandlimitedGraphFunctionEstimator2 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth2);
			
			% define Kernel function
			diffusionGraphKernel = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigma);
			%define non-parametric estimator
			nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix);
			
			%define semi-parametric estimator
			semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			
			
			% Simulation
			%dont sample all the vertices no point
			for s_sampleSetIndex=1:(size(v_sampleSetSize,2)-1)
				
				
				s_numberOfSamples=v_sampleSetSize(s_sampleSetIndex);
				%sample
				s_numberOfOutliers=round(s_numberOfSamples/8);
				
				sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
				m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
				m_outlyingFunction = outlyingFunctionGenerator.realization(s_monteCarloSimulations);
			
				[m_samples,m_positions] = sampler.sample(m_graphFunction);
                m_samplesWithOutliers=SemiParametricSimulations.combineFunctionWithOutliers(m_graphFunction,m_outlyingFunction,m_samples,m_positions,s_numberOfOutliers);

				
				%estimate
				m_graphFunctionEstimateBL1 = bandlimitedGraphFunctionEstimator1.estimate(m_samplesWithOutliers,m_positions);
				m_graphFunctionEstimateBL2= bandlimitedGraphFunctionEstimator2.estimate(m_samplesWithOutliers,m_positions);
				m_graphFunctionEstimateNP=nonParametricGraphFunctionEstimator.estimate(m_samplesWithOutliers,m_positions,s_lambda);
				m_graphFunctionEstimateSP=semiParametricGraphFunctionEstimator.estimate(m_samplesWithOutliers,m_positions,s_lambda);
				m_graphFunctionEpsilonInsesitiveEstimateSP=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samplesWithOutliers,m_positions,s_lambda,s_epsilon);
				
				% Performance assessment
				m_indicator=SemiParametricSimulations.createIndicatorMatrix(m_graphFunction,m_positions);
				m_meanSquaredError(s_sampleSetIndex,1) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_indicator.*m_graphFunctionEstimateBL1,m_indicator.*m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,2) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_indicator.*m_graphFunctionEstimateBL2,m_indicator.*m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,3) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_indicator.*m_graphFunctionEstimateNP,m_indicator.*m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,4) = SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_indicator.*m_graphFunctionEstimateSP,m_indicator.*m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,5) = SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_indicator.*m_graphFunctionEpsilonInsesitiveEstimateSP,m_indicator.*m_graphFunction);
				
			end
			%m_meanSquaredError(m_meanSquaredError>1)=1;
			%save('real.mat');
			
			F = F_figure('X',v_sampleSetSize,'Y',m_meanSquaredError','xlab','Number of observed vertices (S)','ylab','NMSE','leg',{strcat('Bandlimited  ',sprintf(' W=%g',s_bandwidth1)),strcat('Bandlimited',sprintf(' W=%g',s_bandwidth2)),'Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)'});
		end
		
		function F = compute_fig_1002(obj,niter)
			F = obj.load_F_structure(1001);
			F.ylimit=[0 1];
			F.xlimit=[10 89];
			F.styles = {'-','--','-o','-x','--^'};
			F.pos=[680 729 509 249];
			F.leg={strcat('Bandlimited  ',sprintf(' W=10')),strcat('Bandlimited',sprintf(' W=20')),'Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)'};
			
			F.leg_pos = 'north';      % it can be 'northwest',
			%F.leg_pos_vec = [0.547 0.673 0.182 0.114];
		end
		function F = compute_fig_1102(obj,niter)
			F = obj.load_F_structure(1101);
			
			F.styles = {'-','--','-o','-x','--^','-*'};
			F.pos=[680 729 509 249];
			F.leg={strcat('Bandlimited  ',sprintf(' W=8')),strcat('Bandlimited',sprintf(' W=12')),'Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)','Parametric (SL)'};
			
			F.leg_pos = 'north';      % it can be 'northwest',
			%F.leg_pos_vec = [0.547 0.673 0.182 0.114];
		end
		function F = compute_fig_1202(obj,niter)
			F = obj.load_F_structure(1201);
			F.ylimit=[0 1];
			F.xlimit=[10 80];
			F.styles = {'-','--','-o','-x','--^'};
			F.pos=[680 729 509 249];
			F.leg={strcat('Bandlimited  ',sprintf(' W=10')),strcat('Bandlimited',sprintf(' W=20')),'Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)'};
			
			F.leg_pos = 'north';      % it can be 'northwest',
			%F.leg_pos_vec = [0.547 0.673 0.182 0.114];
		end

		%% Synthetic data simulations 
		%  Data used: piece wise constant signals
		%  Goal: methods comparison NMSE
		function F = compute_fig_1003(obj,niter)
						
			%0. define parameters
			s_sigmaDiffusionKernel=0.0005;
            s_sigmaLaplacianInvKernel=10^-10;
			s_numberOfClusters=6;
			s_type=2;
			s_lambda=5*10^-4;
			s_monteCarloSimulations=niter;
			s_bandwidth1=10;
			s_bandwidth2=20;
			s_SNR=10;
			s_dataSetSize=200;
			s_functionTypes=5;
			s_epsilon=1;
            s_weightForBasis=1.2; %weight of parametric part
			v_sampleSetSize=round((0.1:0.1:1)*s_dataSetSize);
			m_meanSquaredError=zeros(size(v_sampleSetSize,2),s_functionTypes);
			% define graph function generator Parametric basis
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability',.6,'s_numberOfVertices',s_dataSetSize);
			graph = graphGenerator.realization;
			
			m_sparseBasis=graph.getClusters(s_numberOfClusters,s_type);
			m_basis=s_weightForBasis*full(m_sparseBasis);
			%m_basis=m_basis*1;
			
			functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth1);
			functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
			
			% define bandlimited function estimator
			bandlimitedGraphFunctionEstimator1 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth1);
			bandlimitedGraphFunctionEstimator2 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth2);
			% define Kernel function
			diffusionGraphKernel = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigmaDiffusionKernel);
			%define non-parametric estimator
			nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix);
			
			%define semi-parametric estimator
			semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			%define semi-parametric epsinlon  insensitive estimator
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
		    parametricGraphFunctionEstimator=ParametricGraphFunctionEstimator('m_basis',m_basis);
            
             %define culp2011 estimator
            rLaplacianReg = @(lambda,epsilon) lambda + epsilon;
            h_rFun_inv = @(lambda) 1./rLaplacianReg(lambda,s_sigmaLaplacianInvKernel);
            laplacianKernel = LaplacianKernel('m_laplacian',graph.getLaplacian,'h_r_inv',{h_rFun_inv});
            semiParametricGraphFunctionEstimatorCulp = SemiParametricGraphFunctionEstimator('m_kernels',laplacianKernel.getKernelMatrix,'m_basis',m_basis);
            
			% Simulation
			for s_sampleSetIndex=1:size(v_sampleSetSize,2)
				
				
				s_numberOfSamples=v_sampleSetSize(s_sampleSetIndex);
				%sample
				sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
				m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
				[m_samples,m_positions] = sampler.sample(m_graphFunction);
				%estimate
				m_graphFunctionEstimateBL1 = bandlimitedGraphFunctionEstimator1.estimate(m_samples,m_positions);
				m_graphFunctionEstimateBL2= bandlimitedGraphFunctionEstimator2.estimate(m_samples,m_positions);
				m_graphFunctionEstimateNP=nonParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
				m_graphFunctionEstimateSP=semiParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
                m_graphFunctionEstimateSPCulp=semiParametricGraphFunctionEstimatorCulp.estimate(m_samples,m_positions,s_lambda);
				%m_graphFunctionEpsilonInsesitiveEstimateSP=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samples,m_positions,s_lambda,s_epsilon);
				m_graphFunctionEstimateP=parametricGraphFunctionEstimator.estimate(m_samples,m_positions);
				% Performance assessment
				m_meanSquaredError(s_sampleSetIndex,1) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL1,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,2) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL2,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,3) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateP,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,4) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateNP,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,6) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSP,m_graphFunction);
				%m_meanSquaredError(s_sampleSetIndex,6) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEpsilonInsesitiveEstimateSP,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,5) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSPCulp,m_graphFunction);
			end
			%m_meanSquaredError(m_meanSquaredError>1)=1;
			%save('synthetic.mat');
            F = F_figure('X',v_sampleSetSize,'Y',m_meanSquaredError','xlab','Number of observed stations (S)','ylab','NMSE','leg',{strcat('LS',sprintf(' B=%g',s_bandwidth1)),strcat('LS',sprintf(' B=%g',s_bandwidth2)),'P','NP','SP','SP-GK'});
            F.ylimit=[0 1];
            %F.logy=1;
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_lambda),...
							sprintf(' diffusion parameter sigma=%g\n',s_sigmaDiffusionKernel),sprintf(' reguralized laplacian parameter sigma=%g\n',s_sigmaLaplacianInvKernel)... s_numberOfClusters
                            ,sprintf(' # clusters=%g\n',s_numberOfClusters),sprintf(' # epsilon=%g\n',s_epsilon),sprintf(' # s_SNR=%g\n',s_SNR)];
        end
		
		function F = compute_fig_1004(obj,niter)
			F = obj.load_F_structure(1003);
			F.ylimit=[0 1];
			F.xlimit=[20 200];
            F.logy=1;
			F.styles = {'-o','-x','-^','--o','--x','--^'};
            F.colorset=[0 0 0;
                0 .7 0;
                .5 .5 0;
                .9 0 .9 ;
                .5 0 1;
                1 0 0];
            F.xlab='Number of observed nodes (S)';
			F.pos=[680 729 509 249];
			%.leg={strcat('Bandlimited  ',sprintf(' W=10')),strcat('Bandlimited',sprintf(' W=20')),'Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)'};
			
			F.leg_pos = 'southwest';      % it can be 'northwest',
			%F.leg_pos_vec = [0.547 0.673 0.182 0.114];
        end
        function F = compute_fig_10041(obj,niter)
			F = obj.load_F_structure(1003);
			F.ylimit=[0 1];
			F.xlimit=[20 200];
            F.logy=1;
            F.Y(5,:)=[];
%             F.leg{1,5}= F.leg{1,6};
			F.styles = {'-o','-x','-^','--o','--^'};
            F.colorset=[0 0 0;
                0 .7 0;
                .5 .5 0;
                .9 0 .9 ;
                %.5 0 1;
                1 0 0];
            F.xlab='Number of observed nodes (S)';
			F.pos=[680 729 509 249];
			%.leg={strcat('Bandlimited  ',sprintf(' W=10')),strcat('Bandlimited',sprintf(' W=20')),'Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)'};
			
			F.leg_pos = 'southwest';      % it can be 'northwest',
			%F.leg_pos_vec = [0.547 0.673 0.182 0.114];
		end
        
		% ADD outlyers 
		function F = compute_fig_1103(obj,niter)
						
			%0. define parameters
			s_sigmaDiffusionKernel=0.0005;
			s_numberOfClusters=6;
			s_type=2;
			s_lambda=10^-5;
			s_outlyingPercentage=0.90;
			s_monteCarloSimulations=niter;
			s_bandwidth1=10;
			s_bandwidth2=20;
			s_SNR=105;
			s_dataSetSize=200;
			s_functionTypes=2;
			s_epsilon=0.0005;
            s_weightParam=1.05;
             
			v_sampleSetSize=round((0.1:0.1:1)*s_dataSetSize);
			m_meanSquaredError=zeros(size(v_sampleSetSize,2),s_functionTypes);
			% define graph function generator Parametric basis
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability',.3,'s_numberOfVertices',s_dataSetSize);
			graph = graphGenerator.realization;
			
			m_sparseBasis=graph.getClusters(s_numberOfClusters,s_type);
			m_basis=s_weightParam*full(m_sparseBasis);
			%m_basis=m_basis*1;
			
			functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth1);
			functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
			m_laplacianEigenvectors=(graph.getLaplacianEigenvectors);
			
			% define bandlimited function estimator
			bandlimitedGraphFunctionEstimator1 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth1);
			bandlimitedGraphFunctionEstimator2 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth2);
			% define Kernel function
			diffusionGraphKernel = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigmaDiffusionKernel);
			%define non-parametric estimator
			nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix);
			
			%define semi-parametric estimator
			semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			%define semi-parametric epsinlon  insensitive estimator
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
		    parametricGraphFunctionEstimator=ParametricGraphFunctionEstimator('m_basis',m_basis);

			% Simulation
			for s_sampleSetIndex=1:size(v_sampleSetSize,2)
				
				
				s_numberOfSamples=v_sampleSetSize(s_sampleSetIndex);
				%sample
				sampler = UniformWithOutliersGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR,'s_outlyingPercentage',s_outlyingPercentage);
				m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
				[m_samples,m_positions] = sampler.sample(m_graphFunction);
				%estimate
% 				m_graphFunctionEstimateBL1 = bandlimitedGraphFunctionEstimator1.estimate(m_samples,m_positions);
% 				m_graphFunctionEstimateBL2= bandlimitedGraphFunctionEstimator2.estimate(m_samples,m_positions);
% 				m_graphFunctionEstimateNP=nonParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
				m_graphFunctionEstimateSP=semiParametricGraphFunctionEstimator.estimateGD(m_samples,m_positions,s_lambda);
				m_graphFunctionEpsilonInsesitiveEstimateSP=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samples,m_positions,s_lambda,s_epsilon);
% 				m_graphFunctionEstimateP=parametricGraphFunctionEstimator.estimate(m_samples,m_positions);
				% Performance assessment
% 				m_meanSquaredError(s_sampleSetIndex,1) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL1,m_graphFunction);
% 				m_meanSquaredError(s_sampleSetIndex,2) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL2,m_graphFunction);
% 				m_meanSquaredError(s_sampleSetIndex,3) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateP,m_graphFunction);
% 				m_meanSquaredError(s_sampleSetIndex,4) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateNP,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,1) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSP,m_graphFunction);
                m_meanSquaredError(s_sampleSetIndex,2) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEpsilonInsesitiveEstimateSP,m_graphFunction);
				
			end
			%m_meanSquaredError(m_meanSquaredError>1)=1;
			%save('synthetic.mat');
			F = F_figure('X',v_sampleSetSize,'Y',m_meanSquaredError','xlab','Number of sampled vertices (S)','ylab','NMSE','leg',{'SP-GK','SP-GK(\epsilon)'});
			%F.ylimit=[0 1];
			F.xlimit=[s_dataSetSize/10 s_dataSetSize+1];
			F.styles = {'--^','--*'};
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_lambda),...
							sprintf(' diffusion parameter sigma=%g\n',s_sigmaDiffusionKernel)... s_numberOfClusters
                            ,sprintf(' # clusters=%g\n',s_numberOfClusters),sprintf(' # epsilon=%g\n',s_epsilon),sprintf(' # s_SNR=%g\n',s_SNR)];
 
			%F.pos=[680 729 509 249];
			
		end
		
		
		function F = compute_fig_1104(obj,niter)
			F = obj.load_F_structure(1103);
			F.logy=1;
			F.xlimit=[20 200];
			F.styles = {'--^','--*'};
            F.colorset=[1 0 0; 0 0 1];

			F.pos=[680 729 509 249];
			%F.leg={strcat('Bandlimited  ',sprintf(' W=10')),strcat('Bandlimited',sprintf(' W=20')),'Parametric','Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)'};
			
			%F.leg_pos = 'northeast';      % it can be 'northwest',
			F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        % ADD outlyers 
		function F = compute_fig_1105(obj,niter)
						
			%0. define parameters
			s_sigmaDiffusionKernel=0.0005;
			s_numberOfClusters=6;
			s_type=2;
			s_lambda1=10^-5;
            s_lambda2=10^-4;
			s_outlyingPercentage=0.90;
			s_monteCarloSimulations=niter;
			s_bandwidth1=10;
			s_bandwidth2=20;
			s_SNRnonoise=10;
            s_SNRoutl=-10;
			s_dataSetSize=200;
			s_functionTypes=4;
			s_epsilon=0.0005;
            s_weightParam=1.05;
             
			v_sampleSetSize=round((0.1:0.1:1)*s_dataSetSize);
			m_meanSquaredError=zeros(size(v_sampleSetSize,2),s_functionTypes);
			% define graph function generator Parametric basis
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability',.3,'s_numberOfVertices',s_dataSetSize);
			graph = graphGenerator.realization;
			
			m_sparseBasis=graph.getClusters(s_numberOfClusters,s_type);
			m_basis=s_weightParam*full(m_sparseBasis);
			%m_basis=m_basis*1;
			
			functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth1);
			functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
			m_laplacianEigenvectors=(graph.getLaplacianEigenvectors);
			
			% define bandlimited function estimator
			bandlimitedGraphFunctionEstimator1 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth1);
			bandlimitedGraphFunctionEstimator2 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth2);
			% define Kernel function
			diffusionGraphKernel = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigmaDiffusionKernel);
			%define non-parametric estimator
			nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix);
			
			%define semi-parametric estimator
			semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			%define semi-parametric epsinlon  insensitive estimator
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
		    parametricGraphFunctionEstimator=ParametricGraphFunctionEstimator('m_basis',m_basis);

			% Simulation
			for s_sampleSetIndex=1:size(v_sampleSetSize,2)
				
				
				s_numberOfSamples=v_sampleSetSize(s_sampleSetIndex);
				%sample
				sampler = UniformWithOutliersGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNRoutl,'s_outlyingPercentage',s_outlyingPercentage);
                samplerNonoise = UniformWithOutliersGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNRnonoise,'s_outlyingPercentage',s_outlyingPercentage);
				m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
				[m_samples,m_positions] = sampler.sample(m_graphFunction);
                [m_samplesNonoise,m_positionsNonoise] = samplerNonoise.sample(m_graphFunction);
				%estimate
% 				m_graphFunctionEstimateBL1 = bandlimitedGraphFunctionEstimator1.estimate(m_samples,m_positions);
% 				m_graphFunctionEstimateBL2= bandlimitedGraphFunctionEstimator2.estimate(m_samples,m_positions);
% 				m_graphFunctionEstimateNP=nonParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
				m_graphFunctionEstimateSP=semiParametricGraphFunctionEstimator.estimateGD(m_samples,m_positions,s_lambda1);
				m_graphFunctionEpsilonInsesitiveEstimateSP=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samples,m_positions,s_lambda1,s_epsilon);
                m_graphFunctionEstimateSPNonoise=semiParametricGraphFunctionEstimator.estimateGD(m_samplesNonoise,m_positionsNonoise,s_lambda2);
				m_graphFunctionEpsilonInsesitiveEstimateSPNonoise=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samplesNonoise,m_positionsNonoise,s_lambda2,s_epsilon);
% 				m_graphFunctionEstimateP=parametricGraphFunctionEstimator.estimate(m_samples,m_positions);
				% Performance assessment
% 				m_meanSquaredError(s_sampleSetIndex,1) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL1,m_graphFunction);
% 				m_meanSquaredError(s_sampleSetIndex,2) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL2,m_graphFunction);
% 				m_meanSquaredError(s_sampleSetIndex,3) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateP,m_graphFunction);
% 				m_meanSquaredError(s_sampleSetIndex,4) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateNP,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,1) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSP,m_graphFunction);
                m_meanSquaredError(s_sampleSetIndex,2) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEpsilonInsesitiveEstimateSP,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,3) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSPNonoise,m_graphFunction);
                m_meanSquaredError(s_sampleSetIndex,4) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEpsilonInsesitiveEstimateSPNonoise,m_graphFunction);
				
			end
			%m_meanSquaredError(m_meanSquaredError>1)=1;
			%save('synthetic.mat');
			F = F_figure('X',v_sampleSetSize,'Y',m_meanSquaredError','xlab','Number of sampled vertices (S)','ylab','NMSE','leg',{sprintf('SP-GK SNR=%g,\mu=%g',s_SNRoutl,s_lambda1),sprintf('SP-GK(\epsilon) SNR=%g,\mu=%g',s_SNRoutl,s_lambda1),...
                sprintf('SP-GK SNR=%g,\mu=%g',s_SNRnonoise,s_lambda2),sprintf('SP-GK(\epsilon) SNR=%g,\mu=%g,',s_SNRnonoise,s_lambda2)});
			%F.ylimit=[0 1];
			F.xlimit=[s_dataSetSize/10 s_dataSetSize+1];
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_lambda1),...
							sprintf(' diffusion parameter sigma=%g\n',s_sigmaDiffusionKernel)... s_numberOfClusters
                            ,sprintf(' # clusters=%g\n',s_numberOfClusters),sprintf(' # epsilon=%g\n',s_epsilon),sprintf(' # s_SNR=%g\n',s_SNRoutl)];
 
			%F.pos=[680 729 509 249];
			
		end
		
		
		function F = compute_fig_1106(obj,niter)
			F = obj.load_F_structure(1105);
			F.logy=1;
			F.xlimit=[20 200];
			F.styles = {'-^','-*','--^','--*'};
			F.pos=[680 729 509 249];
			%F.leg={strcat('Bandlimited  ',sprintf(' W=10')),strcat('Bandlimited',sprintf(' W=20')),'Parametric','Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)'};
			
			%F.leg_pos = 'northeast';      % it can be 'northwest',
			F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
  
        %% Compare GD and Closed form
        %  Data used: piece wise constant signals
		%  Goal: methods comparison NMSE
        function F = compute_fig_1013(obj,niter)
						
			%0. define parameters
			s_sigma=0.0001; %For diffusion kernel
			s_numberOfClusters=4;
			s_type=2;
			s_lambda=10^-3;
			s_monteCarloSimulations=niter;
			s_bandwidth1=10;
			s_bandwidth2=15;
			s_SNR=4;
			s_dataSetSize=10^4;
			s_functionTypes=3;
			s_epsilon=1;
            s_weightForBasis=2; %weight of parametric part
% 			v_sampleSetSize=round((0.1:0.1:1)*s_dataSetSize);
            v_sampleSetSize=round(0.8*s_dataSetSize);
			m_meanSquaredError=zeros(size(v_sampleSetSize,2),s_functionTypes);
			% define graph function generator Parametric basis
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability',.6,'s_numberOfVertices',s_dataSetSize);
			graph = graphGenerator.realization;
			
			m_sparseBasis=graph.getClusters(s_numberOfClusters,s_type);
			m_basis=s_weightForBasis*full(m_sparseBasis);
			%m_basis=m_basis*1;
			
			functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth2+5);
			functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
			m_laplacianEigenvectors=(graph.getLaplacianEigenvectors);
			
			% define bandlimited function estimator
			bandlimitedGraphFunctionEstimator1 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth1);
			bandlimitedGraphFunctionEstimator2 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth2);
			% define Kernel function
			diffusionGraphKernel = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigma);
			%define non-parametric estimator
			nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix);
			
			%define semi-parametric estimator
			semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			%define semi-parametric epsinlon  insensitive estimator
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
		    parametricGraphFunctionEstimator=ParametricGraphFunctionEstimator('m_basis',m_basis);

			% Simulation
			for s_sampleSetIndex=1:size(v_sampleSetSize,2)
				
				
				s_numberOfSamples=v_sampleSetSize(s_sampleSetIndex);
				%sample
				sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
				m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
				[m_samples,m_positions] = sampler.sample(m_graphFunction);
				%estimate
                tic
% 				m_graphFunctionEstimateSPGD=semiParametricGraphFunctionEstimator.estimateGD(m_samples,m_positions,s_lambda);
                s_timeIter=toc
                tic
% 				m_graphFunctionEstimateSPGDArm=semiParametricGraphFunctionEstimator.estimateGDArm(m_samples,m_positions,s_lambda);
                s_timeIterArm=toc
                tic
				m_graphFunctionEstimateSP=semiParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
                s_timeClosedForm=toc

% 				m_meanSquaredError(s_sampleSetIndex,1) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSPGD,m_graphFunction);
%                  m_meanSquaredError(s_sampleSetIndex,2) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSPGDArm,m_graphFunction);
                m_meanSquaredError(s_sampleSetIndex,3) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSP,m_graphFunction);
				
			end
			%m_meanSquaredError(m_meanSquaredError>1)=1;
			%save('synthetic.mat');
			F = F_figure('X',v_sampleSetSize,'Y',m_meanSquaredError','xlab','Number of observed vertices (S)','ylab','NMSE','leg',{'Semiparametric-GD (SL)','Semiparametric-GDArm (SL)','Semiparametric (SL)'});
			%F.ylimit=[0 1];
			F.xlimit=[10 s_dataSetSize];
            F.styles = {'-','--','-*'};
			
		end
		
        
		
		
		%% Real data simulations
		%  Data used: Swiss temperature
		%  Goal: epsilon tuning
		function F = compute_fig_1005(obj,niter)
			%choose epsilon
			%0. define parameters
			s_sigma=1.3;
			s_numberOfClusters=5;
			s_lambda=10^-5;
			s_monteCarloSimulations=niter;
			s_bandwidth1=10;
			s_bandwidth2=20;
			s_SNR=1000;
			
			s_beta=0.02;
			s_alpha=0.005;
			s_niter=10;
			v_epsilonSet=[0,0.001,0.005,0.01,0.05,0.1,0.5,1,5];
			s_functionTypes=6;
			s_normalize=1;
			v_sampleSetSize=(0.5);
			
			m_meanSquaredError=zeros(size(v_sampleSetSize,2),s_functionTypes);
			% define graph
			[Ho,Mo,Alto,Hn,Mn,Altn] = readTemperatureDataset;
			%tic
			graphGenerator = GraphLearningSmoothSignalGraphGenerator('m_observed',Ho,'s_niter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta,'m_missingValuesIndicator',[]);
			%toc
			
			% tic
			% graphGenerator=SmoothSignalGraphGenerator('m_observed',Ho,'s_maxIter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta);
			% toc
			
			graph = graphGenerator.realization;
			%graph1 = graphGenerator1.realization;
			%L1=graph.getLaplacian
			%L2=graph1.getLaplacian
			v_sampleSetSize=round(v_sampleSetSize*graph.getNumberOfVertices);
			
			
			m_basis= SemiParametricSimulations.parametricPartForTempData(graph.getLaplacian,Alto,s_numberOfClusters);
			
			%functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth);
			%functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
			%signal
			v_realSignal=Hn(:,3);
			functionGenerator=RealDataGraphFunctionGenerator('graph',graph,'v_realSignal',v_realSignal,'s_normalize',s_normalize);
			% define bandlimited function estimator
			%m_laplacianEigenvectors=(graph.getLaplacianEigenvectors);
			%bandlimitedGraphFunctionEstimator1 = BandlimitedGraphFunctionEstimator('m_laplacianEigenvectors',m_laplacianEigenvectors(:,1:s_bandwidth1));
			%bandlimitedGraphFunctionEstimator2 = BandlimitedGraphFunctionEstimator('m_laplacianEigenvectors',m_laplacianEigenvectors(:,1:(s_bandwidth2)));
			
			% define Kernel function
			diffusionGraphKernel = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigma);
			%define non-parametric estimator
			%nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix);
			
			%define semi-parametric estimator
			%semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			
			
			% Simulation
			for s_sampleSetIndex=1:size(v_sampleSetSize,2)
				for s_epsilonSetIndex=1:size(v_epsilonSet,2)
					
					s_numberOfSamples=v_sampleSetSize(s_sampleSetIndex);
					%sample
					sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
					m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
					[m_samples,m_positions] = sampler.sample(m_graphFunction);
					%estimate
					m_graphFunctionEpsilonInsesitiveEstimateSP=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samples,m_positions,s_lambda,v_epsilonSet(s_epsilonSetIndex));
					m_meanSquaredError(s_sampleSetIndex,s_epsilonSetIndex) = SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEpsilonInsesitiveEstimateSP,m_graphFunction);
				end
			end
			%m_meanSquaredError(m_meanSquaredError>1)=1;
			%save('real.mat');
			
			F = F_figure('X',v_epsilonSet,'Y',m_meanSquaredError,'xlab','epsilon parameter','ylab','NMSE','leg',cellstr(num2str(v_sampleSetSize(:))),'logx',1);
		end
		
		%% Synthetic data simulations 
		%  Data used: piece wise constant signals
		% Goal: epsilon tuning
		function F = compute_fig_1007(obj,niter)
			%choose epsilon
			%0. define parameters
			s_sigma=1.3;
			s_numberOfClusters=5;
			s_lambda=10^-5;
			s_monteCarloSimulations=niter;
			s_bandwidth1=10;
			s_bandwidth2=20;
			s_SNR=6;
			s_type=2;
			s_beta=0.02;
			s_dataSetSize=100;
			s_alpha=0.005;
			s_niter=10;
			v_epsilonSet=[0,0.001,0.005,0.01,0.05,0.1,0.5,1,5];
			s_functionTypes=6;
			v_sampleSetSize=(0.5);
			
			m_meanSquaredError=zeros(size(v_sampleSetSize,2),s_functionTypes);
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability',.6,'s_numberOfVertices',s_dataSetSize);
			graph = graphGenerator.realization;
			
			m_sparseBasis=graph.getClusters(s_numberOfClusters,s_type);
			m_basis=1.5*full(m_sparseBasis);
			%m_basis=m_basis*1;
			
			functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth1);
			functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
			
			v_sampleSetSize=round(v_sampleSetSize*graph.getNumberOfVertices);
			
			
			% define Kernel function
			diffusionGraphKernel = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigma);
			%define non-parametric estimator
			%nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix);
			
			%define semi-parametric estimator
			%semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			
			
			% Simulation
			for s_sampleSetIndex=1:size(v_sampleSetSize,2)
				for s_epsilonSetIndex=1:size(v_epsilonSet,2)
					
					s_numberOfSamples=v_sampleSetSize(s_sampleSetIndex);
					%sample
					sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
					m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
					[m_samples,m_positions] = sampler.sample(m_graphFunction);
					%estimate
					m_graphFunctionEpsilonInsesitiveEstimateSP=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samples,m_positions,s_lambda,v_epsilonSet(s_epsilonSetIndex));
					m_meanSquaredError(s_sampleSetIndex,s_epsilonSetIndex) = SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEpsilonInsesitiveEstimateSP,m_graphFunction);
				end
			end
			%m_meanSquaredError(m_meanSquaredError>1)=1;
			%save('real.mat');
			
			F = F_figure('X',v_epsilonSet,'Y',m_meanSquaredError,'xlab','epsilon parameter','ylab','NMSE','leg',cellstr(num2str(v_sampleSetSize(:))),'logx',1);
        end
        
       
        
        
        
        
        
        
        
        %% Real data simulations
		%  Data used: Swiss temperature
		%  Goal: Plot the gap for the epsilon insensitive function
		%  Data taken from CVX output just ploted here
		function F = compute_fig_1009(obj,niter)
			gap=[
				3.20E+04
				3.80E+02
				2.70E+01
				7.90E-01
				1.40E-02
				1.60E-04
				9.00E-06
				8.90E-07
				1.50E-08
				4.40E-10
				];
			F = F_figure('X',(1:size(gap,1)),'Y',gap','xlab','number of iterations','ylab','dual-primal gap','leg','Semi-parametric (\epsilon-IL)','logy',1);
			F.pos=[680 729 509 249];
		end
		
		
		
		
		%% Real data simulations
		%  Data used: Movielens Data
		%  Goal: compare the NMSE error for the known etries..
		%  IS NOT CORRECT VERSION DOES NOT FIND THE RIGHT NMSE
		% IMPLEMENT AGAIN HIDE ENTRIES
		function F=compute_fig_2001(obj,niter)
			%Movielens Simmulation
			%NMSE
			%0. define parameters
			s_sigma=0.12;
			s_lambda=10^-11;
			s_monteCarloSimulations=niter;
			s_bandwidth1=10;
			s_bandwidth2=5;
			s_fontSize=35;
			s_normalize=0;
			s_SNR=100;
			s_epsilon=0.5;
			s_functionTypes=5;
			s_type=2;
			v_sampleSetSize=(0.1:0.1:0.9);
			%specify number of items/signals
			s_totalItems=10;
			%# clusters for spectral
			s_numberOfClusters=5;
			
			m_meanSquaredErrorInner=zeros(size(v_sampleSetSize,2),s_functionTypes);
			m_meanSquaredErrorOuter=zeros(size(v_sampleSetSize,2),s_functionTypes);
			
			% define graph
			[m_clustUser,m_clustUserInfoAge,m_clustUserInfoSex,m_clustUserInfoOccup,m_train,m_test,c_userInfo,c_movieInfo]=prepareMLdat;
			m_adjacency= SemiParametricSimulations.generateAdjancency(m_train);
			graph = Graph('m_adjacency',m_adjacency);
			%define basis
			%m_basis=m_clustUserInfoOccup;
			m_basis=full(graph.getClusters(s_numberOfClusters,s_type));
			% define bandlimited function estimator
			m_laplacianEigenvectors=(graph.getLaplacianEigenvectors);
			bandlimitedGraphFunctionEstimator1 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth1);
			bandlimitedGraphFunctionEstimator2 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth2);
			
			% define Kernel function
			%%Laplacian gives disconnected components must fix to proceeed maybe have
			%%different graphs.
			kernelgraphDiffusion = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigma);
			%define non-parametric estimator
			nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',kernelgraphDiffusion.generateKernelMatrix);
			
			%define semi-parametric estimator
			semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',kernelgraphDiffusion.generateKernelMatrix,'m_basis',m_basis);
			
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',kernelgraphDiffusion.generateKernelMatrix,'m_basis',m_basis);
			% Simulation
			
			for s_itemIndex=1:s_totalItems
				%Should sample only from the know values...
				
				v_realSignal=m_test(:,s_itemIndex);
				%indicator matrix so that only the known values are
				%considered for NMSE and signal estimation
				v_knownValuesInd=v_realSignal~=0;
				s_knownValues=nnz(v_realSignal);
				m_knownValuesInd=repmat(v_knownValuesInd, 1,s_monteCarloSimulations);
				v_sampleSetSizeForItem=round(v_sampleSetSize*s_knownValues);
				
				
				
				%functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth);
				%functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
				%signal
				functionGenerator=RealDataGraphFunctionGenerator('graph',graph,'v_realSignal',v_realSignal,'s_normalize',s_normalize);
				
				
				
				
				for s_sampleSetIndex=1:size(v_sampleSetSizeForItem,2)
					
					s_numberOfSamples=v_sampleSetSizeForItem(s_sampleSetIndex);
					% % % %     if(s_numberOfSamples>s_monteCarloSimulations)
					% % % %         s_monteCarloSimulations=1;
					% % % %     end
					%sample
					sampler = PartiallyObservedGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR,'v_knownValuesInd',v_knownValuesInd);
					m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
					[m_samples,m_positions] = sampler.sample(m_graphFunction);
					%estimate
					m_graphFunctionEstimateBL1 = bandlimitedGraphFunctionEstimator1.estimate(m_samples,m_positions);
					m_graphFunctionEstimateBL2= bandlimitedGraphFunctionEstimator2.estimate(m_samples,m_positions);
					m_graphFunctionEstimateNP=nonParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
					m_graphFunctionEstimateSP=semiParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
					m_graphFunctionEpsilonInsesitiveEstimateSP=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samples,m_positions,s_lambda,s_epsilon);
					
					% Performance assessment
					%compute the mse error only among the known entries
					m_meanSquaredErrorInner(s_sampleSetIndex,1) =SemiParametricSimulations.estimateRootMeanSquaredError(m_graphFunctionEstimateBL1.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd,s_knownValues);
					m_meanSquaredErrorInner(s_sampleSetIndex,2) =SemiParametricSimulations.estimateRootMeanSquaredError(m_graphFunctionEstimateBL2.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd,s_knownValues);
					m_meanSquaredErrorInner(s_sampleSetIndex,3) =SemiParametricSimulations.estimateRootMeanSquaredError(m_graphFunctionEstimateNP.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd,s_knownValues);
					m_meanSquaredErrorInner(s_sampleSetIndex,4) =SemiParametricSimulations.estimateRootMeanSquaredError(m_graphFunctionEstimateSP.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd,s_knownValues);
					m_meanSquaredErrorInner(s_sampleSetIndex,5) =SemiParametricSimulations.estimateRootMeanSquaredError(m_graphFunctionEpsilonInsesitiveEstimateSP.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd,s_knownValues);
				end
				m_meanSquaredErrorOuter=m_meanSquaredErrorOuter+m_meanSquaredErrorInner;
			end
			m_meanSquaredErrorOuter=(1/s_totalItems)*m_meanSquaredErrorOuter;
			F = F_figure('X',100*v_sampleSetSize,'Y',m_meanSquaredErrorOuter','xlab','Percentage of observed vertices (S)','ylab','NMSE','tit',sprintf('#Items=%g',s_totalItems),'leg',{strcat('Bandlimited  ',sprintf(' W=%g',s_bandwidth1)),strcat('Bandlimited',sprintf(' W=%g',s_bandwidth2)),'Nonparametric (SL)','Semiparametric (SL)','Semiparametric (\epsilon-IL)'});
			
			
		end
		function F = compute_fig_2002(obj,niter)
			F = obj.load_F_structure(2001);
			F.ylimit=[0 10];
			F.xlimit=[10 100];
			F.styles = {'-','--','-o','-x','--^'};
			F.pos=[680 729 509 249];
			
			%F.leg_pos = 'northeast';      % it can be 'northwest',
			F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
        
        
        
        
		%% Real data simulations
		%  Data used: Movielens Data Thanos
		%  Goal: compare the NMSE error for the known etries..
        function F=compute_fig_2003(obj,niter)
			%Movielens Simmulation
			%NMSE
			%0. define parameters
			s_sigma=0.001;
			s_lambda=10^-6;
			s_monteCarloSimulations=niter;
			s_bandwidth1=2;
			s_bandwidth2=3;
			s_fontSize=35;
			s_normalize=0;
			s_SNR=Inf;
			s_epsilon=0.5;
			s_functionTypes=5;
			s_type=2;
			%v_sampleSetSize=(0.7:0.1:0.9);
			%specify number of items/signals
			s_totalUsers=100;
			%# clusters for spectral
			s_numberOfClusters=5;
			v_meanSquaredErrorInner=zeros(s_functionTypes,1);
			v_meanSquaredErrorOuter=zeros(s_functionTypes,1);
			
			% define graph
			[m_test,m_adjacencyItem,m_adjacencyUser,m_ratings,m_userFeatures,m_moviesFeatures] = readRSMovielensDataset;
			m_adjacency= m_adjacencyItem;
			graph = Graph('m_adjacency',m_adjacency);
			%define basis
			%m_basis=m_clustUserInfoOccup;
			m_basis=m_moviesFeatures;
			% define bandlimited function estimator
			bandlimitedGraphFunctionEstimator1 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth1);
			bandlimitedGraphFunctionEstimator2 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth2);
			
			% define Kernel function
			%%Laplacian gives disconnected components must fix to proceeed maybe have
			%%different graphs.
			kernelgraphDiffusion = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigma);
            m_kernelGraphDiffusion=kernelgraphDiffusion.generateKernelMatrix;
			%define non-parametric estimator
			nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',m_kernelGraphDiffusion);
			
			%define semi-parametric estimator
			semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',m_kernelGraphDiffusion,'m_basis',m_basis);
			
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',m_kernelGraphDiffusion,'m_basis',m_basis);
			% Simulation
            if(s_monteCarloSimulations~=1)
                error('not supported for multiple realizations');
            end
			m_graphFunctionEstimateSP=zeros(size(m_adjacencyItem,1),s_totalUsers);
            tic
			for s_indUser=1:s_totalUsers
				%Should sample only from the know values...
				
				v_realSignal=m_ratings(:,s_indUser);
				%indicator matrix so that only the known values are
				%considered for NMSE and signal estimation
				v_knownValuesInd=v_realSignal~=0;
				s_knownValues=nnz(v_realSignal);
				%m_knownValuesInd=repmat(v_knownValuesInd, 1,s_monteCarloSimulations);
				
                
                
                %functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth);
                %functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
                %signal
                functionGenerator=RealDataGraphFunctionGenerator('graph',graph,'v_realSignal',v_realSignal,'s_normalize',s_normalize);

                s_numberOfSamples=s_knownValues;
                % % % %     if(s_numberOfSamples>s_monteCarloSimulations)
                % % % %         s_monteCarloSimulations=1;
                % % % %     end
                %sample
                sampler = PartiallyObservedGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR,'v_knownValuesInd',v_knownValuesInd);
                m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
                [m_samples,m_positions] = sampler.sample(m_graphFunction);
                %estimate
                %m_graphFunctionEstimateBL1 = bandlimitedGraphFunctionEstimator1.estimate(m_samples,m_positions);
                %m_graphFunctionEstimateBL2= bandlimitedGraphFunctionEstimator2.estimate(m_samples,m_positions);
                %m_graphFunctionEstimateNP=nonParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
                m_graphFunctionEstimateSP(:,s_indUser)=semiParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
                %m_graphFunctionEpsilonInsesitiveEstimateSP=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samples,m_positions,s_lambda,s_epsilon);
                
                % Performance assessment
                %compute the mse error only among the known entries
                %v_meanSquaredErrorInner(1) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL1.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd);
                %v_meanSquaredErrorInner(2) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL2.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd);
                %v_meanSquaredErrorInner(3) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateNP.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd);
                %v_meanSquaredErrorInner(4) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSP.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd);
                %m_meanSquaredErrorInner(s_sampleSetIndex,5) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEpsilonInsesitiveEstimateSP.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd);
                
                %v_meanSquaredErrorOuter=v_meanSquaredErrorOuter+v_meanSquaredErrorInner;
			end
			%v_meanSquaredErrorOuter=(1/s_totalUsers)*v_meanSquaredErrorOuter;
            s_totalTime=toc
			F =[];
%             F_figure('X',100*v_sampleSetSize,'Y',v_meanSquaredErrorOuter','xlab','Percentage of observed vertices (S)','ylab','NMSE','tit',sprintf('#Items=%g',s_totalUsers),'leg',{strcat('Bandlimited  ',sprintf(' W=%g',s_bandwidth1)),strcat('Bandlimited',sprintf(' W=%g',s_bandwidth2)),'Nonparametric (SL)','Semiparametric (SL)','Semiparametric (\epsilon-IL)'});
% 			F.ylimit=[0 0.5];
% 			F.xlimit=[10 100];
% 			F.styles = {'-','--','-o','-x','--^'};
			
		end
		function F = compute_fig_2004(obj,niter)
			F = obj.load_F_structure(2003);
			F.ylimit=[0 1];
			F.xlimit=[10 100];
			F.styles = {'-','--','-o','-x','--^'};
			F.pos=[680 729 509 249];
			
			%F.leg_pos = 'northeast';      % it can be 'northwest',
			F.leg_pos_vec = [0.647 0.683 0.182 0.114];
        end
        
         
        
        
        
        
        %% Real data simulations
		%  Data used: Jensen joke Data
		%  Goal: compare the NMSE error for the known etries..
		%  WRONG VERSION DELETE GO TO 3003
		function F=compute_fig_3001(obj,niter)
			%Jensen Simmulation
			%NMSE
			%0. define parameters
			s_sigma=0.12;
			s_lambda=10^-11;
			s_monteCarloSimulations=niter;
			s_bandwidth1=10;
			s_bandwidth2=5;
			s_fontSize=35;
			s_normalize=0;
			s_SNR=100;
			s_epsilon=0.5;
			s_functionTypes=5;
			s_type=2;
			v_sampleSetSize=(0.8:0.1:0.9);
			%specify number of items/signals
			s_totalItems=3;
			%# clusters for spectral
			s_numberOfClusters=10;
			
			m_meanSquaredErrorInner=zeros(size(v_sampleSetSize,2),s_functionTypes);
			m_meanSquaredErrorOuter=zeros(size(v_sampleSetSize,2),s_functionTypes);
			
			% define graph
			m_reducedRatings=prepareJokerdat;
			%represent different maybe 0 rated jokes..
			m_reducedRatings((m_reducedRatings==99))=0;
			
			m_adjacency= SemiParametricSimulations.generateAdjancencyJensen(m_reducedRatings);
			graph = Graph('m_adjacency',m_adjacency);
			%define basis
			%m_basis=m_clustUserInfoOccup;
			m_basis=full(graph.getClusters(s_numberOfClusters,s_type));
			% define bandlimited function estimator
			m_laplacianEigenvectors=(graph.getLaplacianEigenvectors);
			bandlimitedGraphFunctionEstimator1 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth1);
			bandlimitedGraphFunctionEstimator2 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth2);
			
			% define Kernel function
			%%Laplacian gives disconnected components must fix to proceeed maybe have
			%%different graphs.
			kernelgraphDiffusion = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigma);
			%define non-parametric estimator
			nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',kernelgraphDiffusion.generateKernelMatrix);
			
			%define semi-parametric estimator
			semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',kernelgraphDiffusion.generateKernelMatrix,'m_basis',m_basis);
			
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',kernelgraphDiffusion.generateKernelMatrix,'m_basis',m_basis);
			% Simulation
			
			for s_itemIndex=1:s_totalItems
				%Should sample only from the know values...
				
				v_realSignal=m_reducedRatings(:,s_itemIndex);
				%indicator matrix so that only the known values are
				%considered for NMSE and signal estimation
				v_knownValuesInd=v_realSignal~=0;
				s_knownValues=nnz(v_realSignal);
				m_knownValuesInd=repmat(v_knownValuesInd, 1,s_monteCarloSimulations);
				v_sampleSetSizeForItem=round(v_sampleSetSize*s_knownValues);
				
				
				
				%functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth);
				%functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
				%signal
				functionGenerator=RealDataGraphFunctionGenerator('graph',graph,'v_realSignal',v_realSignal,'s_normalize',s_normalize);
				
				
				
				
				for s_sampleSetIndex=1:size(v_sampleSetSizeForItem,2)
					
					s_numberOfSamples=v_sampleSetSizeForItem(s_sampleSetIndex);
					% % % %     if(s_numberOfSamples>s_monteCarloSimulations)
					% % % %         s_monteCarloSimulations=1;
					% % % %     end
					%sample
					sampler = PartiallyObservedGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR,'v_knownValuesInd',v_knownValuesInd);
					m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
					[m_samples,m_positions] = sampler.sample(m_graphFunction);
					%estimate
					m_graphFunctionEstimateBL1 = bandlimitedGraphFunctionEstimator1.estimate(m_samples,m_positions);
					m_graphFunctionEstimateBL2= bandlimitedGraphFunctionEstimator2.estimate(m_samples,m_positions);
					m_graphFunctionEstimateNP=nonParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
					m_graphFunctionEstimateSP=semiParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
					m_graphFunctionEpsilonInsesitiveEstimateSP=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samples,m_positions,s_lambda,s_epsilon);
					
					% Performance assessment
					%compute the mse error only among the known entries
					m_meanSquaredErrorInner(s_sampleSetIndex,1) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL1.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd);
					m_meanSquaredErrorInner(s_sampleSetIndex,2) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL2.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd);
					m_meanSquaredErrorInner(s_sampleSetIndex,3) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateNP.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd);
					m_meanSquaredErrorInner(s_sampleSetIndex,4) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSP.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd);
					m_meanSquaredErrorInner(s_sampleSetIndex,5) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEpsilonInsesitiveEstimateSP.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd);
				end
				m_meanSquaredErrorOuter=m_meanSquaredErrorOuter+m_meanSquaredErrorInner;
			end
			m_meanSquaredErrorOuter=(1/s_totalItems)*m_meanSquaredErrorOuter;
			F = F_figure('X',100*v_sampleSetSize,'Y',m_meanSquaredErrorOuter','xlab','Percentage of observed vertices (S)','ylab','NMSE','tit',sprintf('#users=%g',s_totalItems),'leg',{strcat('Bandlimited  ',sprintf(' W=%g',s_bandwidth1)),strcat('Bandlimited',sprintf(' W=%g',s_bandwidth2)),'Nonparametric (SL)','Semiparametric (SL)','Semiparametric (\epsilon-IL)'});
			
			
		end
		
		function F = compute_fig_3002(obj,niter)
			F = obj.load_F_structure(3001);
			F.ylimit=[0 10];
			F.xlimit=[10 100];
			F.tit=sprintf('#users=%g',500);
			F.styles = {'-','--','-o','-x','--^'};
			F.pos=[680 729 509 249];
			
			%F.leg_pos = 'northeast';      % it can be 'northwest',
			F.leg_pos_vec = [0.647 0.683 0.182 0.114];
		end
		%% Real data simulations 
		%  Data used: Jensen joke Data
		%  Goal: compare the NMSE error excluding 2 jokes for each user
		function F=compute_fig_3003(obj,niter)
			%Jensen Simmulation
			%NMSE
			%excluding 2 jokes for each user
			%over these 2 I should do the monte carlos..
			%0. define parameters
			%TODO compute the RMSE online the test data..
			s_sigma=0.08;
			s_lambda=10^-11;
			s_monteCarloSimulations=niter;
			s_bandwidth1=2;
			s_bandwidth2=3;
			s_normalize=0;
			s_SNR=100;
			s_sigmaAdj=1;
			s_epsilon=0.5;
			s_functionTypes=5;
			s_type=3;
			%v_sampleSetSize=(0.8:0.1:0.9);
			
			%specify number of items/signals
			s_totalItems=100;
			%# clusters for spectral
			s_numberOfClusters=20;
			
			m_meanSquaredErrorInner=zeros(1,s_functionTypes);
			m_rootMeanSquaredErrorOuter=zeros(1,s_functionTypes);
			m_rootMeanSquaredErrorOuterOuter=zeros(1,s_functionTypes);
			% define graph
			m_reducedRatings=prepareJokerdat;
			%represent different maybe 0 rated jokes..
			m_reducedRatings((m_reducedRatings==99))=0;
			
			m_adjacency= SemiParametricSimulations.generateAdjancencyJensen(m_reducedRatings,s_sigmaAdj);
			graph = Graph('m_adjacency',m_adjacency);
			%define basis
			%m_basis=m_clustUserInfoOccup;
			m_basis=full(graph.getClusters(s_numberOfClusters,s_type));
			% define bandlimited function estimator
			m_laplacianEigenvectors=(graph.getLaplacianEigenvectors);
			bandlimitedGraphFunctionEstimator1 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth1);
			bandlimitedGraphFunctionEstimator2 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth2);
			
			% define Kernel function
			%%Laplacian gives disconnected components must fix to proceeed maybe have
			%%different graphs.
			kernelgraphDiffusion = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigma);
			%define non-parametric estimator
			nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',kernelgraphDiffusion.generateKernelMatrix);
			
			%define semi-parametric estimator
			semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',kernelgraphDiffusion.generateKernelMatrix,'m_basis',m_basis);
			
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',kernelgraphDiffusion.generateKernelMatrix,'m_basis',m_basis);
			% Simulation
			%randomly exclude 2 measurments from each user
			m_reducedRatingsForEval=m_reducedRatings;
			m_unkwownInd=zeros(size(m_reducedRatings,1),2);
			for k=1:size(m_reducedRatings,1)
				v_ind=find(m_reducedRatings(k,:));
				m_unkwownInd(k,:)=datasample(v_ind,2,'Replace',false);
				m_reducedRatingsForEval(k,m_unkwownInd(k,:))=0;
			end
			s_numberOfTestValues=nnz(m_unkwownInd);
			for s_iterIndex=1:niter
				for s_itemIndex=1:s_totalItems
					%Should sample only from the know values...
					
					v_realSignalForEval=m_reducedRatingsForEval(:,s_itemIndex);
					v_realSignal=m_reducedRatings(:,s_itemIndex);
					%indicator matrix so that only the known values are
					%considered for NMSE and signal estimation
					v_knownValuesIndForEval=v_realSignalForEval~=0;
					v_knownValuesInd=v_realSignal~=0;
					
					s_knownValuesForEval=nnz(v_realSignalForEval);
					m_knownValuesInd=repmat(v_knownValuesInd, 1,s_monteCarloSimulations);
					%v_sampleSetSizeForItem=round(v_sampleSetSize*s_knownValues);
					
					
					
					%functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth);
					%functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
					%signal
					functionGenerator=RealDataGraphFunctionGenerator('graph',graph,'v_realSignal',v_realSignalForEval,'s_normalize',s_normalize);
					
					
					
					
					s_sampleSetIndex=1;
					
					s_numberOfSamples=s_knownValuesForEval;
					% % % %     if(s_numberOfSamples>s_monteCarloSimulations)
					% % % %         s_monteCarloSimulations=1;
					% % % %     end
					%sample
					sampler = PartiallyObservedGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR,'v_knownValuesInd',v_knownValuesIndForEval);
					m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
					[m_samples,m_positions] = sampler.sample(m_graphFunction);
					%estimate
					m_graphFunctionEstimateBL1 = bandlimitedGraphFunctionEstimator1.estimate(m_samples,m_positions);
					m_graphFunctionEstimateBL2= bandlimitedGraphFunctionEstimator2.estimate(m_samples,m_positions);
					m_graphFunctionEstimateNP=nonParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
					m_graphFunctionEstimateSP=semiParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
					%m_graphFunctionEpsilonInsesitiveEstimateSP=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samples,m_positions,s_lambda,s_epsilon);
					
					% Performance assessment
					%compute the mse error only among the known entries
					m_meanSquaredErrorInner(s_sampleSetIndex,1) =SemiParametricSimulations.estimateMeanSquaredError(m_graphFunctionEstimateBL1.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd);
					m_meanSquaredErrorInner(s_sampleSetIndex,2) =SemiParametricSimulations.estimateMeanSquaredError(m_graphFunctionEstimateBL2.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd);
					m_meanSquaredErrorInner(s_sampleSetIndex,3) =SemiParametricSimulations.estimateMeanSquaredError(m_graphFunctionEstimateNP.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd);
					m_meanSquaredErrorInner(s_sampleSetIndex,4) =SemiParametricSimulations.estimateMeanSquaredError(m_graphFunctionEstimateSP.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd);
					%m_meanSquaredErrorInner(s_sampleSetIndex,5) =SemiParametricSimulations.estimateRootMeanSquaredError(m_graphFunctionEpsilonInsesitiveEstimateSP.*m_knownValuesInd,m_graphFunction.*m_knownValuesInd,s_knownValues);
					
					m_rootMeanSquaredErrorOuter=m_rootMeanSquaredErrorOuter+m_meanSquaredErrorInner;
				end
				%here evaluate the squareed
				m_rootMeanSquaredErrorOuter=sqrt((1/s_numberOfTestValues)*m_rootMeanSquaredErrorOuter);
				m_rootMeanSquaredErrorOuterOuter=m_rootMeanSquaredErrorOuterOuter+m_rootMeanSquaredErrorOuter;
			end
			m_rootMeanSquaredErrorOuterOuter=(1/niter)*m_rootMeanSquaredErrorOuterOuter;
			F = F_figure('plot_type_2D','bar','Y',m_rootMeanSquaredErrorOuterOuter,'xlab','Percentage of observed vertices (S)','ylab','NMSE','tit',sprintf('#jokes=%g',s_totalItems),'leg',{strcat('Bandlimited  ',sprintf(' W=%g',s_bandwidth1)),strcat('Bandlimited',sprintf(' W=%g',s_bandwidth2)),'Nonparametric (SL)','Semiparametric (SL)','Semiparametric (\epsilon-IL)'});
			
			
		end
		function F = compute_fig_3004(obj,niter)
			F = obj.load_F_structure(3003);
			F.ylimit=[0 10];
			F.pos=[680 729 509 249];
			
			%F.leg_pos = 'northeast';      % it can be 'northwest',
			F.leg_pos_vec = [0.647 0.683 0.182 0.114];
		end
		%% Real data simulations 
		%  Data used: Temperature Dataset
		%  Goal: Convergence of objective function for the
		%  GraphLearningAlgorithm
		function F = compute_fig_1017(obj,niter)
			%convergence plot for estimate laplacian
			s_beta=0.002;
			s_alpha=0.02;
			s_niter=10;
			[Ho,Mo,Alto,Hn,Mn,Altn] = readTemperatureDataset;
			%tic
			graphGenerator = GraphLearningSmoothSignalGraphGenerator('m_observed',Ho,'s_niter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta,'m_missingValuesIndicator',[]);
			%toc
			
			% tic
			% graphGenerator=SmoothSignalGraphGenerator('m_observed',Ho,'s_maxIter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta);
			% toc
			
			[~,~,v_objective] = graphGenerator.realization;
			v_differenceObjective = -diff(v_objective);
			%s_omega=10^-6;
			%v_bound=s_omega*ones(size(v_differenceObjective));
			%Y=[v_differenceObjective;v_bound];
			F = F_figure('X',(2:size(v_objective,1)),'Y',v_differenceObjective,'xlab','number of iterations','ylab','objective function decrease','leg',{'GL-SigRep'});
		end
		function F = compute_fig_1018(obj,niter)
			F = obj.load_F_structure(1017);
			F.pos=[680 729 509 249];
            F.xlab='number of iterations';
            F.ylab='objective function decrease';
			F.logy=1;
			F.leg={'GL-SigRep'};
			%F.leg_pos = 'northeast';      % it can be 'northwest',
			F.leg_pos_vec = [0.647 0.683 0.182 0.114];
		end
				%% Real data simulations
		% Data used: Airport Dataset
		% Goal compare NMSE error
		function F=compute_fig_4001(obj,niter)
			% define parameters
			s_nodeNum = 80;
            s_departureDelay = 0;
			n_clusters=2;
			% estimation parameters
			s_numberOfSamples = 20;
            ref_lmmse = 0;
			%%%%%%%%%%%%%%%%%%

			% load data            
            [ m_training_data, m_test_data, m_training_adj , m_test_adj ] = AirportGraphGenerator.getSixMonths(s_nodeNum,s_departureDelay,'delaydata2014','delaydata2015');
            m_adjacency = sum(m_training_adj,3);
            m_adjacencyPure = (m_adjacency+m_adjacency') /2;
            m_adjacency = m_adjacencyPure > 100;
            
            sparsity = sum(m_adjacency(:))/(numel(m_adjacency)-size(m_adjacency,1))

			graphLearningInverseCovariance=GraphLearningInverseCovariance('m_adjacency',m_adjacency,'m_training_data',m_training_data);
			graph=graphLearningInverseCovariance.realization();
			m_constrainedLaplacian=graph.getLaplacian();
			% data normalization
			v_mean = mean(m_training_data,2);
			v_std = std(m_training_data')';
			m_normalized_training_data = diag(1./v_std)*(m_training_data - v_mean*ones(1,size(m_training_data,2)));
			m_normalized_test_data = diag(1./v_std)*(m_test_data - v_mean*ones(1,size(m_test_data,2)));

			%create parametric base
            m_adj=m_adjacencyPure;
            v_volumeOfFlights=sum(m_adj,1);
			
			m_clust=kmeans(v_volumeOfFlights',n_clusters);
			% cluster via the altitude information..
			m_basis=zeros(size(m_adj,1),n_clusters);
			for i=1:n_clusters
				m_basis(m_clust==i,i)=1;
			end
			% generator and sampler
            generator = RandomlyPickGraphFunctionGenerator('m_graphFunction',m_normalized_test_data);
            sampler = UniformGraphFunctionSampler('s_SNR',Inf,'s_numberOfSamples',s_numberOfSamples);
		
            % define Kernel function
			s_sigma=1.4;
			diffusionGraphKernel = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigma);
			
            % estimators
            s_mu = 1e-3;
            m_covInv = GraphLearningInverseCovariance.learnInverseCov( cov(m_normalized_training_data') , m_adj );
			cov_estimator = RidgeRegressionGraphFunctionEstimator('s_regularizationParameter',s_mu,'m_kernel',inv(m_covInv));
            %semiparametric
			s_lambdaSP= 1e-3;
			sem_estimator=SemiParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis,'s_lambda',s_lambdaSP);
			%parametric
            par_estimator=ParametricGraphFunctionEstimator('m_basis',m_basis);
            %nonparametric
			s_lambdaNP= 1e-3;
			nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'s_lambda',s_lambdaNP);
			%v_sigma=[1.1,1.3,1.4,1.5]; optimal 1.4
			%check for sigma
% 			for s_sigma_ind=1:size(v_sigma,2)
% 				diffusionGraphKernel = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',v_sigma(s_sigma_ind));
% 				nonParametricGraphFunctionEstimator(s_sigma_ind)=NonParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'s_lambda',s_lambda);
% 			end
% 		    v_lambda=[1e-1,1e-2,1e-3]; optimal 1e-3;
%           check for lambda
%             for s_lambda_ind=1:size(v_lambda,2)
% 				diffusionGraphKernel = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigma);
% 				nonParametricGraphFunctionEstimator(s_lambda_ind)=NonParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'s_lambda',v_lambda(s_lambda_ind));
% 			end
            % BL estimator
            bandwidth_vec = [1 3 8 -1];
            bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian', m_constrainedLaplacian);
            bl_estimator = bl_estimator.replicate('s_bandwidth', num2cell(bandwidth_vec), [], {});
            
            estimator = [cov_estimator;par_estimator;nonParametricGraphFunctionEstimator';sem_estimator; bl_estimator];
            % simulation
			
         	unnormalized_signal_mse = NaN(size(estimator,1),niter);			
            unnormalized_signal_energy = NaN(size(estimator,1),niter);			
			for s_itInd = 1:niter				
				% generate signal
				v_signal = generator.realization();
				
				% sample signal
				[v_samples,v_sampleLocations] = sampler.sample(v_signal);
                
                v_test_indices = 1:length(v_signal);
                v_test_indices(v_sampleLocations) = 0; v_test_indices = v_test_indices(v_test_indices>0);
                for s_estimatorInd = 1:size(estimator,1)
                    % estimate signal
                    v_signalEst = estimator(s_estimatorInd).estimate(v_samples,v_sampleLocations);
                    
                    % revert normalization
                    v_unnormalized_signal = (v_std).*v_signal + v_mean;
                    v_unnormalized_signalEst = (v_std).*v_signalEst + v_mean;
                    
                    % measure error                    
                    unnormalized_signal_mse(s_estimatorInd,s_itInd) = norm( v_unnormalized_signal(v_test_indices) - v_unnormalized_signalEst(v_test_indices) )^2/(length(v_test_indices)) ;%/  norm( v_unnormalized_signal(v_test_indices)  )^2;
                    unnormalized_signal_energy(s_estimatorInd,s_itInd) =  norm( v_unnormalized_signal(v_test_indices)  )^2/(length(v_test_indices));
                end
			end
			
			% average error
            unnormalized_signal_mse = mean(unnormalized_signal_mse,2)
            unnormalized_signal_nmse = unnormalized_signal_mse ./ mean(unnormalized_signal_energy,2)

            rmse_in_minutes = sqrt(unnormalized_signal_mse)
            
%             %%
%             % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 			% print the table into a tex file
% 			% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 			fid = fopen('libGF/simulations/MultikernelSimulations_data/airport.tex','w');
% 			fprintf(fid, '\\begin{tabular}{%s}\n', char('c'*ones(1,length(estimator))));     % heading line
% 			fprintf(fid, '\t\\hline\n\t');
% 			fprintf(fid, 'RR with cov & MKL1 & MKL2 & BL1 & BL2 & BL3 & BL4\\\\\n');
% 			fprintf(fid, '\t\\hline\n\t');
% 			
% 			% print NMSE
% 			fprintf(fid, 'NMSE\t');
% 			for i = 1:length(estimator)
% 				fprintf(fid, ' & %2.2f', unnormalized_signal_nmse(i));
% 			end
% 			fprintf(fid, '\\\\\n\tRMSE(min)\t');
% 			% print variance
% 			for i = 1:length(estimator)
% 				fprintf(fid, ' & %2.2f', rmse_in_minutes(i));
% 			end
% 			fprintf(fid, '\\\\\n');
% 			fprintf(fid, '\t\\hline\n');
% 			fprintf(fid, '\\end{tabular}');		% bottom line
% 			%caption = Parameter.getTitle(graphGenerator,functionGenerator,sampler,estimator);
% 			%fprintf(fid, caption);
%             
%             fclose(fid);
% 			% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 			
% 			
			F = [];
			
        end
        
        %%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Real data simulations 
		%  Data used: US temperature
		% Goal: methods comparison NMSE 
		function F = compute_fig_5001(obj,niter)
			
			%0. define parameters
			s_sigma=1.3;
			s_numberOfClusters=2;
			s_lambda=10^-5;
			s_monteCarloSimulations=niter;%100;
			s_bandwidth1=5;
			s_bandwidth2=10;
			s_SNR=1000;
			s_beta=0.02;
			s_alpha=0.005;
			s_niter=10;
			s_epsilon=0.2;
			s_functionTypes=6;
			v_sampleSetSize=(0.1:0.1:1);
			
			m_meanSquaredError=zeros(size(v_sampleSetSize,2),s_functionTypes);
			% define graph
			 load('temperatureAndAltUS.mat');
			%tic
			%graphGenerator = GraphLearningSmoothSignalGraphGenerator('m_observed',Ho,'s_niter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta,'m_missingValuesIndicator',[]);
			%toc
			
			% tic
			% graphGenerator=SmoothSignalGraphGenerator('m_observed',Ho,'s_maxIter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta);
			% toc
			
			graph = Graph('m_adjacency',m_adjacency);
			%graph1 = graphGenerator1.realization;
			%L1=graph.getLaplacian
			%L2=graph1.getLaplacian
			v_sampleSetSize=round(v_sampleSetSize*graph.getNumberOfVertices);
			
			
			m_basis= SemiParametricSimulations.parametricPartForTempData(graph.getLaplacian,v_altitudes,s_numberOfClusters);
			
			%functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth);
			%functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
			%signal
			v_realSignal=v_meanTemperature;
			functionGenerator=RealDataGraphFunctionGenerator('graph',graph,'v_realSignal',v_realSignal,'s_normalize',0);
			% define bandlimited function estimator
			%m_laplacianEigenvectors=(graph.getLaplacianEigenvectors);
			bandlimitedGraphFunctionEstimator1 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth1);
			bandlimitedGraphFunctionEstimator2 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth2);
			
			% define Kernel function
			diffusionGraphKernel = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigma);
			%define non-parametric estimator
			nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix);
			
			%define semi-parametric estimator
			semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
			
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
            parametricGraphFunctionEstimator=ParametricGraphFunctionEstimator('m_basis',m_basis);

			
			% Simulation
			for s_sampleSetIndex=1:size(v_sampleSetSize,2)
				
				
				s_numberOfSamples=v_sampleSetSize(s_sampleSetIndex);
				%sample
				sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
				m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
				[m_samples,m_positions] = sampler.sample(m_graphFunction);
				%estimate
				m_graphFunctionEstimateBL1 = bandlimitedGraphFunctionEstimator1.estimate(m_samples,m_positions);
				m_graphFunctionEstimateBL2= bandlimitedGraphFunctionEstimator2.estimate(m_samples,m_positions);
				m_graphFunctionEstimateNP=nonParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
				m_graphFunctionEstimateSP=semiParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
				m_graphFunctionEpsilonInsesitiveEstimateSP=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samples,m_positions,s_lambda,s_epsilon);
				m_graphFunctionEstimateP=parametricGraphFunctionEstimator.estimate(m_samples,m_positions);
				% Performance assessment
				m_meanSquaredError(s_sampleSetIndex,1) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL1,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,2) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL2,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,3) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateP,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,4) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateNP,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,5) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSP,m_graphFunction);
                m_meanSquaredError(s_sampleSetIndex,6) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEpsilonInsesitiveEstimateSP,m_graphFunction);
				
			end
			%m_meanSquaredError(m_meanSquaredError>1)=1;
			%save('real.mat');
			
			F = F_figure('X',v_sampleSetSize,'Y',m_meanSquaredError','xlab','Number of observed stations (S)','ylab','NMSE','leg',{strcat('Bandlimited  ',sprintf(' W=%g',s_bandwidth1)),strcat('Bandlimited',sprintf(' W=%g',s_bandwidth2)),'Parametric','Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)'});
            	F.caption=[	sprintf('regularization parameter mu=%g\n',s_lambda),...
							sprintf(' diffusion parameter sigma=%g\n',s_sigma)];
        
        end
		
		function F = compute_fig_5002(obj,niter)
			F = obj.load_F_structure(5001);
			F.ylimit=[0 1];
			F.xlimit=[10 109];
            F.logy=1;
			F.styles = {'-','--','-o','-x','--^','-.'};
			F.pos=[680 729 509 249];
			F%.leg={strcat('Bandlimited  ',sprintf(' W=10')),strcat('Bandlimited',sprintf(' W=20')),'Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)'};
			
			F.leg_pos = 'north';      % it can be 'northwest',
			%F.leg_pos_vec = [0.547 0.673 0.182 0.114];
        end
		
         %%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Real data simulations 
		%  Data used: US temperature
		% Goal: methods comparison NMSE compare with culp2011
		function F = compute_fig_5003(obj,niter)
			
			%0. define parameters
			s_sigmaDiffusionKernel=10^-2;%1.3;
            s_sigmaLaplacianInvKernel=10^-11;
			s_numberOfClusters=4;
			s_lambda=5*10^-5;
			s_monteCarloSimulations=niter;%100;
			s_bandwidth1=5;
			s_bandwidth2=10;
			s_SNR=1000;
			s_beta=0.02;
			s_alpha=0.005;
			s_niter=10;
			s_epsilon=0.2;
			s_functionTypes=7;
			v_sampleSetSize=(0.1:0.1:1);
			
			m_meanSquaredError=zeros(size(v_sampleSetSize,2),s_functionTypes);
			% define graph
			 load('temperatureAndAltUS.mat');
			%tic
			%graphGenerator = GraphLearningSmoothSignalGraphGenerator('m_observed',Ho,'s_niter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta,'m_missingValuesIndicator',[]);
			%toc
			
			% tic
			% graphGenerator=SmoothSignalGraphGenerator('m_observed',Ho,'s_maxIter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta);
			% toc
			
			graph = Graph('m_adjacency',m_adjacency);
			%graph1 = graphGenerator1.realization;
			%L1=graph.getLaplacian
			%L2=graph1.getLaplacian
			v_sampleSetSize=round(v_sampleSetSize*graph.getNumberOfVertices);
			
			
			m_basis= SemiParametricSimulations.parametricPartForTempData(graph.getLaplacian,v_altitudes,s_numberOfClusters);
			
			%functionGeneratorBL = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',s_bandwidth);
			%functionGenerator= SemiParametricGraphFunctionGenerator('graph',graph,'graphFunctionGenerator',functionGeneratorBL,'m_parametricBasis',m_basis);
			%signal
			v_realSignal=v_meanTemperature;
			functionGenerator=RealDataGraphFunctionGenerator('graph',graph,'v_realSignal',v_realSignal,'s_normalize',0);
			% define bandlimited function estimator
			%m_laplacianEigenvectors=(graph.getLaplacianEigenvectors);
			bandlimitedGraphFunctionEstimator1 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth1);
			bandlimitedGraphFunctionEstimator2 = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',s_bandwidth2);
			
			% define Kernel function
			diffusionGraphKernel = DiffusionGraphKernel('m_laplacian',graph.getLaplacian,'s_sigma',s_sigmaDiffusionKernel);
			%define non-parametric estimator
			nonParametricGraphFunctionEstimator=NonParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix);
			
			%define semi-parametric estimator
			semiParametricGraphFunctionEstimator = SemiParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
            semiParametricGraphFunctionEstimatorEarlyStop = SemiParametricGraphFunctionEstimator('m_kernels',diffusionGraphKernel.earlyStopEigVecGenerateKernelMatrix(10^-3),'m_basis',m_basis);

			
            %define culp2011 estimator
            rLaplacianReg = @(lambda,epsilon) lambda + epsilon;
            h_rFun_inv = @(lambda) 1./rLaplacianReg(lambda,s_sigmaLaplacianInvKernel);
            laplacianKernel = LaplacianKernel('m_laplacian',graph.getLaplacian,'h_r_inv',{h_rFun_inv});
            semiParametricGraphFunctionEstimatorCulp = SemiParametricGraphFunctionEstimator('m_kernels',laplacianKernel.getKernelMatrix,'m_basis',m_basis);
            
            %define semi-parametric epsilon incenitve est
			semiParametricGraphFunctionEpsilonInsesitiveEstimator=SemiParametricGraphFunctionEpsilonInsesitiveEstimator('m_kernels',diffusionGraphKernel.generateKernelMatrix,'m_basis',m_basis);
            parametricGraphFunctionEstimator=ParametricGraphFunctionEstimator('m_basis',m_basis);

			
			% Simulation
			for s_sampleSetIndex=1:size(v_sampleSetSize,2)
				
				
				s_numberOfSamples=v_sampleSetSize(s_sampleSetIndex);
				%sample
				sampler = UniformGraphFunctionSampler('s_numberOfSamples',s_numberOfSamples,'s_SNR',s_SNR);
				m_graphFunction = functionGenerator.realization(s_monteCarloSimulations);
				[m_samples,m_positions] = sampler.sample(m_graphFunction);
				%estimate
				m_graphFunctionEstimateBL1 = bandlimitedGraphFunctionEstimator1.estimate(m_samples,m_positions);
				m_graphFunctionEstimateBL2= bandlimitedGraphFunctionEstimator2.estimate(m_samples,m_positions);
				m_graphFunctionEstimateNP=nonParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
				m_graphFunctionEstimateSP=semiParametricGraphFunctionEstimator.estimate(m_samples,m_positions,s_lambda);
				%m_graphFunctionEpsilonInsesitiveEstimateSP=semiParametricGraphFunctionEpsilonInsesitiveEstimator.estimate(m_samples,m_positions,s_lambda,s_epsilon);
				m_graphFunctionEstimateP=parametricGraphFunctionEstimator.estimate(m_samples,m_positions);
                m_graphFunctionEstimateSPcul=semiParametricGraphFunctionEstimatorCulp.estimate(m_samples,m_positions,s_lambda);
                m_graphFunctionEstimateSPEarlyStop=semiParametricGraphFunctionEstimatorEarlyStop.estimate(m_samples,m_positions,s_lambda);
				% Performance assessment
				m_meanSquaredError(s_sampleSetIndex,1) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL1,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,2) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateBL2,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,3) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateP,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,4) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateNP,m_graphFunction);
                m_meanSquaredError(s_sampleSetIndex,5) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSPcul,m_graphFunction);
				m_meanSquaredError(s_sampleSetIndex,6) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSP,m_graphFunction);
                m_meanSquaredError(s_sampleSetIndex,7) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEstimateSPEarlyStop,m_graphFunction);

               %m_meanSquaredError(s_sampleSetIndex,7) =SemiParametricSimulations.estimateNormalizedMeanSquaredError(m_graphFunctionEpsilonInsesitiveEstimateSP,m_graphFunction);
				
			end
			%m_meanSquaredError(m_meanSquaredError>1)=1;
			%save('real.mat');
			
			F = F_figure('X',v_sampleSetSize,'Y',m_meanSquaredError','xlab','Number of observed stations (S)','ylab','NMSE','leg',{strcat('LS',sprintf(' B=%g',s_bandwidth1)),strcat('LS',sprintf(' B=%g',s_bandwidth2)),'P','NP','SP','SP-GK','SP-GK-Early'});
            F.ylimit=[0 1];
            F.logy=1;
            F.caption=[	sprintf('regularization parameter mu=%g\n',s_lambda),...
							sprintf(' diffusion parameter sigma=%g\n',s_sigmaDiffusionKernel),sprintf(' reguralized laplacian parameter sigma=%g\n',s_sigmaLaplacianInvKernel)... s_numberOfClusters
                            ,sprintf(' # clusters=%g\n',s_numberOfClusters),sprintf(' # epsilon=%g\n',s_epsilon)];
        end
		
		function F = compute_fig_5004(obj,niter)
			F = obj.load_F_structure(5003);
			F.ylimit=[0 1];
			F.xlimit=[10 98];
            F.logy=1;
			F.styles = {'-o','-x','-^','--o','--x','--^','--*'};
            F.colorset=[0 0 0;0 .7 0;.5 .5 0; .9 0 .9 ;.5 0 1;1 0 0;   1 0 .5; 0 1 .5;0 1 0];

			F.pos=[680 729 509 249];
			%.leg={strcat('Bandlimited  ',sprintf(' W=10')),strcat('Bandlimited',sprintf(' W=20')),'Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)'};
			
			F.leg_pos = 'southwest';      % it can be 'northwest',
			%F.leg_pos_vec = [0.547 0.673 0.182 0.114];
        end
        function F = compute_fig_50041(obj,niter)
			F = obj.load_F_structure(5003);
			F.ylimit=[0 1];
			F.xlimit=[10 98];
            F.logy=1;
              
            F.Y(5,:)=[];
            
            F.leg{1,5}= F.leg{1,6};
            F.leg{1}='BL B=5';
            F.leg{2}='BL B=10';
            F.leg=F.leg(1:5);
			F.styles = {'-o','-x','-^','--o','--^','--*'};
            F.colorset=[0 0 0;0 .7 0;.5 .5 0; .9 0 .9 ; 1 0 0;   1 0 .5; 0 1 .5;0 1 0];

			F.pos=[680 729 509 249];
			%.leg={strcat('Bandlimited  ',sprintf(' W=10')),strcat('Bandlimited',sprintf(' W=20')),'Nonparametric (SL)','Semi-parametric (SL)','Semi-parametric (\epsilon-IL)'};
			
			F.leg_pos = 'southwest';      % it can be 'northwest',
			%F.leg_pos_vec = [0.547 0.673 0.182 0.114];
		end
	

	end
	
	
	
	
	methods(Static)
		%used for creating graph on recommendation systems maybe move....
		function m_adjacency=generateAdjancency(m_train)
			%% Create a graph of users based on their cosine similarity
			m_adjacency=zeros(size(m_train,1));
			for k=1:size(m_train,1)
				for l=1:k-1
					m_adjacency(k,l)=m_train(k,:)*(m_train(l,:))'/(norm(m_train(k,:))*norm(m_train(l,:)));
				end
			end
			m_adjacency(isnan(m_adjacency)) = 0 ;
			m_adjacency=m_adjacency+m_adjacency';
		end
		function m_adjacency=generateAdjancencyJensen(m_train,sigma)
			%% Create a graph of users based on their cosine similarity
			%Take into account negative values.. how to address..
			%as a first step take abs no abs just take the exp of the inner
			%product
			m_adjacency=zeros(size(m_train,1));
			for k=1:size(m_train,1)
				for l=1:k-1
					m_adjacency(k,l)=exp(sigma^2*m_train(k,:)*(m_train(l,:))'/(norm(m_train(k,:))*norm(m_train(l,:))));
				end
			end
			m_adjacency(isnan(m_adjacency)) = 0 ;
			m_adjacency=m_adjacency+m_adjacency';
		end
		function B=parametricPartForTempData(L,feat,n_clusters)
			
			
			C=kmeans(feat,n_clusters);
			% cluster via the altitude information..
			B=zeros(size(L,1),1);
			for i=1:n_clusters
				B(C==i,i)=1;
			end
			
		end
		%creates indicator matrix of unsampeld vertices
		function m_indicator=createIndicatorMatrix(m_graphFunction,m_samples)
			m_indicator=zeros(size(m_graphFunction));
			for i=1:size(m_graphFunction,2)
			m_indicator(m_samples(:,i),i)=1;
			end
			m_indicator=~m_indicator;
		end
		%
		function m_samplesWithOutliers=combineFunctionWithOutliers(m_graphFunction,m_outlyingFunction,m_samples,m_positions,s_numberOfOutliers)

			m_samplesWithOutliers=m_samples;
			for i=1:size(m_graphFunction,2)
		    v_outlyingPositions=randsample(m_positions(:,i),s_numberOfOutliers);
			m_samplesWithOutliers(ismember(m_positions(:,i),v_outlyingPositions),i)=m_graphFunction(v_outlyingPositions,i)+m_outlyingFunction( v_outlyingPositions,i);
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
		%estimates the root mean squared error over the known values
		function res = estimateMeanSquaredError(m_est,m_observed)
			res=0;
			for i=1:size(m_est,2)
				res=res+norm(m_est(:,i)-m_observed(:,i))^2;
			end
			res=(1/size(m_est,2))*res;
		end
		
		
		
		
	end
	
	
end
