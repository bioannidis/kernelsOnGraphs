%
%  FIGURES FOR THE PAPER ON MULTIKERNEL
%
%  TSP paper figures: 1006, 3100 (print 3108), 3402 (print 3404), 3232,
%  7001(print 7002)
%  (table), 3234, for the revision 3502 (print 3503)
%   
%3305
%

classdef MultikernelSimulations < simFunctionSet
	
	properties
	
	end
	
	methods
		
		% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% %%  1. Generic simulations
		% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		% This is a very simple simulation to test bandlimited LS
		% estimation
		function F = compute_fig_1001(obj,niter)
			
			N = 100; % number of vertices
			B = 30;  % bandwidth
			
			% 1. define graph function generator
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.3,'s_numberOfVertices',N);
			graph = graphGenerator.realization;
			functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			
			% 2. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_numberOfSamples',40,'s_SNR',20);
			
			% 3. define graph function estimator
			estimator = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',B);
			
			% Simulation
			m_graphFunction = functionGenerator.realization();
			[m_samples,m_positions] = sampler.sample(m_graphFunction);
			m_graphFunctionEstimate = estimator.estimate(m_samples,m_positions);
			
			% Performance assessment
			error = norm(m_graphFunctionEstimate - m_graphFunction,'fro')^2/size(m_graphFunction,1)
			
			F = F_figure('X',1:N,'Y',[m_graphFunctionEstimate,m_graphFunction]','leg',{'estimate','true'},'xlab','VERTEX','ylab','FUNCTION');
			
		end
					
		% This is a very simple simulation to test the computation of the
		% cut-off frequency in [narang2013structured] and [anis2016proxies]
		function F = compute_fig_1002(obj,niter)
			
			N = 100; % number of vertices
			B = 30;  % bandwidth
			SNR = 10; % dB
			
			% 1. define graph function generator
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.3,'s_numberOfVertices',N);
			graph = graphGenerator.realization;
			functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			
			% 2. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_numberOfSamples',40,'s_SNR',SNR);
			
			% 3. define graph function estimator
			estimator_known_freq = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',B);
			estimator_unknown_freq = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',-1);
			
			% Simulation
			m_graphFunction = functionGenerator.realization();
			[m_samples,m_positions] = sampler.sample(m_graphFunction);
			m_graphFunctionEstimate_known_freq = estimator_known_freq.estimate(m_samples,m_positions);
			m_graphFunctionEstimate_unknown_freq = estimator_unknown_freq.estimate(m_samples,m_positions);
			
			% Performance assessment
			error_known_freq = norm(m_graphFunctionEstimate_known_freq - m_graphFunction,'fro')^2/size(m_graphFunction,1)
			error_unknown_freq = norm(m_graphFunctionEstimate_unknown_freq - m_graphFunction,'fro')^2/size(m_graphFunction,1)
			
			F = F_figure('X',1:N,'Y',[m_graphFunction,m_graphFunctionEstimate_known_freq,m_graphFunctionEstimate_unknown_freq]','leg',{'true','estimate (known freq.)','estimate (unknown freq.)'},'xlab','VERTEX','ylab','FUNCTION','styles',{'-','--','-.'});
			
		end
				
		% This is a simple simulation to construct a Monte Carlo figure
		function F = compute_fig_1003(obj,niter)
			
			N = 100; % number of vertices
			S_vec = 10:10:100; % number of samples
			B = 20; % bandwidth of the estimated function
			B_vec = 10:10:50; % assumed bandwidth for estimation
			
			% 1. define graph function generator
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.3,'s_numberOfVertices',N);
			graph = graphGenerator.realization;
			bandlimitedFunctionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			graphFunction = bandlimitedFunctionGenerator.realization();
			generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',graphFunction);
			
			% 2. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',20);
			sampler = sampler.replicate([],{},'s_numberOfSamples',num2cell(S_vec));
						
			% 3. define graph function estimator
			estimator = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian);
			estimator = estimator.replicate('s_bandwidth',num2cell(B_vec),'',{});

			% Simulation
			res = Simulator.simStatistic(niter,generator,sampler,estimator);
			mse = Simulator.computeNmse(res,Results('stat',graphFunction));			
			
			% Representation of results
			F = F_figure('X',Parameter.getXAxis(generator,sampler,estimator),'Y',mse,'leg',Parameter.getLegend(generator,sampler,estimator),'xlab',Parameter.getXLabel(generator,sampler,estimator),'ylab','MSE','ylimit',[0 1.5]);
			
		end
		
		% This is a simple simulation to construct a Monte Carlo figure
		% Different from 2001, objets of different classes are concatenated
		function F = compute_fig_1004(obj,niter)
						
			N = 100; % number of vertices			
			B = 20; % bandwidth of the estimated function
			B_vec =         [10 20 30 10 20 30]; % assumed bandwidth for estimation
			SNR_vec = [15 25 15 15 15 25 25 25]; % SNR for each curve (first 2 for multikernel)
			
			S_vec = 10:10:100;
			
			% 1. define graph function generator
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.9,'s_numberOfVertices',N);
			graph = graphGenerator.realization;
			bandlimitedFunctionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			graphFunction = bandlimitedFunctionGenerator.realization();
			generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',graphFunction);			
			
			% 2. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',20);			
			sampler = sampler.replicate('s_SNR',num2cell(SNR_vec),'s_numberOfSamples',num2cell(S_vec));		
						
			% 3. BL graph function estimator
			bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian);			
			bl_estimator.c_replicatedVerticallyAlong = {'ch_name'};
			bl_estimator = bl_estimator.replicate('s_bandwidth',num2cell(B_vec),'',{});
					
			% 4. MKL function estimator
		    m_laplacian = bandlimitedFunctionGenerator.basis(N);
			m_kernel = cat(3,pinv(m_laplacian)+1e-10*eye(N),pinv(m_laplacian^2)+1e-10*eye(N));
			mkl_estimator = MkrGraphFunctionEstimator('m_kernel',m_kernel,'s_regularizationParameter',1e-5);
			mkl_estimator.c_replicatedVerticallyAlong = {'ch_name'};

			est = [mkl_estimator;mkl_estimator;bl_estimator];
			
			% Simulation
			res = Simulator.simStatistic(niter,generator,sampler,est);
			mse = Simulator.computeNmse(res,Results('stat',graphFunction));
			
			% Representation			
			F = F_figure('X',Parameter.getXAxis(generator,sampler,est),...
                'Y',mse,'leg',Parameter.getLegend(generator,sampler,est),...
                'xlab',Parameter.getXLabel(generator,sampler,est));
			
		end
		
		% Figure to check analytic expression for interpolating functions
		% (columns of the kernel matrix) in a circular graph
		function F = compute_fig_1005(obj,niter)
			
			vertexNum = 100;
			columnInd = 25;
			A = circshift(eye(vertexNum),1)+circshift(eye(vertexNum),-1);
			L = diag(sum(A,2))-A;
						
			% Computation through analytic expression for
			% a) Laplacian regularization
			epsilon = .01;
			rLaplacianReg = @(lambda,epsilon) lambda + epsilon;
			KcolLaplacianReg_analytic = MultikernelSimulations.columnLaplacianKernelCircularGraph(vertexNum,@(lambda) rLaplacianReg(lambda,epsilon) , columnInd);
			
			% b) Diffusion kernel
			sigma2 = 3;
			rDiffusionKernel = @(lambda,sigma2) exp(sigma2*lambda/2);
			KcolDiffusionKernel_analytic = MultikernelSimulations.columnLaplacianKernelCircularGraph(vertexNum,@(lambda) rDiffusionKernel(lambda,sigma2) , columnInd);
						
			% direct computation for
			
			% a) regularized laplacian
			h_rFun_inv = @(lambda) 1./rLaplacianReg(lambda,epsilon);
			kG = LaplacianKernel('m_laplacian',L,'h_r_inv',{h_rFun_inv});
			m_KernelMatrix = kG.getKernelMatrix;
			KcolLaplacianReg_direct = m_KernelMatrix(:,columnInd);
			
			% a) diffusion kernel
			h_rFun_inv = @(lambda) 1./rDiffusionKernel(lambda,sigma2);
			kG = LaplacianKernel('m_laplacian',L,'h_r_inv',{h_rFun_inv});
			m_KernelMatrix = kG.getKernelMatrix;
			KcolDiffusionKernel_direct = m_KernelMatrix(:,columnInd);
			
			
			F(1) = F_figure('X',1:vertexNum,'Y',[KcolLaplacianReg_direct';KcolLaplacianReg_analytic'],'styles',{'-','--'});
			F(2) = F_figure('X',1:vertexNum,'Y',[KcolDiffusionKernel_direct';KcolDiffusionKernel_analytic'],'styles',{'-','--'});
			
		end
		
		% Figure to illustrate the interpolating functions (columns of the
		% kernel matrix) in a circular graph
		function F = compute_fig_1006(obj,niter)
			
			vertexNum = 100;
			columnInd = 25;
			A = circshift(eye(vertexNum),1)+circshift(eye(vertexNum),-1);
			L = diag(sum(A,2))-A;
						
			% Computation through analytic expression for
			% a) Laplacian regularization
			rLaplacianReg = @(lambda,s2) 1+s2*lambda;
			v_sigma2_LaplacianReg = [1 20 100];
			for i_sigma2 = length(v_sigma2_LaplacianReg):-1:1				
				KcolLaplacianReg(i_sigma2,:) = MultikernelSimulations.columnLaplacianKernelCircularGraph(vertexNum,@(lambda) rLaplacianReg(lambda,v_sigma2_LaplacianReg(i_sigma2)) , columnInd)';
				leg{i_sigma2} = sprintf('Laplacian reg. (\\sigma^2 = %g )',v_sigma2_LaplacianReg(i_sigma2));
			end
			KcolLaplacianReg = diag(1./max(KcolLaplacianReg,[],2))*KcolLaplacianReg;
			
			
			% b) Diffusion kernel			
			v_sigma2_DiffusionKernel = [1 20 100];
			rDiffusionKernel = @(lambda,sigma2) exp(sigma2*lambda/2);
			i_legLen = length(leg);
			for i_sigma2 = length(v_sigma2_DiffusionKernel):-1:1
				KcolDiffusionKernel(i_sigma2,:) = MultikernelSimulations.columnLaplacianKernelCircularGraph(vertexNum,@(lambda) rDiffusionKernel(lambda,v_sigma2_DiffusionKernel(i_sigma2)) , columnInd)';
				leg{i_sigma2+i_legLen} = sprintf('Diffusion k. (\\sigma^2 = %g )',v_sigma2_DiffusionKernel(i_sigma2));
			end		
			KcolDiffusionKernel = diag(1./max(KcolDiffusionKernel,[],2))*KcolDiffusionKernel;
			
			caption = sprintf('%d-th column of the kernel matrix for a circular graph with N = %d vertices.',columnInd,vertexNum);
			m_Y = [KcolLaplacianReg;KcolDiffusionKernel];
			F = F_figure('X',1:2:vertexNum,'Y',m_Y(:,1:2:vertexNum),'leg',leg,'styles',{'-','-x','-o','--','--x','--o'},'colorp',3,'xlab','Vertex index (n)','ylab','Function value','caption',caption,'leg_pos_vec',[0.5546    0.5271    0.2333    0.3715]);
			
		end
		
		
		% Figure to illustrate the interpolating functions (columns of the
		% kernel matrix) in a circular graph
		function F = compute_fig_1007(obj,niter)
			
			vertexNum = 100;
			v_columnInd = [ 10 25  50 60 70];
			v_amplitudes = [.5 .2 1 .7 .8];
			A = circshift(eye(vertexNum),1)+circshift(eye(vertexNum),-1);
			L = diag(sum(A,2))-A;
						
		
			% b) Diffusion kernel			
			sigma2_DiffusionKernel = [20];
			rDiffusionKernel = @(lambda,sigma2) exp(sigma2*lambda/2);			
			for i_center = length(v_columnInd):-1:1
				KcolDiffusionKernel(i_center,:) = MultikernelSimulations.columnLaplacianKernelCircularGraph(vertexNum,@(lambda) rDiffusionKernel(lambda,sigma2_DiffusionKernel) , v_columnInd(i_center))';
				leg{i_center} = sprintf('\\alpha_{%d} k(v_{n''},v_{%d})',v_columnInd(i_center),v_columnInd(i_center));
			end		
			KcolDiffusionKernel = diag(v_amplitudes'./max(KcolDiffusionKernel,[],2))*KcolDiffusionKernel;
						
			m_Y = [KcolDiffusionKernel;sum(KcolDiffusionKernel,1)];
			F = F_figure('X',1:2:vertexNum,'Y',m_Y(:,1:2:vertexNum),'styles',{'-','-','-','-','-','-'},'colorset',[zeros(length(v_columnInd),3);1 0 0],'xlab','Vertex index','ylab','Function value','xlimit',[1,vertexNum],'gstyle','');
			
		end
		
		
		% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% %%  2. simulations with estimators for bandlimited signals on
		% %%  synthetic data 
		% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			
		% Simple simulation to test [narang2013structured] and
		% [anis2016proxies] cut-off freq. estimation method
		function F = compute_fig_2001(obj,niter)
						
			N = 100; % number of vertices			
			B = 20; % bandwidth of the estimated function
			B_vec =         [20]; % assumed bandwidth for estimation
			SNR_vec = [15 25 25 25]; % SNR for each curve (first 2 for multikernel)
			
			S_vec = 10:10:100;
			
			% 1. define graph function generator
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.9,'s_numberOfVertices',N);
			graph = graphGenerator.realization;
			m_laplacian = graph.getLaplacian(); 
			bandlimitedFunctionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			graphFunction = bandlimitedFunctionGenerator.realization();
			generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',graphFunction);			
			
			% 2. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',20);			
			sampler = sampler.replicate('s_SNR',num2cell(SNR_vec),'s_numberOfSamples',num2cell(S_vec));		
						
			% 3. BL graph function estimator
			bl_estimator_known_freq = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian);			
			bl_estimator_known_freq.c_replicatedVerticallyAlong = {'ch_name'};
			bl_estimator_known_freq = bl_estimator_known_freq.replicate('s_bandwidth',num2cell(B_vec),'',{});
					
			% 4. BL estimator with unknown frequency
			bl_estimator_unknown_freq = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',-1);			
			bl_estimator_unknown_freq.c_replicatedVerticallyAlong = {'ch_name','s_bandwidth'};
						
			% 5. MKL function estimator		    
			sigma2Array = linspace(0.1, .5 , 20);            
            kG = LaplacianKernel('m_laplacian',m_laplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigma2Array));
			m_kernel = kG.getKernelMatrix();
			mkl_estimator = MkrGraphFunctionEstimator('m_kernel',m_kernel,'s_regularizationParameter',1e-3);
			mkl_estimator.c_replicatedVerticallyAlong = {'ch_name'};

			est = [mkl_estimator;mkl_estimator;bl_estimator_known_freq;bl_estimator_unknown_freq];
			
			% Simulation
			res = Simulator.simStatistic(niter,generator,sampler,est);
			mse = Simulator.computeNmse(res,Results('stat',graphFunction));
			
			% Representation			
			F = F_figure('X',Parameter.getXAxis(generator,sampler,est),...
                'Y',mse,'leg',Parameter.getLegend(generator,sampler,est),...
                'xlab',Parameter.getXLabel(generator,sampler,est),'ylimit',...
				[0 1.5],'ylab','NMSE','tit',Parameter.getTitle(graphGenerator,bandlimitedFunctionGenerator,generator,sampler));
			
		end
				
		
		
		% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% %%  3. simulations with MKL on synthetic data
		% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
        % Index
        %       sigma w.r.t bandwidth              Figure 3100
        %       sigma w.r.t sample size            Figure 3101
        %       performance comparison of
        %       single kernel and multikernel      Figure 3102
        %       IIA with bandlimited kernels       Figure 3103
        %       RKHS with bandlimited kernels      Figure 3104
        %       Sparsity of alpha w.r.t mu         Figure 3201
        %       NMSE vs mu(regularization)         Figure 3202
        %       Test parameter for Cortes          Figure 3203
        
        
        % 1) Figures for tuning kernel parameters==========================
        
		% Figure: NMSE vs sigma
        %    Instead of drawing one curve per sample size, draw one curve
        %    per bandwidth (of signal).
        function F = compute_fig_3100(obj,niter)		
			[N,p,SNR,sampleSize,~] = MultikernelSimulations.simulationSetting();
			N = 100;
			SNR = 20;
			sampleSize = 40;
			p = .25; 
            mu = 1e-4;
            bandwidthVec = [5 10 20 30 40];
						
			% generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
            
            % generate signal on this graph
			functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'ch_distribution','uniform');
            %functionGenerator.b_generateSameFunction = 1;
            generator = functionGenerator.replicate('s_bandwidth', ...
                num2cell(bandwidthVec), [], {} );
			
			% generate Kernel matrix
			sigmaArray = sqrt(linspace(0.01, 1.5, 30));
            %sigma = 0.8;
			L = graph.getLaplacian();
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray));
			m_kernel = kG.getKernelMatrix();
            
			% define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR, ...
                's_numberOfSamples', sampleSize);
			
			% define function estimator
            estimator = MkrGraphFunctionEstimator('s_regularizationParameter',mu);
            estimator = estimator.replicate([],{}, ...
                'm_kernel', mat2cell(m_kernel, N, N, ones(1,size(m_kernel,3))));
			
			caption = Parameter.getTitle(graphGenerator,generator,sampler,estimator);
			
			% Simulation
            mse = Simulate(generator, sampler, estimator, niter);
            
            % Representation
			
            F = F_figure('X',sigmaArray.^2,'Y',mse, ...
                'leg',Parameter.getLegend(generator,sampler, estimator),...
                'xlab','\sigma^2','ylab','NMSE',...
                'caption',caption,'styles',{'-','--','-^','--^','-*'},'leg_pos_vec',[ 0.3073    0.6041    0.2206    0.3045]);		  

		end
		
		% Figure NMSE vs samplesize
		%    each curve corresponds to one bandwidth -- bandlimited kernel
		function F = compute_fig_3110(obj,niter)
			[N,p,SNR,sampleSize,~] = MultikernelSimulations.simulationSetting();
			mu = 1e-4;
			B = 20;  % used to generate graph signals
            bandwidthVec = 10:10:60;  % used to build bandlimited kernels
			S_vec = 10:5:90;
			beta = 1000;
						
			% generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
            
            % generate signal on this graph
			generator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth', B);
			
			% generate Kernel matrix
			L = graph.getLaplacian();
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.bandlimitedKernelFunctionHandle( L , bandwidthVec , beta));
			m_kernel = kG.getKernelMatrix();
            
			% define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);
			sampler = sampler.replicate([],{},'s_numberOfSamples', num2cell(S_vec));
			
			% define function estimator
            estimator = MkrGraphFunctionEstimator('s_regularizationParameter',mu);
            estimator = estimator.replicate('m_kernel', mat2cell(m_kernel, N, N, ones(1,size(m_kernel,3))),[],{});
			
			% Simulation
            mse = Simulate(generator, sampler, estimator, niter);
            
            % Representation
			for i = 1:length(bandwidthVec)
				legendStr{i}  = sprintf('B = %d', bandwidthVec(i));
			end
            F = F_figure('X',S_vec,'Y',mse, ...
                'leg',legendStr, 'ylimit', [0 1.5], ...
                'xlab','sample size','ylab','Normalized MSE',...
                'tit', sprintf('N=%d, p=%2.2f, \\mu=%3.1d, signal bandwidth = %d', N, p, mu, B));		  
        end
        
        % Figure NMSE vs bandwidth
        %    comparing single kernel and multikernel
        %    
        function F = compute_fig_3111(obj,niter)
            %% set parameter
			[N,p,SNR,~,~] = MultikernelSimulations.simulationSetting();
            N = 250;
            sampleSize = 100;
            %p = .7;
            mu = 1e-4;
            bandwidthVec = 10:10:N;
						
			%% generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
            
            %% generate signal on this graph
			functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph);
            functionGenerator.b_generateSameFunction = 0;
            generator = functionGenerator.replicate([], {}, 's_bandwidth', num2cell(bandwidthVec));
			
			%% generate Kernel matrix
            L = graph.getLaplacian();
            %sigmaCell = { sqrt(0.01), sqrt(0.1), sqrt(0.5), sqrt(1), sqrt(linspace(0.01, 1, 10)), sqrt(linspace(0.01, 1, 30)) };
            %sigmaCell = { sqrt(0.2), sqrt(0.4), sqrt(0.6), sqrt(.8), sqrt(1), (linspace(sqrt(0.2), sqrt(1), 10)), sqrt(linspace(0.2, 1, 10)) };
            sigmaCell = { sqrt(0.1), sqrt(.15), sqrt(.2), sqrt(.25), sqrt(.3), (linspace(sqrt(0.1), sqrt(.3), 10)), sqrt(linspace(0.01, .2, 10)) };
			%sigmaArray = sqrt(linspace(0.01, 1.5, 30));
            %sigma = 0.8;
            for i = 1:length(sigmaCell)
                kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaCell{i}));
                m_kernel{i} = kG.getKernelMatrix();
            end
            
			%% define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR, 's_numberOfSamples', sampleSize);
			
			%% define function estimator
            est = MkrGraphFunctionEstimator('s_regularizationParameter',mu, 'ch_type', 'kernel superposition');
            est = est.replicate('m_kernel', m_kernel, [], {});
            %estimator = [];
            for i = 1 : length(est)
                if length(sigmaCell{i}) == 1
                    est(i).s_sigma = (sigmaCell{i});
                end
                est(i).c_replicatedVerticallyAlong = {'legendString'};
                %estimator = [estimator; est(i).replicate('ch_type', {'RKHS superposition','kernel superposition'}, [], {}) ];
            end
			
%             % to test reg par
%             v_mu = 10.^(-4);
%             est_mu = MkrGraphFunctionEstimator('s_regularizationParameter',mu, 'ch_type', 'kernel superposition','m_kernel',m_kernel{end});
%             est_mu = est_mu.replicate('s_regularizationParameter', num2cell(v_mu), [], {});
%             est = [est;est_mu];
%             
			%% Simulation
            mse = Simulate(generator, sampler, est, niter);
            
            %% Representation
            F = F_figure('X', bandwidthVec,'Y',mse, ...
                'leg',Parameter.getLegend(generator,sampler, est),...
                'xlab','bandwidth','ylab','Normalized MSE',...
                'tit', sprintf('N=%d, p=%2.2f, \\mu=%3.1d, S = %d', N, p, mu, sampleSize));		  
		end
		
		% Figure: ||alpha_i|| vs \mu
		%      shows sparsity pattern of bandlimited kernels
		%
		function F = compute_fig_3130(obj, niter)
			
			[N,p,SNR,sampleSize,~] = MultikernelSimulations.simulationSetting();
			B = 60;  % used to generate graph signals
			sampleSize = 70;
			u_Vec = logspace(-6,0,50);
			
			s_beta = 1e3; % amplitude parameter of the bandlimited kernel
			v_B_values = 45:5:75; % bandwidth parameter for the bandlimited kernel
			
			% 1. generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
			
			% 2. generate graph function
			functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			v_graphFunction = functionGenerator.realization();
			
			% 3. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR, 's_numberOfSamples',sampleSize);
			
			% 4. generate Kernel matrix
			kG = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',LaplacianKernel.bandlimitedKernelFunctionHandle(graph.getLaplacian(),v_B_values,s_beta));
			m_kernel = kG.getKernelMatrix();
			
			% 5. define function estimator
			estimator = MkrGraphFunctionEstimator('m_kernel', m_kernel);
			estimator = estimator.replicate([],{}, ...
				's_regularizationParameter', num2cell(u_Vec));
			
			[m_samples, m_positions] = sampler.sample(v_graphFunction);
			m_alpha = zeros( length(m_samples), size(m_kernel,3), length(u_Vec) );
			for icount = 1 : length(u_Vec)
				estimator_now = estimator(icount);
				
				[v_graphFunction_estimate, alpha] = estimator_now.estimate(m_samples, m_positions);
				m_alpha(:,:,icount) = alpha;
				
				v_nmse(icount) = norm( v_graphFunction - v_graphFunction_estimate)^2/norm( v_graphFunction )^2;
			end
			
			anorm = sum( m_alpha.^2, 1 );
			anorm = permute(anorm, [3 2 1]);			
			 
            for icount = 1:length(v_B_values)
                legendStr{icount} = sprintf('B = %d',v_B_values(icount));
            end
			
			multiplot_array(1,1) = F_figure('X', u_Vec, 'Y', anorm', 'logx', true, ...
				'xlab', '\mu', 'ylab', '||\alpha_i||^2','leg',legendStr,'leg_pos','East');
			multiplot_array(2,1) = F_figure('X', u_Vec, 'Y', v_nmse, 'logx', true, ...
				'xlab', '\mu', 'ylab', 'NMSE');
			F(1) = F_figure('multiplot_array',multiplot_array);
						
			F(2) = F_figure('Y',graph.getFourierTransform(v_graphFunction)','tit','Fourier transform of target signal','xlab','Freq. index','ylab','Function value');
			
			
		end
				
		% Figure: NMSE vs sigma (diffusion kernel parameter)
		% This figure will show the importance of choosing the right
		%   parameter (sigma for diffusion kernel, may change to other
		%     parameter if different kernel types are used.)
		function F = compute_fig_3101(obj,niter)
			[N,p,SNR,sampleSize,bandwidth] = MultikernelSimulations.simulationSetting();
            S_Vec = 10:10:80;
            mu = 1e-2;
						
			% generate graph and signal
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
			%functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',30);
			functionGenerator = ExponentiallyDecayingGraphFunctionGenerator('graph',graph,'s_bandwidth',30,'s_decayingRate',.5);
			m_graphFunction = functionGenerator.realization();
            generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',m_graphFunction);
			
			% 3. generate Kernel matrix
			sigmaArray = sqrt(linspace(0.01, 1, 30));
			L = graph.getLaplacian();
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray));
			m_kernel = kG.getKernelMatrix();
            
            
			% 4. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);
            sampler = sampler.replicate('s_numberOfSamples', num2cell(S_Vec),[],{}); 
			
			% 5. define function estimator
            estimator = MkrGraphFunctionEstimator('s_regularizationParameter',mu);
            estimator = estimator.replicate([],{}, ...
                'm_kernel', mat2cell(m_kernel, N, N, ones(1,size(m_kernel,3))));
			
			% Simulation
            mse = Simulate(generator, sampler, estimator, niter);
            
            % Representation
            F = F_figure('X',sigmaArray.^2,'Y',mse, ...
                'leg',Parameter.getLegend(generator,sampler, estimator),...
                'xlab','\sigma^2','ylab','Normalized MSE',...
                'tit', sprintf('N=%d, p=%2.2f, \\mu=%3.1d', N, p, mu),...
				'leg_pos','northwest');		  
		end			
		
		% Figure: NMSE vs S (single kernel with different paramter and multi-kernel
        %         with different number of kernels
        function F = compute_fig_3102(obj,niter)
            [N,p,SNR,~,bandwidth] = MultikernelSimulations.simulationSetting();                 
                                             % from Figure 3100 the best kernel
                                             % for bandwidht = 30 is sigma = 0.8
            mu_Vec = 1e-2*ones(5,1);
            S_Vec = 10:10:80;
            
						
			% generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
            
            % generate graph function
			generator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',bandwidth);
% 			m_graphFunction = functionGenerator.realization();
%             generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',m_graphFunction);
 			
            L = graph.getLaplacian();
            
			% define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);
            sampler = sampler.replicate([],{}, 's_numberOfSamples', num2cell(S_Vec)); 
			

			% kernels for single kernel estimators
            sigmaArray = sqrt([0.2 0.80 1 0 0]);
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray(1)));			
            c_kernel{1} = kG.getKernelMatrix();
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray(2)));			
            c_kernel{2} = kG.getKernelMatrix();    
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray(3)));			
            c_kernel{3} = kG.getKernelMatrix();
            
			% kernels for multi-kernel estimators
            sigmaArray2 = sqrt([0.4 0.8 1.2]);
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray2));			
            c_kernel{4} = kG.getKernelMatrix();
            
            sigmaArray20 = sqrt(linspace(0.1,1.5,20)); %[0.1 0.3 0.5 0.8 0.95 1.1 1.3 1.5];
			kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray20));			
            c_kernel{5} = kG.getKernelMatrix();
            
            %c_kernel{4} = kG.getDiffusionKernel(sigmaArray20);
            
            for i = 1 : length(sigmaArray)
                estimator(i,1) = MkrGraphFunctionEstimator('s_regularizationParameter',mu_Vec(i),...
                    's_sigma',sigmaArray(i), 'm_kernel', c_kernel{i}, ...
                    'c_replicatedVerticallyAlong', {'legendString'});
            end
            
            
			% Simulation
            mse = Simulate(generator, sampler, estimator, niter);
            tit = Parameter.getTitle(graphGenerator,generator,sampler, estimator);
			
            % Representation
            F = F_figure('X',S_Vec,'Y',mse, ...
                'leg',Parameter.getLegend(generator,sampler, estimator),...
                'xlab','sample size','ylab','Normalized MSE',...
                'tit', tit);	  
		end
		
		% Simulation to see how IIA works with bandlimited kernels
		% Figure: |theta_i| vs mu for i = 1,..,#kernels
		% Depicts the pattern  of theta in IIA
		% as the regularization paramter mu increases
		function F = compute_fig_3103(obj, niter)
			
            SNR = 20; % dB
			N = 100;
			B = 30; % bandwidth
			S = 80; % number of observed vertices
            u_Vec = logspace(-6,6,50);
			
			s_beta = 10000; % amplitude parameter of the bandlimited kernel
			v_B_values = 10:10:50; % bandwidth parameter for the bandlimited kernel
						
			% 1. generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.5,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
			
            % 2. generate graph function
			functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			v_graphFunction = functionGenerator.realization();
            %generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',m_graphFunction);
			
			% 3. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR, 's_numberOfSamples',S);
			
			% 4. generate Kernel matrix
            kG = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',LaplacianKernel.bandlimitedKernelFunctionHandle(graph.getLaplacian(),v_B_values,s_beta));
			m_kernel = kG.getKernelMatrix();                   
            
            % 5. define function estimator
            estimator = MkrGraphFunctionEstimator('m_kernel', m_kernel,'ch_type','kernel superposition');
            estimator = estimator.replicate([],{}, ...
                's_regularizationParameter', num2cell(u_Vec));
			
            [m_samples, m_positions] = sampler.sample(v_graphFunction);
			m_theta = zeros( size(m_kernel,3), length(u_Vec) );
			v_nmse = zeros( 1 , length(u_Vec) );
			for icount = 1 : length(u_Vec)				
				[v_graphFunction_now,~,m_theta(:,icount)] = estimator(icount).estimate(m_samples, m_positions);				 
				v_nmse(icount) = norm( v_graphFunction - v_graphFunction_now)^2/norm( v_graphFunction )^2;
			end
			
            
            for icount = 1:length(v_B_values)
                legendStr{icount} = sprintf('B = %2.2f',v_B_values(icount));
            end
			
			multiplot_array(1,1) = F_figure('X', u_Vec, 'Y', m_theta, 'logx', true, ...
				'xlab', '\mu', 'ylab', 'Entries of \theta','leg',legendStr,'leg_pos','West');
			multiplot_array(2,1) = F_figure('X', u_Vec, 'Y', v_nmse, 'logx', true, ...
				'xlab', '\mu', 'ylab', 'NMSE');
			F(1) = F_figure('multiplot_array',multiplot_array);
			
			F(2) = F_figure('Y',graph.getFourierTransform(v_graphFunction)','tit','Fourier transform of target signal','xlab','Freq. index','ylab','Function value');
			

		end

		% Simulation to see how MKL with 'RKHS superposition' works with
		% bandlimited kernels  
		% Figure:  ||alpha_i|| vs mu for i = 1,..,#kernels
		% Depicts the sparsity pattern  of theta in IIA
		% as regularization paramter mu increases, theta would become more
		% more sparse
		function F = compute_fig_3104(obj, niter)
			
			SNR = 20; % dB
			N = 100;
			B = 20; % bandwidth
			S = 50; % number of observed vertices
			u_Vec = logspace(-6,0,50);
			
			s_beta = 1e10; % amplitude parameter of the bandlimited kernel
			v_B_values = 10:5:30; % bandwidth parameter for the bandlimited kernel
			
			% 1. generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.5,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
			
			% 2. generate graph function
			functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			v_graphFunction = functionGenerator.realization();
			%generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',m_graphFunction);
			
			% 3. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR, 's_numberOfSamples',S);
			
			% 4. generate Kernel matrix
			kG = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',LaplacianKernel.bandlimitedKernelFunctionHandle(graph.getLaplacian(),v_B_values,s_beta));
			m_kernel = kG.getKernelMatrix();
			
			% 5. define function estimator
			estimator = MkrGraphFunctionEstimator('m_kernel', m_kernel);
			estimator = estimator.replicate([],{}, ...
				's_regularizationParameter', num2cell(u_Vec));
			
			[m_samples, m_positions] = sampler.sample(v_graphFunction);
			m_alpha = zeros( length(m_samples), size(m_kernel,3), length(u_Vec) );
			for icount = 1 : length(u_Vec)
				estimator_now = estimator(icount);
				
				[v_graphFunction_estimate, alpha] = estimator_now.estimate(m_samples, m_positions);
				m_alpha(:,:,icount) = alpha;
				
				v_nmse(icount) = norm( v_graphFunction - v_graphFunction_estimate)^2/norm( v_graphFunction )^2;
			end
			
			anorm = sum( m_alpha.^2, 1 );
			anorm = permute(anorm, [3 2 1]);			
			 
            for icount = 1:length(v_B_values)
                legendStr{icount} = sprintf('B = %d',v_B_values(icount));
            end
			
			multiplot_array(1,1) = F_figure('X', u_Vec, 'Y', anorm', 'logx', true, ...
				'xlab', '\mu', 'ylab', '||\alpha_i||^2','leg',legendStr,'leg_pos','East');
			multiplot_array(2,1) = F_figure('X', u_Vec, 'Y', v_nmse, 'logx', true, ...
				'xlab', '\mu', 'ylab', 'NMSE');
			F(1) = F_figure('multiplot_array',multiplot_array);
						
			F(2) = F_figure('Y',graph.getFourierTransform(v_graphFunction)','tit','Fourier transform of target signal','xlab','Freq. index','ylab','Function value');
			
			
		end


		
		% Figure: NMSE vs sigma
		% like 3100 but with uniform distribution
        function F = compute_fig_3105(obj,niter)		
			[N,p,SNR,sampleSize,~] = MultikernelSimulations.simulationSetting();
            mu = 1e-4;
            bandwidthVec = [5 10 20 30 40];
						
			% generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
            
            % generate signal on this graph
			functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'ch_distribution','uniform');
            functionGenerator.b_generateSameFunction = 0;
            generator = functionGenerator.replicate('s_bandwidth', ...
                num2cell(bandwidthVec), [], {} );
			
			% generate Kernel matrix
			sigmaArray = sqrt(linspace(0.01, 1.5, 30));
            %sigma = 0.8;
			L = graph.getLaplacian();
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray));
			m_kernel = kG.getKernelMatrix();
            
			% define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR, ...
                's_numberOfSamples', sampleSize);
			
			% define function estimator
            estimator = MkrGraphFunctionEstimator('s_regularizationParameter',mu);
            estimator = estimator.replicate([],{}, ...
                'm_kernel', mat2cell(m_kernel, N, N, ones(1,size(m_kernel,3))));
			
			caption = Parameter.getTitle(graphGenerator,generator,sampler,estimator);
			
			% Simulation
            mse = Simulate(generator, sampler, estimator, niter);
            
            % Representation
			
            F = F_figure('X',sigmaArray.^2,'Y',mse, ...
                'leg',Parameter.getLegend(generator,sampler, estimator),...
                'xlab','\sigma^2','ylab','NMSE',...
                'caption',caption,'styles',{'-','--','-^','--^','-*'},'leg_pos_vec',[ 0.3073    0.6041    0.2206    0.3045]);		  

		end
				
		
		% Figure: NMSE vs sigma
		% like 3105 but x axis shows B
        function F = compute_fig_3106(obj,niter)		
			[N,p,SNR,sampleSize,~] = MultikernelSimulations.simulationSetting();
            mu = 1e-4;
            bandwidthVec = [5:5:50];
						
			% generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
            
            % generate signal on this graph
			functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'ch_distribution','uniform');
            functionGenerator.b_generateSameFunction = 0;
            generator = functionGenerator.replicate([], {},'s_bandwidth', ...
                num2cell(bandwidthVec) );
			
			% generate Kernel matrix
			sigmaArray = sqrt([0.1:.2:1.1]);
            %sigma = 0.8;
			L = graph.getLaplacian();
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray));
			m_kernel = kG.getKernelMatrix();
            
			% define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR, ...
                's_numberOfSamples', sampleSize);
			
			% define function estimator
            estimator = MkrGraphFunctionEstimator('s_regularizationParameter',mu);
            estimator = estimator.replicate(...
                'm_kernel', mat2cell(m_kernel, N, N, ones(1,size(m_kernel,3))),[],{});
			
			caption = Parameter.getTitle(graphGenerator,generator,sampler,estimator);
			
			% Simulation
            mse = Simulate(generator, sampler, estimator, niter);
            
            % Representation
			
            F = F_figure('X',bandwidthVec,'Y',mse, ...
                'leg',Parameter.getLegend(generator,sampler),...
                'xlab','B','ylab','NMSE',...
                'caption',caption,'leg_pos_vec',[ 0.3073    0.6041    0.2206    0.3045]);		  

		end
		
		
		% Figure: NMSE vs sigma
		% like 3105 but with exponentially decaying
        function F = compute_fig_3107(obj,niter)		
			[N,p,SNR,sampleSize,~] = MultikernelSimulations.simulationSetting();
			p = 0.5;
            mu = 1e-4;
            bandwidthVec = [5 10 20 30 40];
						
			% generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
            
            % generate signal on this graph
			functionGenerator = ExponentiallyDecayingGraphFunctionGenerator('graph',graph,'s_decayingRate',.5);         
            generator = functionGenerator.replicate('s_bandwidth', ...
                num2cell(bandwidthVec), [], {} );
			
			% generate Kernel matrix
			sigmaArray = sqrt(linspace(0.01, 1.5, 30));
            %sigma = 0.8;
			L = graph.getLaplacian();
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray));
			m_kernel = kG.getKernelMatrix();
            
			% define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR, ...
                's_numberOfSamples', sampleSize);
			
			% define function estimator
            estimator = MkrGraphFunctionEstimator('s_regularizationParameter',mu);
            estimator = estimator.replicate([],{}, ...
                'm_kernel', mat2cell(m_kernel, N, N, ones(1,size(m_kernel,3))));
			
			caption = Parameter.getTitle(graphGenerator,generator,sampler,estimator);
			
			% Simulation
            mse = Simulate(generator, sampler, estimator, niter);
            
            % Representation
			
            F = F_figure('X',sigmaArray.^2,'Y',mse, ...
                'leg',Parameter.getLegend(generator,sampler, estimator),...
                'xlab','\sigma^2','ylab','NMSE',...
                'caption',caption,'styles',{'-','--','-^','--^','-*'},'leg_pos_vec',[ 0.3073    0.6041    0.2206    0.3045]);		  

		end
		
		
		% print version of 3100
		function F = compute_fig_3108(obj,niter)
			F = obj.load_F_structure(3100);
			F.xlab = 'Diffusion kernel parameter';
			F.ylimit = [0 0.7];
		end
		
		
		% 2) Figures for tuning the regularization parameter ==============
		
		% Figure: ||alpha_i|| vs mu
		% Depicts the sparsity pattern  of alpha
		% as regularization paramter mu increases, alpha would become more
		% more sparse, so more and more ||alpha_i|| will go to zero
		function F = compute_fig_3201(obj, niter)
			
            [N,p,SNR,sampleSize,bandwidth] = MultikernelSimulations.simulationSetting();
            u_Vec = logspace(-6,0,50);
						
			% 1. generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
            % 2. generate graph function
			functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',bandwidth);
			m_graphFunction = functionGenerator.realization();
            generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',m_graphFunction);
		
			% 3. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR, 's_numberOfSamples',sampleSize);
			
			% 4. generate Kernel matrix
			sigmaArray = sqrt(linspace(0.01, 1.5, 20));
            %sigmaArray = 0.80;
			L = graph.getLaplacian();
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray));
			m_kernel = kG.getKernelMatrix();
            
            % 5. define function estimator
            estimator = MkrGraphFunctionEstimator('m_kernel', m_kernel);
            estimator = estimator.replicate([],{}, ...
                's_regularizationParameter', num2cell(u_Vec));
			
            [m_samples, m_positions] = sampler.sample(m_graphFunction);
			m_alpha = zeros( length(m_samples), size(m_kernel,3), length(u_Vec) );
			for i = 1 : length(u_Vec)
				estimator_now = estimator(i);
				
				[~, alpha] = estimator_now.estimate(m_samples, m_positions);
				m_alpha(:,:,i) = alpha;
			end
			
			alphanorm = sum( m_alpha.^2, 1 );
			alphanorm = permute(alphanorm, [3 2 1]);
            
            for i = 1:length(sigmaArray)
                legendStr{i} = sprintf('\\sigma=%2.2f',sigmaArray(i));
            end
			
			F = F_figure('X', u_Vec, 'Y', alphanorm', 'logx', true, ...
				'xlab', '\mu', 'ylab', '||\alpha_i||^2','leg',legendStr,'leg_pos','West',...
                'tit',sprintf('N=%d,p=%2.2f,B=%d,S=%d',N,p,bandwidth,sampleSize));

		end
						
		% Figure: NMSE vs mu (regularization parameter)
		% Find the best regularization paramter for each method
		%    To find the best regularization paramter for other methods,
		%    only need to replace the estimator
		function F = compute_fig_3202(obj, niter)
						
            [N,p,SNR,sampleSize,bandwidth] = MultikernelSimulations.simulationSetting();
            u_Vec = logspace(-6,0,50);
						
			% 1. generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
            % 2. generate graph function
			functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',bandwidth);
			m_graphFunction = functionGenerator.realization();
            generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',m_graphFunction);
			
			% 3. generate Kernel matrix
			sigmaArray = linspace(0.1, 1.5, 20);
            %sigmaArray = 0.80;
			L = graph.getLaplacian();
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray));
			m_kernel = kG.getKernelMatrix();
            
            % 4. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR, 's_numberOfSamples',sampleSize);
            
            % 5. define function estimator
            estimator = MkrGraphFunctionEstimator('m_kernel', m_kernel);
            estimator = estimator.replicate([],{}, ...
                's_regularizationParameter', num2cell(u_Vec));
			
			
			% Simulation
            mse = Simulate(generator, sampler, estimator, niter);
			
			F = F_figure('X', u_Vec, 'Y', mse, 'logx', true, ...
				'xlab', '\mu', 'ylab', 'MSE', ...
                'tit', sprintf('N=%d, p=%2.2f, B=%d,S=%d, numOfKernels=%d', ...
                N, p, bandwidth,sampleSize, length(sigmaArray)));
		end
        
		% Simulation to test parameters for Cortes' MKL
		% Figure: |theta_i| vs mu for i = 1,..,#kernels
		% Depicts the pattern  of theta in IIA
		% as regularization paramter mu increases
		function F = compute_fig_3203(obj, niter)
			
            SNR = 20; % dB
			N = 100;
			B = 30; % bandwidth
            u_Vec = logspace(-6,6,50);
						
			% 1. generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.5,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
			
            % 2. generate graph function
			functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			v_graphFunction = functionGenerator.realization();
            %generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',m_graphFunction);
			
			% 3. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR, 's_numberOfSamples',50);
			
			% 4. generate Kernel matrix
			sigmaArray = linspace(0.01, 1.5, 20);            			
            kG = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray));
			m_kernel = kG.getKernelMatrix();                   
            
            % 5. define function estimator
            estimator = MkrGraphFunctionEstimator('m_kernel', m_kernel,'ch_type','kernel superposition');
            estimator = estimator.replicate([],{}, ...
                's_regularizationParameter', num2cell(u_Vec));
			
            [m_samples, m_positions] = sampler.sample(v_graphFunction);
			m_theta = zeros( size(m_kernel,3), length(u_Vec) );
			v_nmse = zeros( 1 , length(u_Vec) );
			for icount = 1 : length(u_Vec)				
				[v_graphFunction_now,~,m_theta(:,icount)] = estimator(icount).estimate(m_samples, m_positions);				 
				v_nmse(icount) = norm( v_graphFunction - v_graphFunction_now)^2/norm( v_graphFunction )^2;
			end
			
            
            for icount = 1:length(sigmaArray)
                legendStr{icount} = sprintf('\\sigma=%2.2f',sigmaArray(icount));
            end
			
			multiplot_array(1,1) = F_figure('X', u_Vec, 'Y', m_theta, 'logx', true, ...
				'xlab', '\mu', 'ylab', 'Entries of \theta','leg',legendStr,'leg_pos','West');
			multiplot_array(2,1) = F_figure('X', u_Vec, 'Y', v_nmse, 'logx', true, ...
				'xlab', '\mu', 'ylab', 'NMSE');
			F = F_figure('multiplot_array',multiplot_array);

        end
        
        %
        % Estimate bandwidth of bandlimited graph signals
        %     Use mkl estimator to find the best kernel, which is generated using
        %     the corresponding bandlimited kernels
        function F = compute_fig_3232(obj, niter)
						
            [~,p,~] = MultikernelSimulations.simulationSetting();
            %u_Vec = logspace(-6,0,50);
			SNR = 20;
			sampleSize = 80;
			N = 250;
            
						
			% 1. generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
            % 2. generate graph function
			functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'ch_distribution','uniform');
			functionGenerator.b_sortedSpectrum = 0;
			functionGenerator.b_generateSameFunction = 0;
			%m_graphFunction = functionGenerator.realization();
            %generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',m_graphFunction);
			
			% 3. generate Kernel matrix
			%sigmaArray = linspace(0.1, 1.5, 20);
            B_vec = 10:5:90;
            beta = 1e3;   % for bandlimited kernel
            %sigmaArray = 0.80;
			L = graph.getLaplacian();
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.bandlimitedKernelFunctionHandle(L, B_vec, beta));
			m_kernel = kG.getKernelMatrix();
            
            % 4. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR, 's_numberOfSamples',sampleSize);
            
            % 5. define function estimator
            estimator = MkrGraphFunctionEstimator('m_kernel', m_kernel, 's_regularizationParameter', 1e-2);
            estimator.b_estimateFreq = 1;
            %estimator = estimator.replicate([],{}, ...
            %    's_regularizationParameter', num2cell(u_Vec));
			
			
			% Simulation
            bandwidth_vec = 10:10:60;
            estimated_bandwidth = zeros(length(bandwidth_vec),niter);
            for iBand = 1:length(bandwidth_vec)
				functionGenerator.s_bandwidth = bandwidth_vec(iBand);
                for iter = 1:niter                    
                    v_graphFunction = functionGenerator.realization();
                    [m_samples, m_positions] = sampler.sample(v_graphFunction);
                    [~,~,~, main_kernel_ind] = estimator.estimate(m_samples, m_positions);
					if isnan(main_kernel_ind)
						disp('discarding realization');
						estimated_bandwidth(iBand, iter) = NaN;
					else
						estimated_bandwidth(iBand, iter) = B_vec(main_kernel_ind);
					end
                end
                MultikernelSimulations.printSimulationProgress(iBand, iter, length(bandwidth_vec), niter)
			end
			est_bandwidth_mean = mean(estimated_bandwidth, 2, 'omitnan');
			est_bandwidth = abs(est_bandwidth_mean-bandwidth_vec');
            %mse = Simulate(generator, sampler, estimator, niter);
			est_std = std(estimated_bandwidth', 'omitnan');
			
			% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			% print the table into a tex file
			% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			fid = fopen('libGF/simulations/MultikernelSimulations_data/est_freq.tex','w');
			fprintf(fid, '\\begin{tabular}{%s}\n', char('c'*ones(1,1+length(est_bandwidth))));     % heading line
			fprintf(fid, '\t\\hline\n\t\t');
			for i = 1:length(bandwidth_vec)
				fprintf(fid, ' & B = %d', bandwidth_vec(i));
			end
			fprintf(fid, '\t\\hline\n\t\t');
			
			% print mean
			fprintf(fid, '\\\\\n\tBIAS\t');
			for i = 1:length(est_bandwidth)
				fprintf(fid, ' & %2.1f', est_bandwidth(i));
			end
			fprintf(fid, '\\\\\n\tSTD\t');
			% print variance
			for i = 1:length(est_std)
				fprintf(fid, ' & %2.1f', est_std(i));
			end
			fprintf(fid, '\\\\\n');
			fprintf(fid, '\t\\hline\n');
			fprintf(fid, '\\end{tabular}');		% bottom line
			caption = Parameter.getTitle(graphGenerator,functionGenerator,sampler,estimator);
			fprintf(fid, caption);
			% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			
			
			F(1) = F_figure('X', bandwidth_vec, 'Y', [bandwidth_vec; est_bandwidth_mean'], ...
				'xlab', 'experiment index', 'ylab', 'bandwidth', ...
                'tit', sprintf('N=%d, p=%2.2f, S=%d, numOfKernels=%d', ...
                N, p, sampleSize, length(B_vec)), ...
                'leg',{'true bandwidth','estimated bandwidth'});
			F(2) = F_figure('Y',estimated_bandwidth);
		end
        
          %
        % sparsity paths for 3232
        function F = compute_fig_3233(obj, niter)
						
            [~,p,~] = MultikernelSimulations.simulationSetting();
            %u_Vec = logspace(-6,0,50);
			SNR = 20;
			sampleSize = 80;
			N = 250;
            
						
			% 1. generate graph
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p,'s_numberOfVertices',N);
			graph = graphGenerator.realization();
            % 2. generate graph function
			functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'ch_distribution','uniform');
			functionGenerator.b_sortedSpectrum = 0;
			functionGenerator.b_generateSameFunction = 0;
			%m_graphFunction = functionGenerator.realization();
            %generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',m_graphFunction);
			
			% 3. generate Kernel matrix
			%sigmaArray = linspace(0.1, 1.5, 20);
            B_vec = 10:5:90;
            beta = 1e3;   % for bandlimited kernel
            %sigmaArray = 0.80;
			L = graph.getLaplacian();
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.bandlimitedKernelFunctionHandle(L, B_vec, beta));
			m_kernel = kG.getKernelMatrix();
            
            % 4. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR, 's_numberOfSamples',sampleSize);
            
            % 5. define function estimator
            estimator = MkrGraphFunctionEstimator('m_kernel', m_kernel, 's_regularizationParameter', 1e-2);
            estimator.b_estimateFreq = 1;
            %estimator = estimator.replicate([],{}, ...
            %    's_regularizationParameter', num2cell(u_Vec));
			
			
			% Simulation
			functionGenerator.s_bandwidth = 20;
			
			v_graphFunction = functionGenerator.realization();
			[m_samples, m_positions] = sampler.sample(v_graphFunction);
			
			v_mu = 10.^(-7:.5:0);
			for k = 1:length(v_mu)
				estimator.s_regularizationParameter = v_mu(k);
				[~,m_alpha,~, main_kernel_ind] = estimator.estimate(m_samples, m_positions);
				alphanorm(:,k) = sum( m_alpha.^2, 1 )';
			end
						            
            for i = 1:length(B_vec)
                legendStr{i} = sprintf('B = %2.2f',B_vec(i));
            end
			styles = {'-','-.','--',':','-v','-.v','--v',':v','-o','-.o','--o',':o','-s','-.s','--s',':s','-s','-.s','--s',':s','-s','-.s','--s',':s','-s','-.s','--s',':s','-s','-.s','--s',':s'};
			F = F_figure('X', v_mu, 'Y', alphanorm, 'logx', true, 'xlimit',[min(v_mu) max(v_mu)],...
				'xlab', 'Regularization parameter (\mu)', 'ylab', '||\alpha_i||^2','leg',legendStr(1:4),'leg_pos','northwest','styles',styles,'colorp',10);

		end
        
        % print version of 3233
		function F = compute_fig_3234(obj, niter)
			F = obj.load_F_structure(3233);
			F.leg_pos = 'northeast';
			F.xlab = 'Regularization parameter';
			F.ylab = '||\bf\alpha_{\rm m}\rm||^2';
			B_vec = 10:5:90;
			for i = 1:4
				legendStr{i} = sprintf('m = %d (B = %2.2f)',i, B_vec(i));
            end
			F.leg = legendStr;
		end
 	
		% 3) Figures to compare MKL and bandlimited =======================
		
		% Simple MC simulation to test MKL methods and compare them with
		% bandlimited estimators
		% - bandlimited signal, but MC does not average across signal
		%   realizations
		function F = compute_fig_3301(obj,niter)
						
			N = 100; % number of vertices			
			B = 20; % bandwidth of the estimated function
			B_vec =         [20]; % assumed bandwidth for estimation
			SNR_vec = [25 25 25 25]; % SNR for each curve (first 2 for multikernel)
			
			S_vec = 10:10:100;
			
			% 1. define graph function generator
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.9,'s_numberOfVertices',N);
			graph = graphGenerator.realization;
			m_laplacian = graph.getLaplacian(); 
			bandlimitedFunctionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			graphFunction = bandlimitedFunctionGenerator.realization();
			generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',graphFunction);			
			
			% 2. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',20);			
			sampler = sampler.replicate('s_SNR',num2cell(SNR_vec),'s_numberOfSamples',num2cell(S_vec));		
						
			% 3. BL graph function estimator
			bl_estimator_known_freq = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian);			
			bl_estimator_known_freq.c_replicatedVerticallyAlong = {'ch_name'};
			bl_estimator_known_freq = bl_estimator_known_freq.replicate('s_bandwidth',num2cell(B_vec),'',{});
					
			% 4. BL estimator with unknown frequency
			bl_estimator_unknown_freq = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian,'s_bandwidth',-1);			
			bl_estimator_unknown_freq.c_replicatedVerticallyAlong = {'ch_name','s_bandwidth'};
						
			% 5. MKL function estimators		    
			sigma2Array = linspace(0.1, .5 , 20);            
            kG = LaplacianKernel('m_laplacian',m_laplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigma2Array));
			m_kernel = kG.getKernelMatrix();
			mkl_estimator = MkrGraphFunctionEstimator('m_kernel',m_kernel,'s_regularizationParameter',1e-3);
			mkl_estimator.c_replicatedVerticallyAlong = {'ch_name'};			
			mkl_estimator = mkl_estimator.replicate('ch_type',{'RKHS superposition','kernel superposition'},'',[]);

			est = [mkl_estimator;bl_estimator_known_freq;bl_estimator_unknown_freq];
			
			% Simulation
			res = Simulator.simStatistic(niter,generator,sampler,est);
			mse = Simulator.computeNmse(res,Results('stat',graphFunction));
			
			% Representation			
			F = F_figure('X',Parameter.getXAxis(generator,sampler,est),...
                'Y',mse,'leg',Parameter.getLegend(generator,sampler,est),...
                'xlab',Parameter.getXLabel(generator,sampler,est),'ylimit',...
				[0 1.5],'ylab','NMSE','tit',Parameter.getTitle(graphGenerator,bandlimitedFunctionGenerator,generator,sampler));
			
		end
				
		% MC simulation to compare MKL and bandlimited estimators 
		% - bandlimited signal, but MC does not average across signal
		%   realizations
		function F = compute_fig_3302(obj,niter)
						
			N = 100; % number of vertices			
			B = 20; % bandwidth of the true function
			B_vec =         [10 20 30 -1]; % assumed bandwidth for estimation
			SNR = 5; % dB
			
			S_vec = 10:10:100;
			
			% 1. define graph function generator
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.7 ,'s_numberOfVertices',N);
			graph = graphGenerator.realization;
			m_laplacian = graph.getLaplacian(); 
			bandlimitedFunctionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			graphFunction = bandlimitedFunctionGenerator.realization();
			generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',graphFunction);			
			
			% 2. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);			
			sampler = sampler.replicate('',[],'s_numberOfSamples',num2cell(S_vec));		
						
			% 3. BL graph function estimator
			bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian);			
			bl_estimator.c_replicatedVerticallyAlong = {'ch_name'};
			bl_estimator = bl_estimator.replicate('s_bandwidth',num2cell(B_vec),'',{});
								
			% 4. MKL function estimators		    
			sigma2Array = linspace(0.1, .5 , 20);            
            kG = LaplacianKernel('m_laplacian',m_laplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigma2Array));
			mkl_estimator = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',5e-3);
			mkl_estimator.c_replicatedVerticallyAlong = {'ch_name'};			
			mkl_estimator = mkl_estimator.replicate('ch_type',{'RKHS superposition','kernel superposition'},'',[]);

			est = [mkl_estimator;bl_estimator];
			
			% Simulation
			res = Simulator.simStatistic(niter,generator,sampler,est);
			mse = Simulator.computeNmse(res,Results('stat',graphFunction));
			
			% Representation			
			F = F_figure('X',Parameter.getXAxis(generator,sampler,est),...
                'Y',mse,'leg',Parameter.getLegend(generator,sampler,est),...
                'xlab',Parameter.getXLabel(generator,sampler,est),'ylimit',...
				[0 1.5],'ylab','NMSE','tit',Parameter.getTitle(graphGenerator,bandlimitedFunctionGenerator,generator,sampler));
			
		end
		
		% MC simulation to compare MKL and bandlimited estimators
		% - bandlimited signal, but MC does AVERAGE across signal
		%   realizations (INCOMPLETE)
		function F = compute_fig_3303(obj,niter)
						
			N = 100; % number of vertices			
			B = 20; % bandwidth of the true function
			B_vec =         [10 20 30 -1]; % assumed bandwidth for estimation
			SNR = 5; % dB
			
			S_vec = 10:10:100;
			
			% 1. define graph function generator
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.7 ,'s_numberOfVertices',N);
			graph = graphGenerator.realization;
			m_laplacian = graph.getLaplacian(); 
			generator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);			
			
			% 2. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);			
			sampler = sampler.replicate('',[],'s_numberOfSamples',num2cell(S_vec));		
						
			% 3. BL graph function estimator
			bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian);			
			bl_estimator.c_replicatedVerticallyAlong = {'ch_name'};
			bl_estimator = bl_estimator.replicate('s_bandwidth',num2cell(B_vec),'',{});
								
			% 4. MKL function estimators		    
			sigma2Array = linspace(0.1, .5 , 20);            
            kG = LaplacianKernel('m_laplacian',m_laplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigma2Array));
			mkl_estimator = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',5e-3);
			mkl_estimator.c_replicatedVerticallyAlong = {'ch_name'};			
			mkl_estimator = mkl_estimator.replicate('ch_type',{'RKHS superposition','kernel superposition'},'',[]);

			est = [mkl_estimator;bl_estimator];
			
			% Simulation
			res = Simulator.simStatistic(niter,generator,sampler,est);
			mse = Simulator.computeNmse(res,Results('stat',graphFunction));
			
			% Representation			
			F = F_figure('X',Parameter.getXAxis(generator,sampler,est),...
                'Y',mse,'leg',Parameter.getLegend(generator,sampler,est),...
                'xlab',Parameter.getXLabel(generator,sampler,est),'ylimit',...
				[0 1.5],'ylab','NMSE','tit',Parameter.getTitle(graphGenerator,generator,sampler));
			
		end

		% MC simulation to compare MKL and bandlimited estimators 
		% - signal with exp. decaying spectrum. MC does not average across
		%   signal realizations
		function F = compute_fig_3304(obj,niter)
						
			N = 100; % number of vertices			
			B = 20; % bandwidth of the true function
			B_vec =         [10 20 30 -1]; % assumed bandwidth for estimation
			SNR = 5; % dB
			s_decayingRate = .5; % for decaying spectrum
			
			S_vec = 10:10:100;
			
			% 1. define graph function generator
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.7 ,'s_numberOfVertices',N);
			graph = graphGenerator.realization;
			m_laplacian = graph.getLaplacian(); 
			bandlimitedFunctionGenerator = ExponentiallyDecayingGraphFunctionGenerator('graph',graph,'s_bandwidth',B,'s_decayingRate',s_decayingRate);
			graphFunction = bandlimitedFunctionGenerator.realization();
			generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',graphFunction);			
			
			% 2. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);			
			sampler = sampler.replicate('',[],'s_numberOfSamples',num2cell(S_vec));		
						
			% 3. BL graph function estimator
			bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian);			
			bl_estimator.c_replicatedVerticallyAlong = {'ch_name'};
			bl_estimator = bl_estimator.replicate('s_bandwidth',num2cell(B_vec),'',{});
								
			% 4. MKL function estimators		    
			sigma2Array = linspace(0.1, .5 , 20);            
            kG = LaplacianKernel('m_laplacian',m_laplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigma2Array));
			mkl_estimator = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',5e-3);
			mkl_estimator.c_replicatedVerticallyAlong = {'ch_name'};			
			mkl_estimator = mkl_estimator.replicate('ch_type',{'RKHS superposition','kernel superposition'},'',[]);

			est = [mkl_estimator;bl_estimator];
			
			% Simulation
			res = Simulator.simStatistic(niter,generator,sampler,est);
			mse = Simulator.computeNmse(res,Results('stat',graphFunction));
			
			% Representation			
			F(1) = F_figure('X',Parameter.getXAxis(generator,sampler,est),...
                'Y',mse,'leg',Parameter.getLegend(generator,sampler,est),...
                'xlab',Parameter.getXLabel(generator,sampler,est),'ylimit',...
				[0 1.5],'ylab','NMSE','tit',Parameter.getTitle(graphGenerator,bandlimitedFunctionGenerator,generator,sampler),'leg_pos','southwest');
			F(2) = F_figure('Y',graph.getFourierTransform(graphFunction)','tit','Fourier transform of target signal','xlab','Freq. index','ylab','Function value');
			
		end
		
		
		% 4) Figures illustrate bandlimited kernels =======================
		
		% MC simulation to compare BL estimators and MKL estimators with BL
		% kernels. 
		% - bandlimited signal, but MC does not average across signal
		%   realizations
		function F = compute_fig_3401(obj,niter)
						
			N = 100; % number of vertices			
			B = 20; % bandwidth of the true function
			B_vec = [10 20 30 -1]; % assumed bandwidth for estimation
			SNR = 10; % dB
			S_vec = 10:5:100;
			
			s_beta = 1e5; % amplitude parameter of the bandlimited kernel
			v_B_values = 10:5:30; % bandwidth parameter for the bandlimited kernel
						
			% 1. define graph function generator
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.7 ,'s_numberOfVertices',N);
			graph = graphGenerator.realization;			
			bandlimitedFunctionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			graphFunction = bandlimitedFunctionGenerator.realization();
			generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',graphFunction);			
			
			% 2. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);			
			sampler = sampler.replicate('',[],'s_numberOfSamples',num2cell(S_vec));		
						
			% 3. BL graph function estimator
			bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian);			
			bl_estimator.c_replicatedVerticallyAlong = {'ch_name'};
			bl_estimator = bl_estimator.replicate('s_bandwidth',num2cell(B_vec),'',{});
								
			% 4. MKL function estimators	
			% 4. generate Kernel matrix
			kG = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',LaplacianKernel.bandlimitedKernelFunctionHandle(graph.getLaplacian(),v_B_values,s_beta));			
			mkl_estimator = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',5e-3);
			mkl_estimator.c_replicatedVerticallyAlong = {'ch_name'};			
			mkl_estimator = mkl_estimator.replicate('ch_type',{'RKHS superposition','kernel superposition'},'',[]);
			est = [mkl_estimator;bl_estimator];
			
			% Simulation
			res = Simulator.simStatistic(niter,generator,sampler,est);
			mse = Simulator.computeNmse(res,Results('stat',graphFunction));
			
			% Representation			
			F(1) = F_figure('X',Parameter.getXAxis(generator,sampler,est),...
                'Y',mse,'leg',Parameter.getLegend(generator,sampler,est),...
                'xlab',Parameter.getXLabel(generator,sampler,est),'ylimit',...
				[0 1.5],'ylab','NMSE','tit',Parameter.getTitle(graphGenerator,bandlimitedFunctionGenerator,generator,sampler));
			F(2) = F_figure('Y',graph.getFourierTransform(graphFunction)','tit','Fourier transform of target signal','xlab','Freq. index','ylab','Function value');
			
		end
		
		
		% MC simulation to compare BL estimators and MKL estimators with BL
		% kernels. 
		% - bandlimited signal,
		function F = compute_fig_3402(obj,niter)
						
			N = 100; % number of vertices			
			B = 20; % bandwidth of the true function
			B_vec = [10 20 30 -1]; % assumed bandwidth for estimation
			SNR = 10; % dB
			S_vec = 10:5:100;
			
			s_beta = 1e4; % amplitude parameter of the bandlimited kernel
			v_B_values = 10:5:30; % bandwidth parameter for the bandlimited kernel
						
			% 1. define graph function generator
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.25 ,'s_numberOfVertices',N);
			graph = graphGenerator.realization;			
			generator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			%graphFunction = bandlimitedFunctionGenerator.realization();
			%generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',graphFunction);			
			
			% 2. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);			
			sampler = sampler.replicate('',[],'s_numberOfSamples',num2cell(S_vec));		
						
			% 3. BL graph function estimator
			bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian);			
			bl_estimator.c_replicatedVerticallyAlong = {'ch_name'};
			bl_estimator = bl_estimator.replicate('s_bandwidth',num2cell(B_vec),'',{});
								
			% 4. MKL function estimators	
			% 4. generate Kernel matrix
			kG = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',LaplacianKernel.bandlimitedKernelFunctionHandle(graph.getLaplacian(),v_B_values,s_beta));			
			%mkl_estimator(1,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',5e-3,'ch_type','RKHS superposition');
			%mkl_estimator(2,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',1e-2,'ch_type','RKHS superposition');
			mkl_estimator(1,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',.05,'ch_type','RKHS superposition');
			%mkl_estimator(4,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',1e-3,'ch_type','RKHS superposition');
			mkl_estimator(2,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',.01,'ch_type','kernel superposition');
			mkl_estimator(1,1).c_replicatedVerticallyAlong = {'ch_name','ch_type'};			
			mkl_estimator(2,1).c_replicatedVerticallyAlong = {'ch_name','ch_type'};			
			%mkl_estimator = mkl_estimator.replicate('ch_type',{'RKHS superposition','kernel superposition'},'',[]);
            est = [mkl_estimator;bl_estimator];
			

			% Simulation
			%res = Simulator.simStatistic(niter,generator,sampler,est);
			%mse = Simulator.computeNmse(res,Results('stat',graphFunction));
			mse =  Simulate(generator,sampler,est,niter);
			
			% Representation			
			F = F_figure('X',Parameter.getXAxis(generator,sampler,est),...
                'Y',mse,'leg',Parameter.getLegend(generator,sampler,est),'leg_pos','northwest',...
                'xlab','Number of observed vertices (S)','ylimit',[0 1.5],'xlimit',[min(Parameter.getXAxis(generator,sampler,est)),max(Parameter.getXAxis(generator,sampler,est))],...
				'ylab','NMSE','caption',Parameter.getTitle(graphGenerator,generator,sampler,est),'styles',{'-v','-^','--x','--o','--s','-.'});
			%F(2) = F_figure('Y',graph.getFourierTransform(graphFunction)','tit','Fourier transform of target signal','xlab','Freq. index','ylab','Function value');
			
		end
		
		
		% MC simulation to compare BL estimators and MKL estimators with BL
		% kernels. 
		% - bandlimited signal, but MC does not average across signal
		%   realizations
		% - single kernel finish implements CV		
		function F = compute_fig_3403(obj,niter)
            [N,p,SNR,sampleSize, bandwidth] = MultikernelSimulations.simulationSetting();
            B = 20;
			B_vec = [10 20 30 -1]; % assumed bandwidth for estimation
			S_vec = 10:5:100;      
			
			s_beta = 1e5; % amplitude parameter of the bandlimited kernel
			v_B_values = 10:5:30; % bandwidth parameter for the bandlimited kernel
						
			% 1. define graph function generator
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p ,'s_numberOfVertices',N);
			graph = graphGenerator.realization;			
			generator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			
			% 2. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);			
			sampler = sampler.replicate('',[],'s_numberOfSamples',num2cell(S_vec));		
						
			% 3. BL graph function estimator
			bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian);			
			bl_estimator.c_replicatedVerticallyAlong = {'ch_name'};
			bl_estimator = bl_estimator.replicate('s_bandwidth',num2cell(B_vec),'',{});
								
			% 4. MKL function estimators	
			% 4. generate Kernel matrix
			kG = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',LaplacianKernel.bandlimitedKernelFunctionHandle(graph.getLaplacian(),v_B_values,s_beta));			
			mkl_estimator = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',1e-2);
			mkl_estimator.c_replicatedVerticallyAlong = {'ch_name'};			
			mkl_estimator = mkl_estimator.replicate('ch_type',{'RKHS superposition','RKHS superposition','kernel superposition'},'',[]);
            v_regPar = 10.^(-6:0);
			mkl_estimator(1).singleKernelPostEstimator= RidgeRegressionGraphFunctionEstimator('s_regularizationParameter',v_regPar);			
			mkl_estimator(1).c_replicatedVerticallyAlong = [mkl_estimator(1).c_replicatedVerticallyAlong,'s_regularizationParameter'];
			est = [mkl_estimator;bl_estimator];
			
			% Simulation
			%res = Simulator.simStatistic(niter,generator,sampler,est);
			%mse = Simulator.computeNmse(res,Results('stat',graphFunction));
            mse = Simulate(generator, sampler, est, niter);
			
			% Representation			
			F(1) = F_figure('X',Parameter.getXAxis(generator,sampler,est),...
                'Y',mse,'leg',Parameter.getLegend(generator,sampler,est),...
                'xlab',Parameter.getXLabel(generator,sampler,est),'ylimit', [0 1.5],...
				'ylab','NMSE','tit',Parameter.getTitle(graphGenerator,generator,sampler));
			%F(2) = F_figure('Y',graph.getFourierTransform(graphFunction)','tit','Fourier transform of target signal','xlab','Freq. index','ylab','Function value');
			
		end
		
		% print version of 3402
		function F = compute_fig_3404(obj,niter)
			F = obj.load_F_structure(3402);
			F.translation_table = {'kernel superposition','KS';'RKHS superposition','RS';'Bandlimited','BL';'Ass. B = cut-off freq.','cut-off'; ...
				'BL, Ass.','BL for'};
			F.leg_pos = 'northeast';
		end
		
		% for tuning parameters of 3402
		function F = compute_fig_3405(obj,niter)
						
			N = 100; % number of vertices			
			B = 20; % bandwidth of the true function
			B_vec = [10 20 30 -1]; % assumed bandwidth for estimation
			SNR = 10; % dB
			S_vec = 10:5:100;
			
			s_beta = 1e4; % amplitude parameter of the bandlimited kernel
			v_B_values = 10:5:30; % bandwidth parameter for the bandlimited kernel
						
			% 1. define graph function generator
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.25 ,'s_numberOfVertices',N);
			graph = graphGenerator.realization;			
			generator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			%graphFunction = bandlimitedFunctionGenerator.realization();
			%generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',graphFunction);			
			
			% 2. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);			
			sampler = sampler.replicate('',[],'s_numberOfSamples',num2cell(S_vec));		
						
			% 3. BL graph function estimator
			bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian);			
			bl_estimator.c_replicatedVerticallyAlong = {'ch_name'};
			bl_estimator = bl_estimator.replicate('s_bandwidth',num2cell(B_vec),'',{});
								
			% 4. MKL function estimators	
			% 4. generate Kernel matrix
			kG = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',LaplacianKernel.bandlimitedKernelFunctionHandle(graph.getLaplacian(),v_B_values,s_beta));			
			%mkl_estimator(1,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',5e-3,'ch_type','RKHS superposition');
			%mkl_estimator(2,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',1e-2,'ch_type','RKHS superposition');
			mkl_estimator(1,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',5e-3,'ch_type','RKHS superposition');
			%mkl_estimator(4,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',1e-3,'ch_type','RKHS superposition');			
			mkl_estimator(1,1).c_replicatedVerticallyAlong = {'ch_name','ch_type'};			
			
			mkl_estimator = mkl_estimator.replicate('s_regularizationParameter',num2cell([.05 10.^(-5:0)]),'',[]);
            est = [mkl_estimator];
			

			% Simulation
			%res = Simulator.simStatistic(niter,generator,sampler,est);
			%mse = Simulator.computeNmse(res,Results('stat',graphFunction));
			mse =  Simulate(generator,sampler,est,niter);
			
			% Representation			
			F = F_figure('X',Parameter.getXAxis(generator,sampler,est),...
                'Y',mse,'leg',Parameter.getLegend(generator,sampler,est),'leg_pos','northwest',...
                'xlab','Number of observed vertices (S)','ylimit',[0 1.5],'xlimit',[min(Parameter.getXAxis(generator,sampler,est)),max(Parameter.getXAxis(generator,sampler,est))],...
				'ylab','NMSE','caption',Parameter.getTitle(graphGenerator,generator,sampler,est));
			%F(2) = F_figure('Y',graph.getFourierTransform(graphFunction)','tit','Fourier transform of target signal','xlab','Freq. index','ylab','Function value');
			
		end
		
		
		% Like 3402 but with uniform BL signals
		function F = compute_fig_3406(obj,niter)
						
			N = 100; % number of vertices			
			B = 20; % bandwidth of the true function
			B_vec = [10 20 30 -1]; % assumed bandwidth for estimation
			SNR = 10; % dB
			S_vec = 10:5:100;
			
			s_beta = 1e4; % amplitude parameter of the bandlimited kernel
			v_B_values = 10:5:30; % bandwidth parameter for the bandlimited kernel
						
			% 1. define graph function generator
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.25 ,'s_numberOfVertices',N);
			graph = graphGenerator.realization;			
			generator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B,'ch_distribution','uniform');
			%graphFunction = bandlimitedFunctionGenerator.realization();
			%generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',graphFunction);			
			
			% 2. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);			
			sampler = sampler.replicate('',[],'s_numberOfSamples',num2cell(S_vec));		
						
			% 3. BL graph function estimator
			bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian);			
			bl_estimator.c_replicatedVerticallyAlong = {'ch_name'};
			bl_estimator = bl_estimator.replicate('s_bandwidth',num2cell(B_vec),'',{});
								
			% 4. MKL function estimators	
			% 4. generate Kernel matrix
			kG = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',LaplacianKernel.bandlimitedKernelFunctionHandle(graph.getLaplacian(),v_B_values,s_beta));			
			%mkl_estimator(1,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',5e-3,'ch_type','RKHS superposition');
			%mkl_estimator(2,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',1e-2,'ch_type','RKHS superposition');
			mkl_estimator(1,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',.05,'ch_type','RKHS superposition');
			%mkl_estimator(4,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',1e-3,'ch_type','RKHS superposition');
			mkl_estimator(2,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',.01,'ch_type','kernel superposition');
			mkl_estimator(1,1).c_replicatedVerticallyAlong = {'ch_name','ch_type'};			
			mkl_estimator(2,1).c_replicatedVerticallyAlong = {'ch_name','ch_type'};			
			%mkl_estimator = mkl_estimator.replicate('ch_type',{'RKHS superposition','kernel superposition'},'',[]);
            est = [mkl_estimator;bl_estimator];
			

			% Simulation
			%res = Simulator.simStatistic(niter,generator,sampler,est);
			%mse = Simulator.computeNmse(res,Results('stat',graphFunction));
			mse =  Simulate(generator,sampler,est,niter);
			
			% Representation			
			F = F_figure('X',Parameter.getXAxis(generator,sampler,est),...
                'Y',mse,'leg',Parameter.getLegend(generator,sampler,est),'leg_pos','northwest',...
                'xlab','Number of observed vertices (S)','ylimit',[0 1.5],'xlimit',[min(Parameter.getXAxis(generator,sampler,est)),max(Parameter.getXAxis(generator,sampler,est))],...
				'ylab','NMSE','caption',Parameter.getTitle(graphGenerator,generator,sampler,est),'styles',{'-v','-^','--x','--o','--s','-.'});
			%F(2) = F_figure('Y',graph.getFourierTransform(graphFunction)','tit','Fourier transform of target signal','xlab','Freq. index','ylab','Function value');
			
		end
		
		
		% 5) Figures to illustrate MKL with heterogeneous dictionaries ====
		
		% MC simulation to compare multiple kernel dictionaries
		%      NMSE vs S
		% - generated signal is bandlimited
		function F = compute_fig_3501(obj,niter)
						
			% 1. Graph and signal 
			N = 100; % number of vertices			
			B = 20; % bandwidth of the true function
			
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.25 ,'s_numberOfVertices',N);
			graph = graphGenerator.realization;			
			generator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',B);
			%graphFunction = bandlimitedFunctionGenerator.realization();
			%generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',graphFunction);			
						
			% 2. Sampler
			SNR = 10; % dB
			S_vec = 10:5:100;
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);			
			sampler = sampler.replicate('',[],'s_numberOfSamples',num2cell(S_vec));		
						
			% 3. Estimators
			sigmaArray_diffusion = linspace(0.1,.5,5);
			sigmaArray_regularized = linspace(0.1,.5,5);
						
			% 4. MKL function estimators	
			% 4. generate Kernel matrix
			
			h_r_inv_diffusion = LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray_diffusion);
			h_r_inv_regularized = LaplacianKernel.regularizedKernelFunctionHandle(sigmaArray_regularized);
			dictionary_diffusion = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',h_r_inv_diffusion);			
			dictionary_regularized = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',h_r_inv_regularized);			
			dictionary_both = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',[h_r_inv_diffusion, h_r_inv_regularized]);			
			%%
			% *BOLD TEXT* );
			%mkl_estimator(1,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',5e-3,'ch_type','RKHS superposition');
			%mkl_estimator(2,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',1e-2,'ch_type','RKHS superposition');
			rs_estimators = MkrGraphFunctionEstimator('s_regularizationParameter',.05,'ch_type','RKHS superposition');
			rs_estimators = rs_estimators.replicate('m_kernel',{dictionary_diffusion.getKernelMatrix(),dictionary_regularized.getKernelMatrix(),dictionary_both.getKernelMatrix()},'',{});
			
			ks_estimators = MkrGraphFunctionEstimator('s_regularizationParameter',.05,'ch_type','kernel superposition');
			ks_estimators = ks_estimators.replicate('m_kernel',{dictionary_diffusion.getKernelMatrix(),dictionary_regularized.getKernelMatrix(),dictionary_both.getKernelMatrix()},'',{});
			
            est = [rs_estimators;ks_estimators];
			
			for k = 1:6
				est(k,1).c_replicatedVerticallyAlong = {'ch_type'};			
			end			
			leg = Parameter.getLegend(generator,sampler,est);
			for k = 1:6
				leg{k} = [leg{k}, sprintf(', D%d',mod(k-1,3)+1) ];
			end
			

			% Simulation
			mse =  Simulate(generator,sampler,est,niter);
			
			% Representation			
			F = F_figure('X',Parameter.getXAxis(generator,sampler,est),...
                'Y',mse,'leg',leg,'leg_pos','northwest',...
                'xlab','Number of observed vertices (S)','ylimit',[0 1.5],'xlimit',[min(Parameter.getXAxis(generator,sampler,est)),max(Parameter.getXAxis(generator,sampler,est))],...
				'ylab','NMSE','caption',Parameter.getTitle(graphGenerator,generator,sampler,est),'styles',{'-v','-^','-x','--v','--^','--x'});
			%F(2) = F_figure('Y',graph.getFourierTransform(graphFunction)','tit','Fourier transform of target signal','xlab','Freq. index','ylab','Function value');
			F.translation_table = {'kernel superposition','KS';'RKHS superposition','RS';'Bandlimited','BL';'Ass. B = cut-off freq.','cut-off'; ...
				'BL, Ass.','BL for'};
		end
		
		
		% MC simulation to compare multiple kernel dictionaries
		%      NMSE vs S
		% - generated signal is Exponentially decaying [anis2016proxies]
		function F = compute_fig_3502(obj,niter)
						
			% 1. Graph and signal 
			N = 100; % number of vertices			
			B = 20; % bandwidth of the true function
			
			graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.25 ,'s_numberOfVertices',N);
			graph = graphGenerator.realization;					
			generator = ExponentiallyDecayingGraphFunctionGenerator('graph',graph,'s_bandwidth',B,'s_decayingRate',4);
			%graphFunction = bandlimitedFunctionGenerator.realization();
			%generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',graphFunction);			
	
			%graph.plotFourierTransform(generator.realization());
			
			
			% 2. Sampler
			SNR = 10; % dB
			S_vec = 10:5:100;
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);			
			sampler = sampler.replicate('',[],'s_numberOfSamples',num2cell(S_vec));		
						
			% 3. Estimators
			sigmaArray_diffusion = linspace(0.1,.5,5);
			sigmaArray_regularized = linspace(0.1,.5,5);
						
			% 4. MKL function estimators	
			% 4. generate Kernel matrix
			
			h_r_inv_diffusion = LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray_diffusion);
			h_r_inv_regularized = LaplacianKernel.regularizedKernelFunctionHandle(sigmaArray_regularized);
			dictionary_diffusion = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',h_r_inv_diffusion);			
			dictionary_regularized = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',h_r_inv_regularized);			
			dictionary_both = LaplacianKernel('m_laplacian',graph.getLaplacian(),'h_r_inv',[h_r_inv_diffusion, h_r_inv_regularized]);			
			
			%dictionary_diffusion.plotEigenvalueFunctions;
			plot_dictionary_r_functions = 0;
			if plot_dictionary_r_functions
				subplot(3,1,1)
				dictionary_diffusion.plotEigenvalueFunctions;
				subplot(3,1,2)
				dictionary_regularized.plotEigenvalueFunctions;
				subplot(3,1,3)
				dictionary_both.plotEigenvalueFunctions;
			end
			
			
			%%
			% *BOLD TEXT* );
			%mkl_estimator(1,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',5e-3,'ch_type','RKHS superposition');
			%mkl_estimator(2,1) = MkrGraphFunctionEstimator('m_kernel',kG.getKernelMatrix(),'s_regularizationParameter',1e-2,'ch_type','RKHS superposition');
			rs_estimators = MkrGraphFunctionEstimator('s_regularizationParameter',.05,'ch_type','RKHS superposition');
			rs_estimators = rs_estimators.replicate('m_kernel',{dictionary_diffusion.getKernelMatrix(),dictionary_regularized.getKernelMatrix(),dictionary_both.getKernelMatrix()},'',{});
			
			ks_estimators = MkrGraphFunctionEstimator('s_regularizationParameter',.05,'ch_type','kernel superposition');
			ks_estimators = ks_estimators.replicate('m_kernel',{dictionary_diffusion.getKernelMatrix(),dictionary_regularized.getKernelMatrix(),dictionary_both.getKernelMatrix()},'',{});
			
            est = [rs_estimators;ks_estimators];
			
			for k = 1:6
				est(k,1).c_replicatedVerticallyAlong = {'ch_type'};			
			end			
			leg = Parameter.getLegend(generator,sampler,est);
			for k = 1:6
				leg{k} = [leg{k}, sprintf(', D%d',mod(k-1,3)+1) ];
			end
			

			% Simulation
			mse =  Simulate(generator,sampler,est,niter);
			
			% Representation			
			F = F_figure('X',Parameter.getXAxis(generator,sampler,est),...
                'Y',mse,'leg',leg,'leg_pos','northwest',...
                'xlab','Number of observed vertices (S)','ylimit',[0 1.5],'xlimit',[min(Parameter.getXAxis(generator,sampler,est)),max(Parameter.getXAxis(generator,sampler,est))],...
				'ylab','NMSE','caption',Parameter.getTitle(graphGenerator,generator,sampler,est),'styles',{'-v','-^','-x','--v','--^','--x'});
			%F(2) = F_figure('Y',graph.getFourierTransform(graphFunction)','tit','Fourier transform of target signal','xlab','Freq. index','ylab','Function value');
			
		end
		
		% print version of 3502
		function F = compute_fig_3503(obj,niter)
			F = obj.load_F_structure(3502);
			F.translation_table = {'kernel superposition','KS';'RKHS superposition','RS';'Bandlimited','BL';'Ass. B = cut-off freq.','cut-off'; ...
				'BL, Ass.','BL for'};
			F.leg_pos = 'northeast';
			F.ylimit = [0 1];
		end
		
		
		
		% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% %%  4. simulations with MKL on real data for recommender systems
		% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		
		% 1) Simulations from Narang et al, LOCALIZED ITERATIVE METHODS FOR 
		% INTERPOLATION IN GRAPH STRUCTURED DATA, 2013
		
		% 
		function F = compute_fig_4101(obj,niter)
						
			narang_estimator = NarangGraphFunctionEstimator('s_regularizationParameter',1e-2,'ch_type','LSR');			
	
			s_case = 0;
			switch s_case
				case 0
					v_sigma2 = .7;
					%kG = LaplacianKernel('h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(v_sigma2));
					kG = DiffusionGraphKernel('s_sigma',v_sigma2);
					h_kernelMat = @(graph) kG.getNewKernelMatrix(graph);
					mkl_estimator = RidgeRegressionGraphFunctionEstimator('s_regularizationParameter',1e-1,'h_kernelMat',h_kernelMat);
					mkl_estimator = mkl_estimator.replicate('s_regularizationParameter',num2cell([1e10 10.^(-10:2:6)]),'',[]);
					%mkl_estimator = mkl_estimator.replicate('s_regularizationParameter',num2cell(10.^(10)),'',[]);
				
				case 1
					v_sigma2 = sqrt(.2:.2:1);
					kG = LaplacianKernel('h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(v_sigma2));
					h_kernelMat = @(graph) kG.getNewKernelMatrix(graph);
					mkl_estimator = MkrGraphFunctionEstimator('s_regularizationParameter',1e-1,'ch_type','kernel superposition','h_kernelMat',h_kernelMat);
					mkl_estimator = mkl_estimator.replicate('s_regularizationParameter',num2cell([1e-3 1e-2 1e-1 1 10]),'',[]);
				case 2
					%c_regPar = {sqrt(linspace(.2,1.2,4)),sqrt(linspace(.2,1.2,8)),sqrt(linspace(.2,1.2,12)),sqrt(.2:.2:1)};
					c_regPar = {sqrt(linspace(.2,1.2,12))};
					for k = 1:length(c_regPar)
						kG = LaplacianKernel('h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(c_regPar{k}));
						h_kernelMat = @(graph) kG.getNewKernelMatrix(graph);
						mkl_estimator(k,1) = MkrGraphFunctionEstimator('s_regularizationParameter',1e-1,'ch_type','kernel superposition','h_kernelMat',h_kernelMat);
					end
				case 3					
					v_sigma2 = sqrt(linspace(.2,1.2,12));
					kG = LaplacianKernel('h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(v_sigma2));
					h_kernelMat = @(graph) kG.getNewKernelMatrix(graph);
					
					mkl_estimator(1,1) = MkrGraphFunctionEstimator('s_regularizationParameter',1e-1,'h_kernelMat',h_kernelMat,'ch_type','RKHS superposition');
					mkl_estimator(2,1) = MkrGraphFunctionEstimator('s_regularizationParameter',1e-1,'h_kernelMat',h_kernelMat,'ch_type','kernel superposition');
				case 4
					v_sigma2 = sqrt(linspace(.2,1.2,12));
					kG = LaplacianKernel('h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(v_sigma2));
					h_kernelMat = @(graph) kG.getNewKernelMatrix(graph);
					mkl_estimator = MkrGraphFunctionEstimator('s_regularizationParameter',1e-1,'ch_type','RKHS superposition','h_kernelMat',h_kernelMat);
					mkl_estimator = mkl_estimator.replicate('s_regularizationParameter',num2cell([1e-3 1e-2 1e-1 1 10]),'',[]);
			end
            
			
			%ridge_estimator = RidgeRegressionGraphFunctionEstimator('s_regularizationParameter',v_regPar,'h_kernelMat',h_kernelMat);
			
			%estimator = [narang_estimator;ridge_estimator];
			estimator = [narang_estimator;mkl_estimator];
			%estimator = narang_estimator;
			
			[v_CVSets,v_range] = ReadMovieLensDataset.getCVSets();
			graphConstructor = @(table) Graph.constructGraphFromTable(table,'cosine');
			
			s_case
			mse = RecommenderSystemsSimulator.simulateDataset( v_CVSets , graphConstructor, estimator )
			
			rmse = sqrt( mse ) / (v_range(2)-v_range(1))
			%save('1.mat')
			F = [];
		end
	
		
	end
	
	% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% %%  5. airports
	% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods
		
		% 0) data analysis  ===============================================
		
		% Check graph construction code
		function F = compute_fig_5000(obj,niter)
			
			%0. define parameters
% 			s_sigma=1.3;
% 			s_numberOfClusters=5;
% 			s_lambda=10^-5;
% 			s_monteCarloSimulations=niter;%100;
% 			s_bandwidth1=10;
% 			s_bandwidth2=20;
% 			s_SNR=1000;
% 			
% 			
% 			s_epsilon=0.2;
% 			s_functionTypes=5;
% 			v_sampleSetSize=(0.1:0.1:1);
% 		
			
			s_niter=10;
			s_beta=0.02;
			s_alpha=0.005;
			
			
			% define graph
			[Ho,Mo,Alto,Hn,Mn,Altn] = readTemperatureDataset;
			%tic
			%m_constraintLaplacian=zeros(size(Ho,1));
			%m_constraintLaplacian(4:15,1)=1;
			m_constraintLaplacian = triu(rand(size(Ho,1))>.8);
			m_constraintLaplacian = m_constraintLaplacian + m_constraintLaplacian';
			graphGenerator = GraphLearningSmoothSignalGraphGenerator('m_observed',Ho,'s_niter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta,'m_constraintLaplacian',m_constraintLaplacian,'s_dont_estimate_the_signal',0);
			%toc
			G = graphGenerator.realization()
			G.plotFourierTransform(Ho)
			
			F = [];
			return
			
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
		
		% visualization: scatter plot all airports
		function F = compute_fig_5001(obj,niter)
			folder = 'libGF/datasets/AirportsDataset';
			load([folder '/delaytables.mat'],'arrDelayMatrix','depDelayMatrix');
			load([folder '/adjacency.mat'],'adjacency');
			
			A = sum(adjacency,3);
			s_airportNum = size(arrDelayMatrix,1);
			for row = 1:s_airportNum
				for col = 1:s_airportNum
					if A(row,col)>0
						plot(arrDelayMatrix(row,:),arrDelayMatrix(col,:),'.')
						hold all
						plot(depDelayMatrix(row,:),depDelayMatrix(col,:),'.')
						hold off
						xlim([-20 20])
						ylim([-20 20])
						pause						
					end
				end				
			end
			
			
			F=[];
		end
		
		% visualization: scatter plot for 50 busiest airports
		function F = compute_fig_5002(obj,niter)
			folder = 'libGF/datasets/AirportsDataset';
			load([folder '/delaytables.mat'],'arrDelayMatrix','depDelayMatrix');
			load([folder '/adjacency.mat'],'adjacency');
			
			A = sum(adjacency,3);
			
			% select busiest airports
			s_selectedAirportNum = 50;
			[~,v_inds] = sort(sum(A,2),'descend');
			v_inds = v_inds(1:s_selectedAirportNum); % indices of the most crowded airports
			
			arrDelayMatrix = arrDelayMatrix(v_inds,:);
			depDelayMatrix = depDelayMatrix(v_inds,:);
			adjacency = adjacency(v_inds,v_inds,:);
			
			
			s_airportNum = size(arrDelayMatrix,1);
			for row = 1:s_airportNum
				for col = 1:s_airportNum
					if A(row,col)>0
						plot(arrDelayMatrix(row,:),arrDelayMatrix(col,:),'.')
						hold all
						plot(depDelayMatrix(row,:),depDelayMatrix(col,:),'.')
						hold off
						legend('arrival delay','departure delay');
						grid on
						xlim([-20 20])
						ylim([-20 20])
						pause						
					end
				end				
			end
			
			
			F=[];
		end
		
        % Tests with covariance matrix
		%   1. Covariance eigenvalues (principal components)
		%   2. Energy of the data along the principal directions
		%   2. Energy of the data along the principal directions of the
		%      Laplacian
		% - Laplacian matrix is not constrained to have zeros at certain
		% positions
        function F = compute_fig_5003(obj,niter)
            % define parameters
			s_nodeNum = 20;
            s_departureDelay = 0;

			% load data
            [ m_training_delay, m_test_delay, m_training_adj , m_test_adj ] = MultikernelSimulations.getTwoMonths(s_nodeNum,s_departureDelay);             
			m_data = [m_training_delay m_test_delay];
            
			% mean subtraction
			m_data = m_data - mean(m_data,2)*ones(1,size(m_data,2));
			
			% std normalization
			v_std = std(m_data')';
			m_data = diag(1./v_std)*m_data;
			
			% covariance
			C = cov(m_data');
			
			% principal components
			[m_pDirections, v_pComponents] = eig(C) ; % principal components
			v_pComponents = diag(v_pComponents);
			
			multiplot_array(1,1) = F_figure('Y',flipud(v_pComponents)','tit','Principal components of C');
			multiplot_array(2,1) = F_figure('Y',cumsum(flipud(v_pComponents))'/sum(v_pComponents),'tit','Cumulative sum of principal components of C','ylab','fraction of variance');
			F(1) = F_figure('multiplot_array',multiplot_array);
            
			% energy of the data along the principal directions
			m_projections = m_pDirections'*m_data;
			[H,v_centers] = hist(m_projections',40);
			for k = 1:length(v_pComponents)
				leg{k} = sprintf('n = %d (\\lambda_n = %g)',k,v_pComponents(k));
			end			
			F(2) = F_figure('Y',H','X',v_centers','leg',leg,'tit','Histogram of energy distribution along PCs');
            
			% approximation of inverse covariance via Laplacian			
			m_covInv = inv(C);
			m_laplacian = MultikernelSimulations.approximateWithLaplacian(m_covInv);

			% principal directions of Laplacian
			[m_laplacianPDirections, v_laplacianPComponents] = eig(m_laplacian) ; 
			v_laplacianPComponents = diag(v_laplacianPComponents);
			
			% energy of the data along the principal directions of the
			% Laplacian
			m_projections = m_laplacianPDirections'*m_data;
			[H,v_centers] = hist(m_projections',20);
			for k = 1:length(v_laplacianPComponents)
				leg{k} = sprintf('n = %d (\\lambda_n = %g)',k,v_laplacianPComponents(k));
				if k<6
					styles{k} = '-';					
				else
					styles{k} = '--';
				end
			end			
			F(3) = F_figure('Y',H','X',v_centers','leg',leg,'tit','Histogram of energy distribution along Laplacian PCs','styles',styles);
			
		end
        
		% Tests with covariance matrix (different from 5003, the topology
		% is used)
		%   1. Covariance eigenvalues (principal components)
		%   2. Energy of the data along the principal directions
		%   2. Energy of the data along the principal directions of the
		%      Laplacian
		% - Laplacian matrix IS constrained to have zeros for links with no
		%   flights
        function F = compute_fig_5004(obj,niter)
            % define parameters
			s_nodeNum = 25;
            s_departureDelay = 1;

			% load data
            [ m_training_delay, m_test_delay, m_training_adj , m_test_adj ] = MultikernelSimulations.getTwoMonths(s_nodeNum,s_departureDelay);             
			m_data = [m_training_delay m_test_delay];
            m_adjacency = sum(m_training_adj,3)+sum(m_test_adj,3);
m_adjacency = m_adjacency > 10;
			sparsity = sum(m_adjacency(:))/(numel(m_adjacency)-size(m_adjacency,1))

			% mean subtraction
			m_data = m_data - mean(m_data,2)*ones(1,size(m_data,2));
			
			% std normalization
			v_std = std(m_data')';
			m_data = diag(1./v_std)*m_data;
			
			% covariance
			C = cov(m_data');
			
			% principal components
			[m_covariancePDirections, v_covariancePComponents] = eig(C) ; % principal components
			v_covariancePComponents = diag(v_covariancePComponents);
			
			multiplot_array(1,1) = F_figure('Y',flipud(v_covariancePComponents)','tit','Principal components of C');
			multiplot_array(2,1) = F_figure('Y',cumsum(flipud(v_covariancePComponents))'/sum(v_covariancePComponents),'tit','Cumulative sum of principal components of C','ylab','fraction of variance');
			F(1) = F_figure('multiplot_array',multiplot_array);
            
			% energy of the data along the principal directions
			m_unconstrainedLaplacianProjections = m_covariancePDirections'*m_data;
			[m_hist,v_centers] = hist(m_unconstrainedLaplacianProjections',40);
			for k = 1:length(v_covariancePComponents)
				leg{k} = sprintf('n = %d (\\lambda_n = %g)',k,v_covariancePComponents(k));
			end			
			F(2) = F_figure('Y',m_hist','X',v_centers','leg',leg,'tit','Histogram of energy distribution along PCs');
            
			% approximation of inverse covariance via unconstrained Laplacian			
			m_covInv = inv(C);
			m_unconstrainedLaplacian = MultikernelSimulations.approximateWithLaplacian(m_covInv);

			% principal directions of unconstrained Laplacian
			[m_unconstrainedLaplacianPDirections, v_unconstrainedLaplacianPComponents] = eig(m_unconstrainedLaplacian) ; 
			v_unconstrainedLaplacianPComponents = diag(v_unconstrainedLaplacianPComponents);
			

			% approximation of inverse covariance via constrained Laplacian			
			m_covInv = inv(C);
			m_constrainedLaplacian = MultikernelSimulations.approximateWithLaplacian(m_covInv,m_adjacency);

			% principal directions of constrained Laplacian
			[m_constrainedLaplacianPDirections, v_constrainedLaplacianPComponents] = eig(m_constrainedLaplacian) ; 
			v_constrainedLaplacianPComponents = diag(v_constrainedLaplacianPComponents);
			
			
			% energy of the data along the principal directions of the
			% Laplacian
			s_componentsNum = 5;
			s_binNum = 15;
			
			m_covarianceProjections = m_covariancePDirections'*m_data;
			m_covarianceProjections = m_covarianceProjections(end:-1:end-s_componentsNum+1,:);			
			[m_histCovariance,v_covarianceCenters] = hist(m_covarianceProjections',s_binNum);
						
			m_unconstrainedLaplacianProjections = m_unconstrainedLaplacianPDirections'*m_data;
			m_unconstrainedLaplacianProjections = m_unconstrainedLaplacianProjections(1:s_componentsNum,:);			
			[m_histUnconstrained,v_unconstrainedCenters] = hist(m_unconstrainedLaplacianProjections',s_binNum);
			
			m_constrainedLaplacianProjections = m_constrainedLaplacianPDirections'*m_data;
			m_constrainedLaplacianProjections = m_constrainedLaplacianProjections(1:s_componentsNum,:);			
			[m_histConstrained,v_constrainedCenters] = hist(m_constrainedLaplacianProjections',s_binNum);
			
			
			for k = 1:s_componentsNum
				leg{k} = sprintf('Cov. n = %d (\\lambda_n = %g)',k,v_covariancePComponents(end-k+1));
				leg{k+s_componentsNum} = sprintf('Unc. n = %d (\\lambda_n = %g)',k,v_unconstrainedLaplacianPComponents(k));
				leg{k+2*s_componentsNum} = sprintf('Cons. n = %d (\\lambda_n = %g)',k,v_constrainedLaplacianPComponents(k));
				styles{k} = '-';
				styles{k+s_componentsNum} = '--';
				styles{k+2*s_componentsNum} = '--x';
			end			
			Y = [m_histCovariance';m_histUnconstrained';m_histConstrained'];
			X = [v_covarianceCenters';v_unconstrainedCenters';v_constrainedCenters'];
			X = kron(X,ones(s_componentsNum,1));
			
			F(3) = F_figure('Y',Y,'X',X,'leg',leg,'tit','Histogram of energy distribution along PCs','colorp',s_componentsNum,'styles',styles);
			
		end
        
		
		% Tests with covariance matrix (different from 5004, it estimates
		% the covariance with a sparsity pattern given by the underlying
		% graph		
		%   1. Covariance eigenvalues (principal components)
		%   2. Energy of the data along the principal directions
		%   2. Energy of the data along the principal directions of the
		%      Laplacian
		% - Laplacian matrix IS constrained to have zeros for links with no
		%   flights
        function F = compute_fig_5005(obj,niter)
            % define parameters
			s_nodeNum = 25;
            s_departureDelay = 1;

			% load data
            [ m_training_delay, m_test_delay, m_training_adj , m_test_adj ] = MultikernelSimulations.getTwoMonths(s_nodeNum,s_departureDelay);             
			m_data = [m_training_delay m_test_delay];
            m_adjacency = sum(m_training_adj,3)+sum(m_test_adj,3);
m_adjacency = m_adjacency > 100;
			sparsity = sum(m_adjacency(:))/(numel(m_adjacency)-size(m_adjacency,1))

			% mean subtraction
			m_data = m_data - mean(m_data,2)*ones(1,size(m_data,2));
			
			% std normalization
			v_std = std(m_data')';
			m_data = diag(1./v_std)*m_data;
			
			% covariance
			sample_C = cov(m_data');
			m_covInv = MultikernelSimulations.learnInverseCov( sample_C , m_adjacency );
			C = inv(m_covInv);
			
			
			% principal components
			[m_covariancePDirections, v_covariancePComponents] = eig(C) ; % principal components
			v_covariancePComponents = diag(v_covariancePComponents);
			
			multiplot_array(1,1) = F_figure('Y',flipud(v_covariancePComponents)','tit','Principal components of C');
			multiplot_array(2,1) = F_figure('Y',cumsum(flipud(v_covariancePComponents))'/sum(v_covariancePComponents),'tit','Cumulative sum of principal components of C','ylab','fraction of variance');
			F(1) = F_figure('multiplot_array',multiplot_array);
            
			% energy of the data along the principal directions
			m_unconstrainedLaplacianProjections = m_covariancePDirections'*m_data;
			[m_hist,v_centers] = hist(m_unconstrainedLaplacianProjections',40);
			for k = 1:length(v_covariancePComponents)
				leg{k} = sprintf('n = %d (\\lambda_n = %g)',k,v_covariancePComponents(k));
			end			
			F(2) = F_figure('Y',m_hist','X',v_centers','leg',leg,'tit','Histogram of energy distribution along PCs');
            
			% approximation of inverse covariance via unconstrained Laplacian			
			m_covInv = inv(C);
			m_unconstrainedLaplacian = MultikernelSimulations.approximateWithLaplacian(m_covInv);

			% principal directions of unconstrained Laplacian
			[m_unconstrainedLaplacianPDirections, v_unconstrainedLaplacianPComponents] = eig(m_unconstrainedLaplacian) ; 
			v_unconstrainedLaplacianPComponents = diag(v_unconstrainedLaplacianPComponents);
			

			% approximation of inverse covariance via constrained Laplacian			
			m_covInv = inv(C);
			m_constrainedLaplacian = MultikernelSimulations.approximateWithLaplacian(m_covInv,m_adjacency);

			% principal directions of constrained Laplacian
			[m_constrainedLaplacianPDirections, v_constrainedLaplacianPComponents] = eig(m_constrainedLaplacian) ; 
			v_constrainedLaplacianPComponents = diag(v_constrainedLaplacianPComponents);
			
			
			% energy of the data along the principal directions of the
			% Laplacian
			s_componentsNum = 5;
			s_binNum = 15;
			
			m_covarianceProjections = m_covariancePDirections'*m_data;
			m_covarianceProjections = m_covarianceProjections(end:-1:end-s_componentsNum+1,:);			
			[m_histCovariance,v_covarianceCenters] = hist(m_covarianceProjections',s_binNum);
						
			m_unconstrainedLaplacianProjections = m_unconstrainedLaplacianPDirections'*m_data;
			m_unconstrainedLaplacianProjections = m_unconstrainedLaplacianProjections(1:s_componentsNum,:);			
			[m_histUnconstrained,v_unconstrainedCenters] = hist(m_unconstrainedLaplacianProjections',s_binNum);
			
			m_constrainedLaplacianProjections = m_constrainedLaplacianPDirections'*m_data;
			m_constrainedLaplacianProjections = m_constrainedLaplacianProjections(1:s_componentsNum,:);			
			[m_histConstrained,v_constrainedCenters] = hist(m_constrainedLaplacianProjections',s_binNum);
			
			
			for k = 1:s_componentsNum
				leg{k} = sprintf('Cov. n = %d (\\lambda_n = %g)',k,v_covariancePComponents(end-k+1));
				leg{k+s_componentsNum} = sprintf('Unc. n = %d (\\lambda_n = %g)',k,v_unconstrainedLaplacianPComponents(k));
				leg{k+2*s_componentsNum} = sprintf('Cons. n = %d (\\lambda_n = %g)',k,v_constrainedLaplacianPComponents(k));
				styles{k} = '-';
				styles{k+s_componentsNum} = '--';
				styles{k+2*s_componentsNum} = '--x';
			end			
			Y = [m_histCovariance';m_histUnconstrained';m_histConstrained'];
			X = [v_covarianceCenters';v_unconstrainedCenters';v_constrainedCenters'];
			X = kron(X,ones(s_componentsNum,1));
			
			F(3) = F_figure('Y',Y,'X',X,'leg',leg,'tit','Histogram of energy distribution along PCs','colorp',s_componentsNum,'styles',styles);
			
		end
        
		
		
		
		
		% 1) Dorina's graphs ==============================================
		
		% visualization: graph construction just by Dorina's method (no
		% constraints)
		function F = compute_fig_5101(obj,niter)
			
			% Graph construction parameters
			s_niter=10;
			s_beta=1e-3;  % decrease to increase sparsity in the graph
			s_alpha=1e-6;%0.005; % decrease to increase bandlimitedness
			
			
			% load dataset
			folder = 'libGF/datasets/AirportsDataset';
			load([folder '/delaytables.mat'],'arrDelayMatrix','depDelayMatrix');
			load([folder '/adjacency.mat'],'adjacency');
			
			A = sum(adjacency,3);
			
			% select busiest airports
			s_selectedAirportNum = 50;
			[~,v_inds] = sort(sum(A,2),'descend');
			v_inds = v_inds(1:s_selectedAirportNum); % indices of the most crowded airports			
			arrDelayMatrix = arrDelayMatrix(v_inds,:);
			depDelayMatrix = depDelayMatrix(v_inds,:);
			adjacency = adjacency(v_inds,v_inds,:);
					
			Ho = depDelayMatrix;
			m_constraintLaplacian = zeros(size(Ho,1));			
			graphGenerator = GraphLearningSmoothSignalGraphGenerator('m_observed',Ho,'s_niter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta,'m_constraintLaplacian',m_constraintLaplacian,'s_dont_estimate_the_signal',0);
			%toc
			G = graphGenerator.realization()
			figure(1)
			G.plotFourierTransform(Ho)
			figure(2)
			hist(G.m_adjacency,100)
			F = [];
		end
		
		% visualization: graph construction via a constrained versin of 
		% Dorina's method
		function F = compute_fig_5102(obj,niter)
			
			% Graph construction parameters
			s_niter=10;
			s_beta=.5; %1e-2  % decrease to increase sparsity in the graph
			s_alpha=1e-4;%1e-5; %0.005; % decrease to increase bandlimitedness
			
			
			% load dataset
			folder = 'libGF/datasets/AirportsDataset';
			load([folder '/delaytables.mat'],'arrDelayMatrix','depDelayMatrix');
			load([folder '/adjacency.mat'],'adjacency');
			
			A = sum(adjacency,3);
			
			% select busiest airports
			s_selectedAirportNum = 50;
			[~,v_inds] = sort(sum(A,2),'descend');
			v_inds = v_inds(1:s_selectedAirportNum); % indices of the most crowded airports			
			arrDelayMatrix = arrDelayMatrix(v_inds,:);
			depDelayMatrix = depDelayMatrix(v_inds,:);
			adjacency = adjacency(v_inds,v_inds,:);
					
			Ho = depDelayMatrix;
			m_constraintLaplacian = (sum(adjacency,3)==0);
			m_constraintLaplacian = m_constraintLaplacian - diag(diag(m_constraintLaplacian));
			graphGenerator = GraphLearningSmoothSignalGraphGenerator('m_observed',Ho,'s_niter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta,'m_constraintLaplacian',m_constraintLaplacian,'s_dont_estimate_the_signal',0);
			%toc
			G = graphGenerator.realization()
			figure(1)
			G.plotFourierTransform(Ho)
			figure(2)
			hist(G.m_adjacency(:),100)
			F = [];
		end
		
			
		% visualization: graph construction via a constrained version of 
		% Dorina's method
		% Different from 5004, we now learn the graph using the first 15
		% days and plot the signal Fourier transform in the second 15 days
		function F = compute_fig_5103(obj,niter)
			
			% Graph construction parameters
			s_niter=10;
			s_beta=1e-3; %1e-2  % decrease to increase sparsity in the graph
			s_alpha=1e-4;%1e-5; %0.005; % decrease to increase bandlimitedness
			
			
			% load dataset
			folder = 'libGF/datasets/AirportsDataset';
			load([folder '/delaytables.mat'],'arrDelayMatrix','depDelayMatrix');
			load([folder '/adjacency.mat'],'adjacency');
			s_dayNum = size(arrDelayMatrix,2);
			
			A = sum(adjacency,3);
			
			% select busiest airports
			s_selectedAirportNum = 50;
			[~,v_inds] = sort(sum(A,2),'descend');
			v_inds = v_inds(1:s_selectedAirportNum); % indices of the most crowded airports			
			arrDelayMatrix = arrDelayMatrix(v_inds,:);
			depDelayMatrix = depDelayMatrix(v_inds,:);
			adjacency = adjacency(v_inds,v_inds,:);
					
			Ho = depDelayMatrix;
			Ho_first_half = Ho(:,1:floor(size(Ho,2)/2));
			Ho_second_half = Ho(:,floor(size(Ho,2)/2)+1:end);
			m_constraintLaplacian = (sum(adjacency,3)==0);
			m_constraintLaplacian = m_constraintLaplacian - diag(diag(m_constraintLaplacian));
			graphGenerator = GraphLearningSmoothSignalGraphGenerator('m_observed',Ho_first_half,'s_niter',s_niter,'s_alpha',s_alpha,'s_beta',s_beta*size(Ho_first_half,2),'m_constraintLaplacian',m_constraintLaplacian,'s_dont_estimate_the_signal',0);
			%toc
			G = graphGenerator.realization();
			figure(1)
			G.plotFourierTransform(Ho_second_half)
			[counts,bins]= hist(G.m_adjacency(:),100);
			F = F_figure('X',bins,'Y',counts/sum(counts),'plot_type_2D','bar');
        end
		
        
        
        % Figure to test estimation using Dorina's graphs
        function F = compute_fig_5104(obj,niter)
            % define parameters
			s_nodeNum = 50;
            s_departureDelay = 1;
                       
            max_iter = 1000;          % max iteration for learning laplacian
			alpha = 1/100;
			beta = 500/100;	%150          % alpha, beta paramters for learning laplacian
			S_vec = 10:2:30;		  % for creating uniform sampler
			B_vec = [5 10 20 -1];    % for creating BL estimator
			mu = 1e-1;                % regularization parameter for MK estimator
			SNR = Inf;
            s_verbose = 1;
                       
            
            [ m_training_delay, m_test_delay, m_training_adj , m_test_adj ] = MultikernelSimulations.getTwoMonths(s_nodeNum,s_departureDelay);
             
            %
            
			% read temperature dataset and create the graph
			% However, if the graph is already exists, then skip the process
			
			
            %[m_training_delay,Mo,Alto,m_test_delay,Mn,Altn] = readTemperatureDataset();
            filename = 'airportLaplacian.mat';
			if (exist(filename, 'file') == 2)
				 load(filename,'g_graph')				
				%graph = Graph('m_adjacency',Graph.createAdjacencyFromLaplacian(L));
			else
				% learn laplacian
				% use old tempearature to learn graph laplacian
				gl = GraphLearningSmoothSignalGraphGenerator('m_observed', m_training_delay, 's_niter', max_iter, 's_alpha', alpha, 's_beta', beta);
				%gl = SmoothSignalGraphGenerator('m_observed', Ho, 's_niter', max_iter, 's_alpha', alpha, 's_beta', beta);
				g_graph = gl.realization();
                save(filename,'g_graph');
			end
			m_laplacian = g_graph.getLaplacian(); 
            
            graph_sparsity = 2*sum(sum( g_graph.m_adjacency ~= 0 ))/size(g_graph.m_adjacency,1)^2
            
			if s_verbose
                figure(1)
                hist(g_graph.m_adjacency(:),50);

                figure(2)
                g_graph.plotFourierTransform(m_training_delay);
                subplot(2,1,1);title('training data')
                                
                figure(3)
                g_graph.plotFourierTransform(m_test_delay);
                subplot(2,1,1);title('test data')
                
                s_componentsNum = size(m_laplacian,1)-rank(m_laplacian)
                pause()
            end
			%
			% define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);
			sampler = sampler.replicate([],{},'s_numberOfSamples',num2cell(S_vec));		
			%		
			% BL graph function estimator
			%bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian',g_graph.getLaplacian);			
			%bl_estimator.c_replicatedVerticallyAlong = {'ch_name'};
			%bl_estimator = bl_estimator.replicate('s_bandwidth',num2cell(B_vec),'',{});
			
			s_case = 1;
            switch s_case
                case 1
                    % MKL function estimators
                    c_sigma = {sqrt(linspace(200, 400, 30)),sqrt(linspace(.01, 1, 30)),sqrt(linspace(50, 100, 10)),sqrt(linspace(100, 200, 10))};
                    for k = 1:length(c_sigma)
                        kG = LaplacianKernel('m_laplacian',m_laplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(c_sigma{k}));
                        m_kernel1 = kG.getKernelMatrix();
                        mkl_estimator(k,1) = MkrGraphFunctionEstimator('s_regularizationParameter',mu, 'm_kernel', m_kernel1,'ch_type', 'kernel superposition');   % first 1
                        leg{k} = sprintf('k = %d',k);
                    end
                case 2
                    v_mu = 10.^(-8:0);
                    c_sigma = sqrt(linspace(100, 200, 10));
                    kG = LaplacianKernel('m_laplacian',m_laplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(c_sigma));
                    m_kernel1 = kG.getKernelMatrix();
                    for k = 1:length(v_mu)                        
                        mkl_estimator(k,1) = MkrGraphFunctionEstimator('s_regularizationParameter',v_mu(k), 'm_kernel', m_kernel1,'ch_type', 'kernel superposition');   % first 1
                        leg{k} = sprintf('k = %d',k);
                    end
                    
            end
            
            
            
            
            
%             sigma2_vec = sqrt([1 2 5]);
% 			kG = LaplacianKernel('m_laplacian',m_laplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigma2_vec));
% 			m_kernel2 = kG.getKernelMatrix();
% 			
% 			
% 			%mkl_estimator_RKHS = mkl_estimator_RKHS.replicate('m_kernel', m_kernel, [], {} );
% 			
% 			%mkl_estimator_kernel = MkrGraphFunctionEstimator('s_regularizationParameter',mu,'ch_type','kernel superposition');
% 			%mkl_estimator_kernel = mkl_estimator_kernel.replicate('m_kernel', m_kernel, [], {} );
% 			
% 			%mkl_estimator = [];
% 			%
% 			%for i = 1:length(mkl_estimator_RKHS)
%             mkl_estimator_RKHS.c_replicatedVerticallyAlong = {'ch_name','legendString'};
%             mkl_estimator_replicated = mkl_estimator_RKHS.replicate('ch_type',{'RKHS superposition','kernel superposition'},'',[]);
%             
%             mkl_estimator_RR(1) = MkrGraphFunctionEstimator('s_regularizationParameter',mu, 'm_kernel', m_kernel2(:,:,1), 'ch_type', 'kernel superposition', 's_sigma', sigma2_vec(1));
%             mkl_estimator_RR(2) = MkrGraphFunctionEstimator('s_regularizationParameter',mu, 'm_kernel', m_kernel2(:,:,2), 'ch_type', 'kernel superposition', 's_sigma', sigma2_vec(2));
%             mkl_estimator_RR(3) = MkrGraphFunctionEstimator('s_regularizationParameter',mu, 'm_kernel', m_kernel2(:,:,3), 'ch_type', 'kernel superposition', 's_sigma', sigma2_vec(3));
%             %mkl_estimator = [mkl_estimator; mkl_estimator_replicated];
			%end	
			%est = [mkl_estimator_replicated;mkl_estimator_RR';bl_estimator];
			est = mkl_estimator;
 m_test_delay = m_training_delay;           
			%
			% Simulation
			nmse = zeros(length(est), length(sampler));
			for i = 1:size(m_test_delay,2)
				generator = FixedGraphFunctionGenerator('graph',g_graph, 'graphFunction', m_test_delay(:,i));
				%generator = FixedGraphFunctionGenerator('graph',graph, 'graphFunction', Mn);
				nmse = nmse + Simulate(generator, sampler, est, niter, true);
				%res = Simulator.simStatistic(niter,generator,sampler,est);
				%mse = Simulator.computeNmse(res,Results('stat',graphFunction));
			end
			nmse = nmse / size(m_test_delay,2);

			% Representation	
           % leg = Parameter.getLegend(generator,sampler, est);
			F = F_figure('X',S_vec,...
                'Y',nmse,'leg',leg,...
                'xlab', 'sample size','ylimit',...
				[0 1.1],'ylab','NMSE',...
				'tit',sprintf('Airports dataset mu=%g',mu));
            
            
            
		end
        
		
		
		% 2) Graphs from inverse covariance ===============================
		
		% Estimation test
        function F = compute_fig_5200(obj,niter)
            % define parameters
			s_nodeNum = 15;
            s_departureDelay = 1;
			
			% estimation parameters
			s_mu = 1e-2;
			s_numberOfSamples = 14;
			
			%%%%%%%%%%%%%%%%%%

			% load data
            [ m_training_data, m_test_data, m_training_adj , m_test_adj ] = MultikernelSimulations.getTwoMonths(s_nodeNum,s_departureDelay);             
	
debug = 1;
if debug
	m_training_data = [m_training_data m_test_data];
end
            m_adjacency = sum(m_training_adj,3);
			m_adjacency = m_adjacency > 100;
			
			sparsity = sum(m_adjacency(:))/(numel(m_adjacency)-size(m_adjacency,1))

			
			% data normalization
			v_mean = mean(m_training_data,2);
			v_std = std(m_training_data')';
			m_normalized_training_data = diag(1./v_std)*(m_training_data - v_mean*ones(1,size(m_training_data,2)));
			m_normalized_test_data = diag(1./v_std)*(m_test_data - v_mean*ones(1,size(m_test_data,2)));
			
			% covariance of normalized data
			m_covInv = MultikernelSimulations.learnInverseCov( cov(m_normalized_training_data') , m_adjacency );

if debug
	m_cov = cov(m_normalized_training_data');
end
			
			% generator and sampler
			generator = RandomlyPickGraphFunctionGenerator('m_graphFunction',m_normalized_test_data);
			sampler = UniformGraphFunctionSampler('s_SNR',Inf,'s_numberOfSamples',s_numberOfSamples);
			cov_estimator = RidgeRegressionGraphFunctionEstimator('s_regularizationParameter',s_mu,'m_kernel',inv(m_covInv));
			
			% simulation
			estimator = cov_estimator;
			nmse = NaN(size(estimator,1),niter);			
			for s_itInd = 1:niter				
				% generate signal
				v_signal = generator.realization();
				
				% sample signal
				[v_samples,v_sampleLocations] = sampler.sample(v_signal);
				
				% estimate signal
				v_signalEst = cov_estimator.estimate(v_samples,v_sampleLocations);				
if debug
	% LMMSE estimator
	v_signalEst = m_cov(:,v_sampleLocations)*(  (m_cov(v_sampleLocations,v_sampleLocations) + 1*eye(length(v_sampleLocations)))\v_samples );	
end
				% measure error
				v_unnormalized_signal = (v_std).*v_signal + v_mean;
				v_unnormalized_signalEst = (v_std).*v_signalEst + v_mean;
				
				v_test_indices = 1:length(v_signal);
				v_test_indices(v_sampleLocations) = 0; v_test_indices = v_test_indices(v_test_indices>0);
				nmse(1,s_itInd) = norm( v_unnormalized_signal(v_test_indices) - v_unnormalized_signalEst(v_test_indices) )^2 /  norm( v_unnormalized_signal(v_test_indices)  )^2;
				
				unmse(1,s_itInd) = norm( v_signal(v_test_indices) - v_signalEst(v_test_indices) )^2 /  norm( v_signal(v_test_indices)  )^2;
if debug
	[v_signalEst v_signal]
	[v_signalEst(v_test_indices) v_signal(v_test_indices)]
end
			end
			
			nmse = mean(nmse,2)
			% average error
if debug
	unmse = mean(unmse,2)
end
			
			
			F = [];
			
		end
        
		
		% Estimation test with more data
        function F = compute_fig_5201(obj,niter)
            % define parameters
			s_nodeNum = 30;
            s_departureDelay = 0;
			
			% estimation parameters
			s_mu = 1e-2;
			s_numberOfSamples = 29;
			
			%%%%%%%%%%%%%%%%%%

			% load data
			
			[ m_training_data, m_test_data, m_training_adj , m_test_adj ] = MultikernelSimulations.getSixMonths(s_nodeNum,s_departureDelay);
			
debug = 0;
if debug
	%m_training_data = [m_training_data m_test_data];
end
            m_adjacency = sum(m_training_adj,3);
			m_adjacency = m_adjacency > 100;
			
			sparsity = sum(m_adjacency(:))/(numel(m_adjacency)-size(m_adjacency,1))

			
			% data normalization
			v_mean = mean(m_training_data,2);
			v_std = std(m_training_data')';
			m_normalized_training_data = diag(1./v_std)*(m_training_data - v_mean*ones(1,size(m_training_data,2)));
			m_normalized_test_data = diag(1./v_std)*(m_test_data - v_mean*ones(1,size(m_test_data,2)));
			
			% covariance of normalized data
			m_covInv = MultikernelSimulations.learnInverseCov( cov(m_normalized_training_data') , m_adjacency );

if debug
	m_cov = cov(m_normalized_training_data');
end
			
			% generator and sampler
			generator = RandomlyPickGraphFunctionGenerator('m_graphFunction',m_normalized_test_data);
			sampler = UniformGraphFunctionSampler('s_SNR',Inf,'s_numberOfSamples',s_numberOfSamples);
			cov_estimator = RidgeRegressionGraphFunctionEstimator('s_regularizationParameter',s_mu,'m_kernel',inv(m_covInv));
			
			% simulation
			estimator = cov_estimator;
			normalized_signal_mse = NaN(size(estimator,1),niter);			
			for s_itInd = 1:niter				
				% generate signal
				v_signal = generator.realization();
				
				% sample signal
				[v_samples,v_sampleLocations] = sampler.sample(v_signal);
				
				% estimate signal
				v_signalEst = cov_estimator.estimate(v_samples,v_sampleLocations);				
if debug
	% LMMSE estimator
	v_signalEst = m_cov(:,v_sampleLocations)*(  (m_cov(v_sampleLocations,v_sampleLocations) + 1e-100*eye(length(v_sampleLocations)))\v_samples );	
end
				% revert normalization
				v_unnormalized_signal = (v_std).*v_signal + v_mean;
				v_unnormalized_signalEst = (v_std).*v_signalEst + v_mean;
				
                % measure error
				v_test_indices = 1:length(v_signal);
				v_test_indices(v_sampleLocations) = 0; v_test_indices = v_test_indices(v_test_indices>0);
				unnormalized_signal_mse(1,s_itInd) = norm( v_unnormalized_signal(v_test_indices) - v_unnormalized_signalEst(v_test_indices) )^2/(length(v_test_indices)) ;%/  norm( v_unnormalized_signal(v_test_indices)  )^2;
				unnormalized_signal_energy(1,s_itInd) =  norm( v_unnormalized_signal(v_test_indices)  )^2/(length(v_test_indices));
				
                normalized_signal_mse(1,s_itInd) = norm( v_signal(v_test_indices) - v_signalEst(v_test_indices) )^2/(length(v_test_indices)); %/  norm( v_signal(v_test_indices)  )^2;
                normalized_signal_energy(1,s_itInd) =  norm( v_signal(v_test_indices)  )^2/(length(v_test_indices));
                
if debug
%	[v_signalEst v_signal]
%	[v_signalEst(v_test_indices) v_signal(v_test_indices)]
end
			end
			
			normalized_signal_mse = mean(normalized_signal_mse,2) 
            normalized_signal_nmse = normalized_signal_mse / mean(normalized_signal_energy,2)
			% average error
            unnormalized_signal_mse = mean(unnormalized_signal_mse,2)
            unnormalized_signal_nmse = unnormalized_signal_mse / mean(unnormalized_signal_energy,2)

            rmse_in_minutes = sqrt(unnormalized_signal_mse)
			
			
			F = [];
			
        end
        
		
        % Estimation test with more estimators
        function F = compute_fig_5202(obj,niter)
            % define parameters
			s_nodeNum = 30;
            s_departureDelay = 0;
			
			% estimation parameters
			s_mu = 1e-2;
			s_numberOfSamples = 10;
            ref_lmmse = 0;
			%%%%%%%%%%%%%%%%%%

			% load data            
            [ m_training_data, m_test_data, m_training_adj , m_test_adj ] = MultikernelSimulations.getSixMonths(s_nodeNum,s_departureDelay);
            m_adjacency = sum(m_training_adj,3);
            m_adjacency = (m_adjacency+m_adjacency') /2;
            m_adjacency = m_adjacency > 100;
            
            sparsity = sum(m_adjacency(:))/(numel(m_adjacency)-size(m_adjacency,1))

			
			% data normalization
			v_mean = mean(m_training_data,2);
			v_std = std(m_training_data')';
			m_normalized_training_data = diag(1./v_std)*(m_training_data - v_mean*ones(1,size(m_training_data,2)));
			m_normalized_test_data = diag(1./v_std)*(m_test_data - v_mean*ones(1,size(m_test_data,2)));
			
			% covariance of normalized data
			m_covInv = MultikernelSimulations.learnInverseCov( cov(m_normalized_training_data') , m_adjacency );
			
            % approximation of inverse covariance via constrained Laplacian			
			%m_covInv = inv(C);
			m_constrainedLaplacian = MultikernelSimulations.approximateWithLaplacian(m_covInv,m_adjacency);
			
            
			% generator and sampler
			generator = RandomlyPickGraphFunctionGenerator('m_graphFunction',m_normalized_test_data);
			sampler = UniformGraphFunctionSampler('s_SNR',Inf,'s_numberOfSamples',s_numberOfSamples);
            
            % estimators
			cov_estimator = RidgeRegressionGraphFunctionEstimator('s_regularizationParameter',s_mu,'m_kernel',inv(m_covInv));
            
			% simulation
			estimator = cov_estimator;
			normalized_signal_mse = NaN(size(estimator,1),niter);			
			for s_itInd = 1:niter				
				% generate signal
				v_signal = generator.realization();
				
				% sample signal
				[v_samples,v_sampleLocations] = sampler.sample(v_signal);
                
                % estimate signal
                v_signalEst = cov_estimator.estimate(v_samples,v_sampleLocations);
                if ref_lmmse
                    % LMMSE estimator
                    v_signalEst = m_cov(:,v_sampleLocations)*(  (m_cov(v_sampleLocations,v_sampleLocations) + 1e-100*eye(length(v_sampleLocations)))\v_samples );
                end
                % revert normalization
                v_unnormalized_signal = (v_std).*v_signal + v_mean;
                v_unnormalized_signalEst = (v_std).*v_signalEst + v_mean;
				
                % measure error
				v_test_indices = 1:length(v_signal);
				v_test_indices(v_sampleLocations) = 0; v_test_indices = v_test_indices(v_test_indices>0);
				unnormalized_signal_mse(1,s_itInd) = norm( v_unnormalized_signal(v_test_indices) - v_unnormalized_signalEst(v_test_indices) )^2/(length(v_test_indices)) ;%/  norm( v_unnormalized_signal(v_test_indices)  )^2;
				unnormalized_signal_energy(1,s_itInd) =  norm( v_unnormalized_signal(v_test_indices)  )^2/(length(v_test_indices));
				               
			end
			
			% average error
            unnormalized_signal_mse = mean(unnormalized_signal_mse,2)
            unnormalized_signal_nmse = unnormalized_signal_mse / mean(unnormalized_signal_energy,2)

            rmse_in_minutes = sqrt(unnormalized_signal_mse)
			
			
			F = [];
			
        end
        
        
        function F = compute_fig_5203(obj,niter)
            % define parameters
			s_nodeNum = 30;
            s_departureDelay = 0;
			
			% estimation parameters
			s_numberOfSamples = 10;
            ref_lmmse = 0;
			%%%%%%%%%%%%%%%%%%

			% load data            
            [ m_training_data, m_test_data, m_training_adj , m_test_adj ] = MultikernelSimulations.getSixMonths(s_nodeNum,s_departureDelay);
            m_adjacency = sum(m_training_adj,3);
            m_adjacency = (m_adjacency+m_adjacency') /2;
            m_adjacency = m_adjacency > 100;
            
            sparsity = sum(m_adjacency(:))/(numel(m_adjacency)-size(m_adjacency,1))

			
			% data normalization
			v_mean = mean(m_training_data,2);
			v_std = std(m_training_data')';
			m_normalized_training_data = diag(1./v_std)*(m_training_data - v_mean*ones(1,size(m_training_data,2)));
			m_normalized_test_data = diag(1./v_std)*(m_test_data - v_mean*ones(1,size(m_test_data,2)));
			
			% covariance of normalized data
			m_covInv = MultikernelSimulations.learnInverseCov( cov(m_normalized_training_data') , m_adjacency );
			
            % approximation of inverse covariance via constrained Laplacian			
			%m_covInv = inv(C);
			m_constrainedLaplacian = MultikernelSimulations.approximateWithLaplacian(m_covInv,m_adjacency);
			
            
			% generator and sampler
            generator = RandomlyPickGraphFunctionGenerator('m_graphFunction',m_normalized_test_data);
            sampler = UniformGraphFunctionSampler('s_SNR',Inf,'s_numberOfSamples',s_numberOfSamples);
		
            
            % estimators
            s_mu = 1e-2;
			cov_estimator = RidgeRegressionGraphFunctionEstimator('s_regularizationParameter',s_mu,'m_kernel',inv(m_covInv));
          
            s_muMKL = 1e-4;
            s_sigma = sqrt(linspace(1,4,20));
            kG = LaplacianKernel('m_laplacian',m_constrainedLaplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(s_sigma));
            m_kernel1 = kG.getKernelMatrix();
            mkl_estimator1 = MkrGraphFunctionEstimator('s_regularizationParameter',s_muMKL, 'm_kernel', m_kernel1,'ch_type', 'kernel superposition');   % first 1
            
            s_sigma = sqrt(linspace(0.1,7,30));
            kG = LaplacianKernel('m_laplacian',m_constrainedLaplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(s_sigma));
            m_kernel2 = kG.getKernelMatrix();
            mkl_estimator2 = MkrGraphFunctionEstimator('s_regularizationParameter',s_muMKL, 'm_kernel', m_kernel2,'ch_type', 'kernel superposition');
            
            % BL estimator
            bandwidth_vec = [1 3 -1];
            bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian', m_constrainedLaplacian);
            bl_estimator = bl_estimator.replicate('s_bandwidth', num2cell(bandwidth_vec), [], {});
            
            estimator = [cov_estimator;mkl_estimator1;mkl_estimator2; bl_estimator];
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
			
			
			F = [];
			
        end
        
        function F = compute_fig_5204(obj,niter)
            % define parameters
			s_nodeNum = 50;
            s_departureDelay = 0;
			
			% estimation parameters
			s_numberOfSamples = 10;
            ref_lmmse = 0;
			%%%%%%%%%%%%%%%%%%

			% load data            
            [ m_training_data, m_test_data, m_training_adj , m_test_adj ] = MultikernelSimulations.getSixMonths(s_nodeNum,s_departureDelay);
            m_adjacency = sum(m_training_adj,3);
            m_adjacency = (m_adjacency+m_adjacency') /2;
            m_adjacency = m_adjacency > 100;
            
            sparsity = sum(m_adjacency(:))/(numel(m_adjacency)-size(m_adjacency,1))

			
			% data normalization
			v_mean = mean(m_training_data,2);
			v_std = std(m_training_data')';
			m_normalized_training_data = diag(1./v_std)*(m_training_data - v_mean*ones(1,size(m_training_data,2)));
			m_normalized_test_data = diag(1./v_std)*(m_test_data - v_mean*ones(1,size(m_test_data,2)));
			
			% covariance of normalized data
			m_covInv = MultikernelSimulations.learnInverseCov( cov(m_normalized_training_data') , m_adjacency );
			
            % approximation of inverse covariance via constrained Laplacian			
			%m_covInv = inv(C);
			m_constrainedLaplacian = MultikernelSimulations.approximateWithLaplacian(m_covInv,m_adjacency);
			
            
			% generator and sampler
            generator = RandomlyPickGraphFunctionGenerator('m_graphFunction',m_normalized_test_data);
            sampler = UniformGraphFunctionSampler('s_SNR',Inf,'s_numberOfSamples',s_numberOfSamples);
		
            
            % estimators
            s_mu = 1e-3;
			cov_estimator = RidgeRegressionGraphFunctionEstimator('s_regularizationParameter',s_mu,'m_kernel',inv(m_covInv));
          
            s_muMKL = 1e-4;
            s_sigma = sqrt(linspace(0.1,7,30));
            kG = LaplacianKernel('m_laplacian',m_constrainedLaplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(s_sigma));
            m_kernel1 = kG.getKernelMatrix();
            mkl_estimator1 = MkrGraphFunctionEstimator('s_regularizationParameter',s_muMKL, 'm_kernel', m_kernel1,'ch_type', 'RKHS superposition');   % first 1
            
            s_sigma = sqrt(linspace(0.1,7,30));
            kG = LaplacianKernel('m_laplacian',m_constrainedLaplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(s_sigma));
            m_kernel2 = kG.getKernelMatrix();
            mkl_estimator2 = MkrGraphFunctionEstimator('s_regularizationParameter',s_muMKL, 'm_kernel', m_kernel2,'ch_type', 'kernel superposition');
            
            % BL estimator
            bandwidth_vec = [1 3 8 -1];
            bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian', m_constrainedLaplacian);
            bl_estimator = bl_estimator.replicate('s_bandwidth', num2cell(bandwidth_vec), [], {});
            
            estimator = [cov_estimator;mkl_estimator1;mkl_estimator2; bl_estimator];
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
            
            %%
            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			% print the table into a tex file
			% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			fid = fopen('libGF/simulations/MultikernelSimulations_data/airport.tex','w');
			fprintf(fid, '\\begin{tabular}{%s}\n', char('c'*ones(1,length(estimator))));     % heading line
			fprintf(fid, '\t\\hline\n\t');
			fprintf(fid, 'RR with cov & MKL1 & MKL2 & BL1 & BL2 & BL3 & BL4\\\\\n');
			fprintf(fid, '\t\\hline\n\t');
			
			% print NMSE
			fprintf(fid, 'NMSE\t');
			for i = 1:length(estimator)
				fprintf(fid, ' & %2.2f', unnormalized_signal_nmse(i));
			end
			fprintf(fid, '\\\\\n\tRMSE(min)\t');
			% print variance
			for i = 1:length(estimator)
				fprintf(fid, ' & %2.2f', rmse_in_minutes(i));
			end
			fprintf(fid, '\\\\\n');
			fprintf(fid, '\t\\hline\n');
			fprintf(fid, '\\end{tabular}');		% bottom line
			%caption = Parameter.getTitle(graphGenerator,functionGenerator,sampler,estimator);
			%fprintf(fid, caption);
            
            fclose(fid);
			% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			
			
			F = [];
			
		end
        
        
		  
        function F = compute_fig_5205(obj,niter)
            % define parameters
			s_nodeNum = 50;
            s_departureDelay = 0;
			
			% estimation parameters
			s_numberOfSamples = 20;
            ref_lmmse = 0;
			%%%%%%%%%%%%%%%%%%

			% load data            
            [ m_training_data, m_test_data, m_training_adj , m_test_adj ] = MultikernelSimulations.getSixMonths(s_nodeNum,s_departureDelay);
            m_adjacency = sum(m_training_adj,3);
            m_adjacency = (m_adjacency+m_adjacency') /2;
            m_adjacency = m_adjacency > 100;
            
            sparsity = sum(m_adjacency(:))/(numel(m_adjacency)-size(m_adjacency,1))

			
			% data normalization
			v_mean = mean(m_training_data,2);
			v_std = std(m_training_data')';
			m_normalized_training_data = diag(1./v_std)*(m_training_data - v_mean*ones(1,size(m_training_data,2)));
			m_normalized_test_data = diag(1./v_std)*(m_test_data - v_mean*ones(1,size(m_test_data,2)));
			
			% covariance of normalized data
			m_covInv = MultikernelSimulations.learnInverseCov( cov(m_normalized_training_data') , m_adjacency );
			
            % approximation of inverse covariance via constrained Laplacian			
			%m_covInv = inv(C);
			m_constrainedLaplacian = MultikernelSimulations.approximateWithLaplacian(m_covInv,m_adjacency);
			
            
			% generator and sampler
            generator = RandomlyPickGraphFunctionGenerator('m_graphFunction',m_normalized_test_data);
            sampler = UniformGraphFunctionSampler('s_SNR',Inf,'s_numberOfSamples',s_numberOfSamples);
		
            
            % estimators
            s_mu = 1e-3;
			cov_estimator = RidgeRegressionGraphFunctionEstimator('s_regularizationParameter',s_mu,'m_kernel',inv(m_covInv));
          
            s_muMKL = 1e-4;
            s_sigma = sqrt(linspace(0.1,7,30));
            kG = LaplacianKernel('m_laplacian',m_constrainedLaplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(s_sigma));
            m_kernel1 = kG.getKernelMatrix();
            mkl_estimator1 = MkrGraphFunctionEstimator('s_regularizationParameter',s_muMKL, 'm_kernel', m_kernel1,'ch_type', 'RKHS superposition');   % first 1
            
            s_sigma = sqrt(linspace(0.1,7,30));
            kG = LaplacianKernel('m_laplacian',m_constrainedLaplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(s_sigma));
            m_kernel2 = kG.getKernelMatrix();
            mkl_estimator2 = MkrGraphFunctionEstimator('s_regularizationParameter',s_muMKL, 'm_kernel', m_kernel2,'ch_type', 'kernel superposition');
            
            % BL estimator
            bandwidth_vec = [2 3 8 -1];
            bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian', m_constrainedLaplacian);
            bl_estimator = bl_estimator.replicate('s_bandwidth', num2cell(bandwidth_vec), [], {});
            
            estimator = [cov_estimator;mkl_estimator1;mkl_estimator2; bl_estimator];
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
            
            %%
            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			% print the table into a tex file
			% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			fid = fopen('libGF/simulations/MultikernelSimulations_data/airport.tex','w');
			fprintf(fid, '\\begin{tabular}{%s}\n', char('c'*ones(1,length(estimator))));     % heading line
			fprintf(fid, '\t\\hline\n\t');
			fprintf(fid, 'RR with cov & MKL1 & MKL2 & BL1 & BL2 & BL3 & BL4\\\\\n');
			fprintf(fid, '\t\\hline\n\t');
			
			% print NMSE
			fprintf(fid, 'NMSE\t');
			for i = 1:length(estimator)
				fprintf(fid, ' & %2.2f', unnormalized_signal_nmse(i));
			end
			fprintf(fid, '\\\\\n\tRMSE(min)\t');
			% print variance
			for i = 1:length(estimator)
				fprintf(fid, ' & %2.2f', rmse_in_minutes(i));
			end
			fprintf(fid, '\\\\\n');
			fprintf(fid, '\t\\hline\n');
			fprintf(fid, '\\end{tabular}');		% bottom line
			%caption = Parameter.getTitle(graphGenerator,functionGenerator,sampler,estimator);
			%fprintf(fid, caption);
            
            fclose(fid);
			% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			
			
			F = [];
			
        end
        
		
	end
	
	% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% 7. Real data simulation on Swiss temperature dataset
	% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods
		
		
        % NMSE vs S for 
        %   - 2 MKL with 10 kernels
        %   - 2 MKL with 2 kernels
        %   - LS
        function F = compute_fig_7000(obj,niter)
			% define parameters
			max_iter = 1000;          % max iteration for learning laplacian
			alpha = 1;
			beta = 30;	%150          % alpha, beta paramters for learning laplacian
			S_vec = 10:5:60;		  % for creating uniform sampler
			B_vec = [2 5 10 30 -1];    % for creating BL estimator
			mu = 1e-4;                % regularization parameter for MK estimator
			SNR = Inf;
			
			% read temperature dataset and create the graph
			% However, if the graph is already exists, then skip the process
			addpath ./libGF/datasets/
			[Ho,Mo,Alto,Hn,Mn,Altn] = readTemperatureDataset();
			if (exist('learnedLaplacian.mat', 'file') == 2)
				load learnedLaplacian.mat
				m_adjacency = Graph.createAdjacencyFromLaplacian(L);
				graph = Graph('m_adjacency',m_adjacency);
			else
				% learn laplacian
				% use old tempearature to learn graph laplacian
				gl = GraphLearningSmoothSignalGraphGenerator('m_observed', Ho, 's_niter', max_iter, 's_alpha', alpha, 's_beta', beta);
				%gl = SmoothSignalGraphGenerator('m_observed', Ho, 's_niter', max_iter, 's_alpha', alpha, 's_beta', beta);
				graph = gl.realization();
			end
			m_laplacian = graph.getLaplacian(); 
            L = m_laplacian; save('learnedLaplacian.mat','L');
				
			%
			% define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);
			sampler = sampler.replicate([],{},'s_numberOfSamples',num2cell(S_vec));		
			%		
			% BL graph function estimator
			bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian);			
			bl_estimator.c_replicatedVerticallyAlong = {'ch_name'};
			bl_estimator = bl_estimator.replicate('s_bandwidth',num2cell(B_vec),'',{});
			
			
			% MKL function estimators
			sigma1_vec = sqrt(linspace(1, 20 , 10));
			sigma2_vec = sqrt([1 20]);
            kG = LaplacianKernel('m_laplacian',m_laplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigma1_vec));
			m_kernel{1} = kG.getKernelMatrix();
			kG = LaplacianKernel('m_laplacian',m_laplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigma2_vec));
			m_kernel{2} = kG.getKernelMatrix();
			
			mkl_estimator_RKHS = MkrGraphFunctionEstimator('s_regularizationParameter',mu);
			mkl_estimator_RKHS = mkl_estimator_RKHS.replicate('m_kernel', m_kernel, [], {} );
			
			%mkl_estimator_kernel = MkrGraphFunctionEstimator('s_regularizationParameter',mu,'ch_type','kernel superposition');
			%mkl_estimator_kernel = mkl_estimator_kernel.replicate('m_kernel', m_kernel, [], {} );
			
			mkl_estimator = [];
			%
			for i = 1:length(mkl_estimator_RKHS)
				mkl_estimator_RKHS(i).c_replicatedVerticallyAlong = {'ch_name','legendString'};
				mkl_estimator_replicated = mkl_estimator_RKHS(i).replicate('ch_type',{'RKHS superposition','kernel superposition'},'',[]);
				mkl_estimator = [mkl_estimator; mkl_estimator_replicated];
			end	
			est = [mkl_estimator;bl_estimator];
			
			%
			% Simulation
			nmse = zeros(length(est), length(sampler));
			for i = 1:size(Hn,2)
				generator = FixedGraphFunctionGenerator('graph',graph, 'graphFunction', Hn(:,i));
				%generator = FixedGraphFunctionGenerator('graph',graph, 'graphFunction', Mn);
				nmse = nmse + Simulate(generator, sampler, est, niter, true);
				%res = Simulator.simStatistic(niter,generator,sampler,est);
				%mse = Simulator.computeNmse(res,Results('stat',graphFunction));
			end
			nmse = nmse / size(Hn,2);

			% Representation			
			F = F_figure('X',S_vec,...
                'Y',nmse,'leg',Parameter.getLegend(generator,sampler, est),...
                'xlab', 'sample size','ylimit',...
				[0 1.1],'ylab','NMSE',...
				'tit',sprintf('Temperature dataset mu=%g',mu));
			
        end
        
       
        % NMSE vs S for 
        %   - 2 MKL with 10 kernels
        %   - 2 RR with 2 different kernels
        %   - LS
        function F = compute_fig_7001(obj,niter)
			% define parameters
			max_iter = 1000;          % max iteration for learning laplacian
			alpha = 1;
			beta = 30;	%150          % alpha, beta paramters for learning laplacian
			S_vec = 10:5:60;		  % for creating uniform sampler
			B_vec = [5 10 20 -1];    % for creating BL estimator
			mu = 1e-4;                % regularization parameter for MK estimator
			SNR = Inf;
			
			% read temperature dataset and create the graph
			% However, if the graph is already exists, then skip the process
			addpath ./libGF/datasets/
			[Ho,Mo,Alto,Hn,Mn,Altn] = readTemperatureDataset();
			if (exist('learnedLaplacian.mat', 'file') == 2)
				load learnedLaplacian.mat
				m_adjacency = Graph.createAdjacencyFromLaplacian(L);
				graph = Graph('m_adjacency',m_adjacency);
			else
				% learn laplacian
				% use old tempearature to learn graph laplacian
				gl = GraphLearningSmoothSignalGraphGenerator('m_observed', Ho, 's_niter', max_iter, 's_alpha', alpha, 's_beta', beta);
				%gl = SmoothSignalGraphGenerator('m_observed', Ho, 's_niter', max_iter, 's_alpha', alpha, 's_beta', beta);
				graph = gl.realization();
			end
			m_laplacian = graph.getLaplacian(); 
            L = m_laplacian; save('learnedLaplacian.mat','L');
				
			%
			% define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);
			sampler = sampler.replicate([],{},'s_numberOfSamples',num2cell(S_vec));		
			%		
			% BL graph function estimator
			bl_estimator = BandlimitedGraphFunctionEstimator('m_laplacian',graph.getLaplacian);			
			bl_estimator.c_replicatedVerticallyAlong = {'ch_name'};
			bl_estimator = bl_estimator.replicate('s_bandwidth',num2cell(B_vec),'',{});
			
			
			% MKL function estimators
			sigma1_vec = sqrt(linspace(1, 20, 10));
			sigma2_vec = sqrt([1 2 5]);
            kG = LaplacianKernel('m_laplacian',m_laplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigma1_vec));
			m_kernel1 = kG.getKernelMatrix();
			kG = LaplacianKernel('m_laplacian',m_laplacian,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigma2_vec));
			m_kernel2 = kG.getKernelMatrix();
			
			mkl_estimator_RKHS = MkrGraphFunctionEstimator('s_regularizationParameter',mu, 'm_kernel', m_kernel1);   % first 1
			%mkl_estimator_RKHS = mkl_estimator_RKHS.replicate('m_kernel', m_kernel, [], {} );
			
			%mkl_estimator_kernel = MkrGraphFunctionEstimator('s_regularizationParameter',mu,'ch_type','kernel superposition');
			%mkl_estimator_kernel = mkl_estimator_kernel.replicate('m_kernel', m_kernel, [], {} );
			
			%mkl_estimator = [];
			%
			%for i = 1:length(mkl_estimator_RKHS)
            mkl_estimator_RKHS.c_replicatedVerticallyAlong = {'ch_name','legendString'};
            mkl_estimator_replicated = mkl_estimator_RKHS.replicate('ch_type',{'RKHS superposition','kernel superposition'},'',[]);
            
            mkl_estimator_RR(1) = MkrGraphFunctionEstimator('s_regularizationParameter',mu, 'm_kernel', m_kernel2(:,:,1), 'ch_type', 'kernel superposition', 's_sigma', sigma2_vec(1));
            mkl_estimator_RR(2) = MkrGraphFunctionEstimator('s_regularizationParameter',mu, 'm_kernel', m_kernel2(:,:,2), 'ch_type', 'kernel superposition', 's_sigma', sigma2_vec(2));
            mkl_estimator_RR(3) = MkrGraphFunctionEstimator('s_regularizationParameter',mu, 'm_kernel', m_kernel2(:,:,3), 'ch_type', 'kernel superposition', 's_sigma', sigma2_vec(3));
            %mkl_estimator = [mkl_estimator; mkl_estimator_replicated];
			%end	
			est = [mkl_estimator_replicated;mkl_estimator_RR';bl_estimator];
			
			%
			% Simulation
			nmse = zeros(length(est), length(sampler));
			for i = 1:size(Hn,2)
				generator = FixedGraphFunctionGenerator('graph',graph, 'graphFunction', Hn(:,i));
				%generator = FixedGraphFunctionGenerator('graph',graph, 'graphFunction', Mn);
				nmse = nmse + Simulate(generator, sampler, est, niter, true);
				%res = Simulator.simStatistic(niter,generator,sampler,est);
				%mse = Simulator.computeNmse(res,Results('stat',graphFunction));
			end
			nmse = nmse / size(Hn,2);

			% Representation			
			F = F_figure('X',S_vec,...
                'Y',nmse,'leg',Parameter.getLegend(generator,sampler, est),...
                'xlab', 'sample size','ylimit',...
				[0 1.1],'ylab','NMSE',...
				'tit',sprintf('Temperature dataset mu=%g',mu));
			
		end
        
        % Print version of 7001
		function F = compute_fig_7002(obj,niter)
			 F = obj.load_F_structure(7001);
			 F.styles = {'-^','-v','--o','--s','--d','-.*','-.x','-.+','-.p'};
			 F.translation_table = {'kernel superposition','KS';'RKHS superposition','RS';'Ass. B = cut-off freq.,','cut-off';...
				 'Bandlimited, Ass.','BL for';'Multi-kernel, 1 kernel,','KRR,';'10 kernels,','';', KS','';'5,','5';'10,','10';'Bandlimited','BL'};
			 F.tit = '';
			 F.xlab = 'Sample size (S)';
		 end
		
		
		% find the sigma range for temperature dataset
		function F = compute_fig_7010(obj,niter)
			
			max_iter = 1000;     % max iteration for learning laplacian
			alpha = 1;
			beta = 10;			 % alpha, beta paramters for learning laplacian
			S_Vec = 10:10:60;	 % for creating uniform sampler
			%B_vec = [20 40 60 -1];    % for creating BL estimator
			mu = 1e-4;           % regularization parameter for MK estimator
			SNR = Inf;
			
			%
			% read temperature dataset and create the graph
			% However, if the graph is already exists, then skip the process
			addpath ./libGF/datasets/
			[Ho,Mo,Alto,Hn,Mn,Altn] = readTemperatureDataset();
			if exist('learnedLaplacian.mat', 'file') == 2
				load learnedLaplacian.mat
				m_adjacency = Graph.createAdjacencyFromLaplacian(L);
				graph = Graph('m_adjacency',m_adjacency);
			else
				% learn laplacian
				% use old tempearature to learn graph laplacian
				gl = GraphLearningSmoothSignalGraphGenerator('m_observed', Ho, 's_niter', max_iter, 's_alpha', alpha, 's_beta', beta);
				%gl = SmoothSignalGraphGenerator('m_observed', Ho, 's_niter', max_iter, 's_alpha', alpha, 's_beta', beta);
				graph = gl.realization();
			end
			%m_laplacian = graph.getLaplacian(); 
						
			% generate graph and signal
			%graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', p,'s_numberOfVertices',N);
			%graph = graphGenerator.realization();
			%functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',30);
			%functionGenerator = ExponentiallyDecayingGraphFunctionGenerator('graph',graph,'s_bandwidth',30,'s_decayingRate',.5);
			%m_graphFunction = functionGenerator.realization();
			m_graphFunction = Mn;
            generator =  FixedGraphFunctionGenerator('graph',graph,'graphFunction',m_graphFunction);
			
			% 3. generate Kernel matrix
			sigmaArray = sqrt(linspace(0.01, 20, 30));
			L = graph.getLaplacian();
            kG = LaplacianKernel('m_laplacian',L,'h_r_inv',LaplacianKernel.diffusionKernelFunctionHandle(sigmaArray));
			m_kernel = kG.getKernelMatrix();
            
            
			% 4. define graph function sampler
			sampler = UniformGraphFunctionSampler('s_SNR',SNR);
            sampler = sampler.replicate('s_numberOfSamples', num2cell(S_Vec),[],{}); 
			
			% 5. define function estimator
			N = size(L,1);
            estimator = MkrGraphFunctionEstimator('s_regularizationParameter',mu);
            estimator = estimator.replicate([],{}, ...
                'm_kernel', mat2cell(m_kernel, N, N, ones(1,size(m_kernel,3))));
			%%
			% Simulation
            mse = Simulate(generator, sampler, estimator, niter);
            %%
            % Representation
            F = F_figure('X',sigmaArray.^2,'Y',mse, ...
                'leg',Parameter.getLegend(generator,sampler, estimator),...
                'xlab','\sigma^2','ylab','Normalized MSE',...
                'tit', sprintf('N=%d, \\mu=%3.1d', N, mu),...
				'leg_pos','northwest');		  
		end	
	end
	
	methods(Static)
		
		% =========================================================================
		% utility functions
		% =========================================================================
        function [N,p,SNR,sampleSize, bandwidth] = simulationSetting()
            % generate some commonly used paramters across simulations in order
            % to make all the simulations consistent
            N = 100;  % # of vertices
            p = 0.25; % edge existence prob of Erdos Renyi random graph model
            sampleSize = 40;
            bandwidth = 30;
            SNR = 20; % dB
        end
        
		function NMSE = sim_MKL(trueSignal,S, SNR,estimator,MONTE_CARLO)
			signalPower = norm(trueSignal)^2/length(trueSignal);
			noisePower = signalPower / 10^(SNR/10);
			
			N = length(trueSignal);
			N_SE = zeros(MONTE_CARLO,1);
			for iMonteCarlo = 1 : MONTE_CARLO
				% random generate a sample set
				componentArray = partition_set(N, S);
				sampleSet = componentArray(1).index;
				
				% generate observed signal
				observedSignal = trueSignal(sampleSet) + ...
					sqrt(noisePower) * randn(S,1);
				
				% estimate signal using the estimator
				estimatedSignal = estimator( sampleSet, observedSignal );
				
				% compute square error
				N_SE(iMonteCarlo) = norm(estimatedSignal - trueSignal)^2 / norm(trueSignal)^2;
			end
			
			NMSE = median(N_SE);
			
		end
		
		function Kcol = columnLaplacianKernelCircularGraph(vertexNum,rFun,columnInd)
			% Kcol is a vertexNum x 1 vector that corresponds to the
			% columnInd-th column of the Laplacian kernel matrix of a
			% circular graph when the r function is rFun. 
			%
			% rFun must accept vector-valued inputs.
			
			Dinds = (1:vertexNum)-columnInd;
			
			for rowInd = 1:vertexNum				
				Kcol(rowInd,1) = (1/vertexNum)*sum( exp(1j*2*pi/vertexNum*(0:vertexNum-1)*Dinds(rowInd))./rFun(2*(1-cos(2*pi/vertexNum*(0:vertexNum-1)))));
			end
			Kcol = real(Kcol);
        end
        
        function printSimulationProgress(outer_iter, inner_iter, max_outer_iter, max_inner_iter)
            ROW = max_outer_iter;
            COL = max_inner_iter;
            iRow = outer_iter;
            iCol = inner_iter;
            fprintf('Simulation progress\t%3.1f%%\n', ...
                100*(iCol+(iRow-1)*COL)/(ROW*COL) );
        end
		
        
        function [ m_training_delay, m_test_delay, m_training_adj , m_test_adj ] = getTwoMonths(s_selectedAirportNum,s_departureDelay)
            % s_selectedAirportNum: number of the most crowded airports
            % that will be returned.
            %
            % s_departure_delay:   1: departure delay
            %                      0: arrival delay
            
            folder = 'libGF/datasets/AirportsDataset/';
            %% load dataset
            load([folder 'delaydata2014'])
            depDelaySep2014 = delayData.depDelay{3};
            arrDelaySep2014 = delayData.arrDelay{3};
            airportListSep2014 = delayData.airportList{3};
            adjSep2014 = delayData.adjacency{3};                           
            
            %%            
            load([folder 'delaydata2015'])
            depDelaySep2015 = delayData.depDelay{3};
            arrDelaySep2015 = delayData.arrDelay{3};
            airportListSep2015 = delayData.airportList{3};
            adjSep2015 = delayData.adjacency{3};
            
            %% Sort data in the same order
            % 1. Find common airports
            [v_commonAirports,v_inds2014,v_inds2015] = intersect( airportListSep2014 , airportListSep2015 );
            %nn = norm(  airportListSep2014(v_inds2014) - airportListSep2015(v_inds2015))
                                 
            % 2. Sort data
            depDelaySep2014 = depDelaySep2014(v_inds2014,:);
            arrDelaySep2014 = arrDelaySep2014(v_inds2014,:);
            adjSep2014 = adjSep2014(v_inds2014,v_inds2014,:);
            
            depDelaySep2015 = depDelaySep2015(v_inds2015,:);
            arrDelaySep2015 = arrDelaySep2015(v_inds2015,:);
            adjSep2015 = adjSep2015(v_inds2015,v_inds2015,:);
            
            % check
            %s = [sum(adjSep2014(1:100,1:10,:),3) sum(adjSep2015(1:100,1:10,:),3)]
            
            
           
            
			%% select busiest airports
			A = sum(adjSep2014,3);			
			[~,v_inds] = sort(sum(A,2),'descend');
			v_inds = v_inds(1:s_selectedAirportNum); % indices of the most crowded airports
			
            if s_departureDelay            
                m_training_delay = depDelaySep2014(v_inds,:);
                m_test_delay = depDelaySep2015(v_inds,:);              
            else                
                m_training_delay = arrDelaySep2014(v_inds,:);
                m_test_delay = arrDelaySep2015(v_inds,:);                
            end
            m_training_adj = adjSep2014(v_inds,v_inds,:);
            m_test_adj = adjSep2015(v_inds,v_inds,:);   
            
            %m_cov = cov([m_training_delay m_test_delay]')
            
            
		end
		
		
		 function [ m_training_delay, m_test_delay, m_training_adj , m_test_adj ] = getSixMonths(s_selectedAirportNum,s_departureDelay)
            % s_selectedAirportNum: number of the most crowded airports
            % that will be returned.
            %
            % s_departure_delay:   1: departure delay
            %                      0: arrival delay
            	
			[comDepCell, comArrCell, comAdjCell] = MultikernelSimulations.get6monthData;
			
			
			if s_departureDelay
				comCell = comDepCell;
			else
				comCell = comArrCell;
			end
			m_training_delay = [];
			m_training_adj = zeros(size(comAdjCell{1},1),size(comAdjCell{1},2));
			for k = 1:length(comCell)-1
				m_training_delay = [m_training_delay comCell{k}];
				m_training_adj = m_training_adj + sum(comAdjCell{k},3);
			end
			m_test_delay = comCell{end};
			m_test_adj = sum(comAdjCell{end},3);
			
            
            
			%% select busiest airports
			A = m_training_adj;
			[~,v_inds] = sort(sum(A,2),'descend');
			v_inds = v_inds(1:s_selectedAirportNum); % indices of the most crowded airports
			
            
			m_training_delay = m_training_delay(v_inds,:);
			m_test_delay = m_test_delay(v_inds,:);
            
            m_training_adj = m_training_adj(v_inds,v_inds,:);
            m_test_adj = m_test_adj(v_inds,v_inds,:);   
            
            %m_cov = cov([m_training_delay m_test_delay]')
            
            
		end
	
		
		
		
	
			
		
		
        function [ commonDelay1, commonDelay2, commonAdj1, commonAdj2] = getCommonData(delay1, adj1, airportList1, delay2, adj2, airportList2)
            % s_selectedAirportNum: number of the most crowded airports
            % that will be returned.
            %
            % s_departure_delay:   1: departure delay
            %                      0: arrival delay
            
            %% Sort data in the same order
            % 1. Find common airports
            [v_commonAirports,commonInd1,commonInd2] = intersect( airportList1 , airportList2 );
            %nn = norm(  airportListSep2014(v_inds2014) - airportListSep2015(v_inds2015))
            
            assert( isequal(v_commonAirports, airportList1(commonInd1)) );
            
            % 2. Sort data
            commonDelay1 = delay1(commonInd1,:);
            commonAdj1 = adj1(commonInd1,commonInd1,:);
            
            commonDelay2 = delay2(commonInd2,:);
            commonAdj2 = adj2(commonInd2,commonInd2,:);
            
            % check
            %s = [sum(adjSep2014(1:100,1:10,:),3) sum(adjSep2015(1:100,1:10,:),3)]       
        end
        
        function [commonList, indexCell]  = getCommonIndex( listCell )
            commonList = listCell{1};
            for i = 2 : length(listCell)
                commonList = intersect(commonList, listCell{i});
            end
            
            for i = 1 : length(listCell)
                for k = 1 : length(commonList)
                    index(k) = find( listCell{i} == commonList(k) );
                end
                indexCell{i} = index;
            end
        end
        
		
		function m_laplacian = approximateWithLaplacian(m_input,m_adjacency)
			% m_laplacian is the best Laplacian matrix approximating matrix
			% m_input in the Frobenius norm
			% m_adjacency is an optional parameter. m_laplacian is such
			% that m_laplacian(i,j) = 0 if m_adjacency(i,j) = 0  (i~=j)
			%
			s_nodeNum = size(m_input,1);
			if nargin<2
				m_adjacency = ones(s_nodeNum);
			end
			m_adjacency = m_adjacency + triu(ones(s_nodeNum));
			m_mask = (m_adjacency == 0);
			
			cvx_begin
			   variable L(s_nodeNum,s_nodeNum) symmetric
			minimize( norm(L - m_input,'fro') )
			subject to
			   L*ones(s_nodeNum,1) == zeros(s_nodeNum,1);
			   triu(L,1) <= 0;
			   L(m_mask) == 0;
			cvx_end
			
			m_laplacian = L;
		end
		
		function m_covInv = learnInverseCov( m_sampleCov , m_adjacency )
			% Learns the inverse covariance of a normal distribution 
			% m_adjacency is optional. m_covInv is such
			% that m_covInv(i,j) = 0 if m_covInv(i,j) = 0  (i~=j)If given, then 
			% 
			
			d = size(m_sampleCov,1);
			m_adjacency = m_adjacency + triu(ones(d));
			m_mask = (m_adjacency == 0);
			
			cvx_begin
			   variable S(d,d) symmetric
			minimize( -log_det(S) +trace(S*m_sampleCov) )
 			subject to
 			   S(m_mask) == 0;
			cvx_end
			
			m_covInv = S;
			
		end
		
		

        function [comDepCell, comArrCell, comAdjCell] = get6monthData
			folder = 'libGF/datasets/AirportsDataset/';
			
             %% load dataset
            load([folder 'delaydata2014'])
            depDelay2014 = delayData.depDelay;
            arrDelay2014 = delayData.arrDelay;
            airportList2014 = delayData.airportList;
            adj2014 = delayData.adjacency;          
            %%            
            load([folder 'delaydata2015'])
            depDelay2015 = delayData.depDelay;
            arrDelay2015 = delayData.arrDelay;
            airportList2015 = delayData.airportList;
            adj2015 = delayData.adjacency;
            
            
            %%
            airportListCell = [airportList2014, airportList2015];
            depDelayCell = [depDelay2014, depDelay2015];
            arrDelayCell = [arrDelay2014, arrDelay2015];
            adjCell = [adj2014, adj2015];
            
            %% find common airports and their index
            [~, indexCell]  = MultikernelSimulations.getCommonIndex( airportListCell );
            
            %% get data
            for i = 1 : length(indexCell)
                comDepCell{i} = depDelayCell{i}(indexCell{i},:);
                comArrCell{i} = arrDelayCell{i}(indexCell{i},:);
                comAdjCell{i} = adjCell{i}(indexCell{i}, indexCell{i}, :);
            end
        end
        
        
	end
	
	
end
