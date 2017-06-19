classdef Simulator

	
	% Requirements for classes extending Simulator
	%   - simulation methods should be static
	%   - all output arguments must be scalar. If we would like to return
	%   a matrix, we would have to create a struct/class and include the
	%   matrix as one of its fields. Use for example an object of the class
	%   Results.

	% changes: now a realization corresponds with the third dimension within
	%   each element of results.
	
	properties
	%	x;
		
	end
	
% 	methods 
% 		function obj = Simulator( xin )
% 		
% 			%obj.x = xin;
% 		end
% 	end	
% 	
	methods(Static)
		
		
		function res = simStatistic(niter,gen,samp,est)
			% every element of the M x N array  RES of class Results is a 
			%     size( output of STAT , 1 ) x size( output of STAT , 2 ) x NITER 
			% matrix with realizations of the output of STAT in the field 'stat'
			%
			%   M = max( size(GEN,1) , size(IPROC,1) , size(STAT,1) )
			%   N = max( size(GEN,2) , size(IPROC,2) , size(STAT,2) )
			%
			% The chain is:
			%      GEN   ->   SAMP   ->  EST
			%
			% where GEN is a signal generator, SAMP a sampler
			% and EST an estimator operating on the output of SAMP
			%
			% see "simStatisticOneParamMC" below to see the core of the simulator
			%			
			res = Simulator.mapOutputs('Simulator.simStatisticOneParam',niter,gen,samp,est);

		end
		
		function res = simStatisticOneParam(niter,gen,iproc,stat)
			
			stat = Simulator.breakOneParamSims('Simulator.simStatisticOneParamMC',niter,gen,iproc,stat);
			
			res = Results('stat',stat);			

		end
		
		function stats = simStatisticOneParamMC(niter,gen,iproc,stat)
			
			x = gen.realization(niter);
			if (size(x,2) > 1) && (size(x,3) == 1)
				x = permute(x,[1,3,2]);
			end
			for k = niter:-1:1				
				x_now = x(:,:,k);
				[outsig,sideInfo] = iproc.sample(x_now);

				stats(:,:,k) = stat.estimate(outsig,sideInfo);				
			end

		end		
		
		function res = breakOneParamSims(funname,niter, varargin)
			% This function segments the input of the function
			%     OUTS_MAT = funname( niter, varargin{1},varargin{2}...)
			% into series of invokations with fewer iterations. This
			% function must produce a N x M x NITER matrix, where N x M
			% is the dimension of the output (of any class) of
			% funname. Each column is expected to be one realization of
			% something.
			%
			% I think that the function FUNNAME must be static, since I
			% think using something like "funname = myobj.myfun()" is not
			% gonna work, since myobj is not visible here.
			%
			max_niter = 10000;
			niter_vec = [ max_niter*ones(1, floor(niter/max_niter)) mod(niter,max_niter)*ones(1,mod(niter,max_niter)>0)];
			
			%res(niter) = Results();
			
			for k=length(niter_vec):-1:1
				inds_now =  sum(niter_vec(1:k-1)) + 1 : sum(niter_vec(1:k-1)) + niter_vec(k);
				res(:,:,inds_now) = feval(funname,niter_vec(k),varargin{:});
			end
			
			
		end
		
		function mse = computeMse( res_input , res_ref )
			% mse is a matrix with the same size as res_input
			% res_ref is a scalar/matrix/vector whose dimensions are consistent 
			% with those of res_input.
			%
			% 
			
			% matching the size of res_ref to that of res_input
			if (size(res_input,1)>1) && (size(res_ref,1)==1 )
				res_ref = repmat(res_ref,size(res_input,1),1);
			end
			if (size(res_input,2)>1) && (size(res_ref,2)==1 )
				res_ref = repmat(res_ref,1,size(res_input,2));
			end
						
			% mse computation
			for ncol = size(res_input,2):-1:1
				for nrow = size(res_input,1):-1:1
					
					stat_vals = res_input(nrow,ncol).getByName('stat');
					ref_vals =  res_ref(nrow,ncol).getByName('stat');
					mse_now = 0;
					for nr = 1:size(stat_vals,3)						
						mse_now = mse_now + sum(sum( (abs(stat_vals(:,:,nr)-ref_vals)).^2))/numel(ref_vals);
					end
						
					mse(nrow,ncol) = mse_now/size(stat_vals,3);
				end
			end
			
			
		end
		
		
		function nmse = computeNmse( res_input , res_ref )
			% NMSE is a matrix with the same size as res_input
			% res_ref is a scalar/matrix/vector whose dimensions are consistent 
			% with those of res_input. NMSE contains the normalized mean
			% squared error
			%
			% 
			
			% matching the size of res_ref to that of res_input
			if (size(res_input,1)>1) && (size(res_ref,1)==1 )
				res_ref = repmat(res_ref,size(res_input,1),1);
			end
			if (size(res_input,2)>1) && (size(res_ref,2)==1 )
				res_ref = repmat(res_ref,1,size(res_input,2));
			end
						
			% mse computation
			for ncol = size(res_input,2):-1:1
				for nrow = size(res_input,1):-1:1
					
					stat_vals = res_input(nrow,ncol).getByName('stat');
					ref_vals =  res_ref(nrow,ncol).getByName('stat');
					
					nmse_now = 0;
					for nr = 1:size(stat_vals,3)						
						nmse_now = nmse_now + sum(sum( (abs(stat_vals(:,:,nr)-ref_vals)).^2))./sum(sum( (abs(ref_vals)).^2));
					end
					
					nmse(nrow,ncol) = nmse_now/size(stat_vals,3);
				end
			end
			
			
		end
		
		
		
		function plot_histograms( d0 , d1 )
			
			% truncation
			l = length(d0);
			lout = floor(0.05*l);
			d0 = sort(d0);
			d0 = d0(lout+1:end-lout);			
			d1 = sort(d1);
			d1 = d1(lout+1:end-lout);
			
			nbins = 100;
			[N0,X0] = hist(d0,nbins);
			[N1,X1] = hist(d1,nbins);
			
			plot(X0,N0/sum(N0)/(X0(2)-X0(1)),random_color());
			plot(X1,N1/sum(N1)/(X1(2)-X1(1)),random_color());

			
			
		end
		
		
		function  varargout  = mapOutputs( funname , varargin )
			% MAPOUTPUTS invokes function FUNNAME M*N times according to
			% the M x N matrix expansion of the arguments in VARARGIN.
			%
			% OUTPUT:
			% each element of varargout is a matrix with M rows and N
			% columns.
			% INPUT
			% The elements of varargin can be M x N, M x 1, 1 x N or 1 x 1.
			% 
						
			ninputs = nargin - 1;
			noutputs = nargout; 
			
			for k = 1:ninputs
				nrows(k) = size(varargin{k},1);
				ncols(k) = size(varargin{k},2);
			end
			M = max(nrows);
			N = max(ncols);
			if sum( (nrows~=1)&(nrows~=M) )>0
				error('number of rows of input arguments must be 1 or M');
			end
			if sum( (ncols~=1)&(ncols~=N) )>0
				error('number of cols of input arguments must be 1 or M');
			end
			for k = 1:ninputs
				if size(varargin{k},1)==1
					varargin{k}=repmat(varargin{k},M,1);
				end
				if size(varargin{k},2)==1
					varargin{k}=repmat(varargin{k},1,N);
				end
			end

			outargs = cell(1,noutputs);
			[outargs{:}] = Simulator.mapOutputsToAllMatrixInputs( funname , varargin{:} );
			for k=1:noutputs
				varargout{k} = outargs{k};
			end
		end
		       	
		function outargs = my_feval(noutputs,funname,inargs)
			outargs = cell(1,noutputs);
			[outargs{:}]=feval(funname,inargs{:});
		end
		
		function element = mat_cell_element(in_mat_cell,row,col)
			
			element{length(in_mat_cell)}=[];
			for k=1:length(in_mat_cell)
				element{k} = in_mat_cell{k}(row,col);				
			end
		end
		
		% nonparallel simulation		
		function  varargout  = mapOutputsToAllMatrixInputs_npar( funname , varargin )
			% here all elements of varargin must be M x N matrices. Each
			% element of varargout is an M x N matrix
			ninputs = nargin - 1;
			noutputs = nargout;
			M = size(varargin{1},1);
			N = size(varargin{1},2);
			%outargs = cell(1,noutputs);
			inargs = varargin;
			for m = 1:M
				%fprintf('Computing curve %d\n',m);
				for n = 1:N
					outargs = cell(1,noutputs);
					for k=1:ninputs						 
						inargs{k} = varargin{k}(m,n);
					end
					[outargs{:}]=feval(funname,inargs{:});

					for k=1:noutputs
						varargout{k}(m,n)=outargs{k};
					end
				end
			end						
		end
		
		 % parallel simulation. The one used is the one with name
        % "mapOutputsToAllMatrixInputs". Change the names if you want to
        % run parallel/non-parallel simulations
		function  varargout  = mapOutputsToAllMatrixInputs( funname , varargin )
			% here all elements of varargin must be M x N matrices. Each
			% element of varargout is an M x N matrix
			ninputs = nargin - 1;
			noutputs = nargout;
			M = size(varargin{1},1);
			N = size(varargin{1},2);
			%outargs = cell(1,noutputs);
			%inargs = varargin;
			%argout(M,N) = {};
			
			%vectorize
			for k=1:ninputs
				inargs_vecs{k} = varargin{k}(:);
			end
			for mn = 1:N*M
				inargs{mn} = Simulator.mat_cell_element(inargs_vecs,mn,1);
			end
			
			%for mn = 1:N*M			
			for mn = 1:N*M
				    pftic = tic;
					inargs_now = inargs{mn};
					
					% this is not implemented for multiple outputs. Change
					% to mapOutputsToAllMatrixInputs_npar if you prefer.
					argout(mn) = Simulator.my_feval(noutputs, funname,inargs_now);%outargs;				
				
					output_time_parfor_iteration(mn) = toc(pftic);
				fprintf('parfor iter %d finished in %s\n',mn,print_time(output_time_parfor_iteration(mn)));
            end
            output_time_parfor_iteration = reshape( output_time_parfor_iteration , M , N )
	
			%fprintf('out of parfor\n');
			rsargout = reshape(argout,M,N);
			for k=1:noutputs
				for m=1:M
					for n=1:N
						cell_now = rsargout(m,n);
						varargout{k}(m,n) = cell_now{k};
					end
				end
			end
			
		end
		
		
	end
	
end




% 
% 
% 		function  varargout  = mapOutputs( funname , varargin )
% 			% MAPOUTPUTS invokes function FUNNAME once per element in the
% 			% matrix elements of the  output cell array varargout with the
% 			% corresponding elements in varargin. 			
% 			%
% 			% OUTPUT:
% 			% each element of varargout is a matrix with M rows and N
% 			% columns.
% 			% INPUT
% 			% two options:
% 			% A)
% 			% among the elements of varargin, all but two must be scalar.
% 			% One of the non-scalar elements (say A) can be a vector with M
% 			% rows and the other a vector (say B) with N columns. Then,
% 			% every element in varargout has one row per element in A and
% 			% one column per element in B.
% 			% B)
% 			% one element is an M x N matrix. the others are scalars
% 						
% 			ninputs = nargin - 1;
% 			noutputs = nargout; 
% 			ir = zeros(1,ninputs);
% 			ic = zeros(1,ninputs);
% 			im = zeros(1,ninputs);
% 			for k = 1:ninputs
% 				ir(k) = isrow( varargin{k} );
% 				ic(k) = iscolumn( varargin{k} );
% 				im(k) = ismat( varargin{k} );
% 			end
% 			if (sum(ir)>1)
% 				error('at most 1 argument can be row vector');
% 			end
% 			if (sum(ic)>1)
% 				error('at most 1 argument can be column vector');
% 			end		
% 			if (sum(im)>1)
% 				error('at most 1 argument can be a matrix');
% 			end
% 			if (sum(im)==1)&&(sum(ir)+sum(ic)>0)
% 				error('If one element is a matrix, the others must be scalars');
% 			end
% 			
% 			if sum(im)==1 
% 				outargs = cell(1,noutputs);
% 				[outargs{:}]  = Simulator.mapOutputsToMatrixInputs( funname , varargin{:} );
% 				for k=1:noutputs
% 					varargout{k}=outargs{k};
% 				end
% 				return
% 			end
% 				
% 			rind = find(ir);
% 			cind = find(ic);
% 			if (isempty(rind))&&(isempty(cind))
% 				rind = -1;
% 				cind = -1;
% 			elseif (~isempty(rind))&&(isempty(cind))
% 				cind = -1;
% 				rvec = varargin{rind};
% 				cvec = {};
% 			elseif (isempty(rind))&&(~isempty(cind))
% 				rind = -1;
% 				cvec = varargin{cind};
% 				rvec = {};
% 			else
% 				rvec = varargin{rind};
% 				cvec = varargin{cind};
% 			end
% 			
% 			inargs = varargin;
% 			outmats = cell(1,noutputs);
% 			for m = 1:max(1,length(cvec))
% 				for n = 1:max(1,length(rvec))
% 					outargs = cell(1,noutputs);
% 					if cind ~= -1
% 						inargs{cind} = cvec(m);
% 					end
% 					if rind ~= -1
% 						rind
% 						rvec(n)
% 						inargs{rind} = rvec(n);
% 					end
% 					[outargs{:}]=feval(funname,inargs{:});
% 					for k=1:noutputs
% 						outmats{k}(m,n)=outargs{k};
% 					end
% 				end
% 			end
% 			for k=1:noutputs
% 				varargout{k}=outmats{k};
% 			end						
% 		end
% 		
