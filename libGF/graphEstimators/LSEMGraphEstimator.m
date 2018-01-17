classdef LSEMGraphEstimator< GraphEstimator
	% This was written by Vassilis
	%KalmanFilterOnGraphsFunctionEstimator
	%it its implemented for one step prediction
	%needs initial conditions
	%it is programmed so that it resets the new values
	%needed for next iteration
	
	properties(Constant)
	end
	
	properties % Required by superclass Parameter
		c_parsToPrint    = {};
		c_stringToPrint  = {};
		c_patternToPrint = {};
	end
	%%
	properties
		ch_name = 'LSEMGraphEstimator';
        s_lambda1;
        s_lambda2;
        m_exogenous=-1;
        s_rho;
        s_tol;
        s_maxIter;
        b_display=1;
        m_noisyAdjacency=-1;
	end
	
	
	methods
		
		function obj = LSEMGraphEstimator(varargin)
			obj@GraphEstimator(varargin{:});
		end

	end
	
	methods
		function [m_adjacency]=estimate(obj,m_graphFunction)
            %no exogenous input
            s_lambda1=obj.s_lambda1;
            s_lambda2=obj.s_lambda2;
            b_display=obj.b_display;
            s_tol=obj.s_tol;
            s_rho=obj.s_rho;
            s_maxIter=obj.s_maxIter;
            m_graphFunction=m_graphFunction';
            [T,N] = size(m_graphFunction);
            m_K_Y = m_graphFunction'*m_graphFunction; %inner product matrices
            %K_X = X'*X;
            %K_YX = Y'*X;
            m_tmp_inv = inv(m_K_Y + (s_lambda2+s_rho)*eye(N));
            m_tmp2=m_tmp_inv*m_K_Y;
            %tmp_inv2 = inv(K_X + rho*eye(N));
            
            
            m_Delta1 = zeros(N); %initialize variables
            m_Delta2 = zeros(N);
            m_B = zeros(N);
            m_Z = zeros(N);
            m_C = zeros(N);
            m_adjacency = zeros(N);
            s_const = s_lambda1/s_rho; %threshold value for soft-thresholding
            
            err1 = zeros(s_maxIter,1); err2 = zeros(s_maxIter,1); err3 = zeros(s_maxIter,1);
            iter = 1; convergence = 0;
            
            while ~convergence && iter < s_maxIter
                
                A_new =m_tmp2+ m_tmp_inv*(  - m_Delta1 + s_rho*m_Z); % update for A
                
                tmp_mat = A_new + m_Delta1./s_rho;
                TODO check if +1 correct?
                J = max(0,abs(tmp_mat) - s_const).*(sign(tmp_mat)+1)/2; %update for Z, soft-thresholding
                
                Z_new = J - diag(diag(J));
                
                m_Delta1 = m_Delta1 + s_rho*(A_new - Z_new); %Auxiliary variable update
                
                
                %convergence checks
                err1(iter) = norm(Z_new-A_new,'fro'); err2(iter) = 0;
                err3(iter) = norm(m_adjacency - A_new,'fro');
                if err1(iter) <= s_tol && err2(iter) <= s_tol && err3(iter) <= s_tol
                    convergence = 1;
                end
                m_adjacency = A_new;  m_Z = Z_new;
                iter = iter+1; %update iteration counter
            end
            
            
            if b_display
                disp(['EnSEM ADMM finished after ',num2str(iter),' iterations']);
                figure;
                hold all;
                plot(err1(1:iter-1));
                plot(err2(1:iter-1));
                plot(err3(1:iter-1));
                legend({'err1','err2','err3'});
            end
            
        end
    end
    
end
