classdef LSVARGraphEstimator< GraphEstimator
	% This was written by Vassilis
	
	
	properties(Constant)
	end
	
	properties % Required by superclass Parameter
		c_parsToPrint    = {};
		c_stringToPrint  = {};
		c_patternToPrint = {};
	end
	%%
	properties
		ch_name = 'LSVARGraphEstimator';
        s_lambda10;
        s_lambda20;
        s_lambda11;
        s_lambda21;
        m_exogenous=-1;
        s_rho;
        s_tol;
        s_maxIter;
        b_display=0;
        m_noisyAdjacency=-1;
        v_initGraphSignal;
	end
	
	
	methods
		
		function obj = LSVARGraphEstimator(varargin)
			obj@GraphEstimator(varargin{:});
		end

	end
	
	methods
		function [m_A0,m_A1]=estimate(obj,m_graphFunction)
            %no exogenous input
            s_lambda10=obj.s_lambda10;
            s_lambda20=obj.s_lambda20;
            s_lambda11=obj.s_lambda11;
            s_lambda21=obj.s_lambda21;
            b_display=obj.b_display;
            s_tol=obj.s_tol;
            s_rho=obj.s_rho;
            s_maxIter=obj.s_maxIter;
            m_graphFunction=m_graphFunction';
            m_graphFunctionPrev=circshift(m_graphFunction,1,1);
            m_graphFunctionPrev(1,:)=obj.v_initGraphSignal';
            m_corrFunction=m_graphFunction'*m_graphFunction;
            m_crossCorrFunctionAndPrev=m_graphFunction'*m_graphFunctionPrev;
            m_crossCorrPrevAndFunction=m_crossCorrFunctionAndPrev';
            m_corrFunctionPrev=m_graphFunctionPrev'*m_graphFunctionPrev;
            
            [T,N] = size(m_graphFunction);
            
            %m_K_Y = m_graphFunction'*m_graphFunction; %inner product matrices
            %K_X = X'*X;
            %K_YX = Y'*X;
            
            m_tmp_inv = inv(m_corrFunction+ (s_lambda20+s_rho)*eye(N));
            m_tmp_invPrev = inv(m_corrFunctionPrev+ (s_lambda21+s_rho)*eye(N));
%             m_tmp2=m_tmp_inv*m_K_Y;
%             m_tmp2=m_tmp_inv*m_K_Y;
            %tmp_inv2 = inv(K_X + rho*eye(N));
            %initialize variables
            m_temp_prod=m_tmp_inv*m_corrFunction;
            m_temp_prod_Prev=m_tmp_invPrev*m_corrFunctionPrev;
            m_temp_cross_prod0=m_tmp_inv*m_crossCorrFunctionAndPrev;
            m_temp_cross_prod1=m_tmp_invPrev*m_crossCorrPrevAndFunction;
            m_S1=zeros(N); 
            m_U0 = zeros(N); 
            m_U1 = zeros(N);
            m_A1 = zeros(N);
            m_S0 = zeros(N);
            m_C = zeros(N);
            m_A0 = zeros(N);
            s_const0 = s_lambda10/s_rho; %threshold value for soft-thresholding
            s_const1 = s_lambda11/s_rho;
            
            err1 = zeros(s_maxIter,1); err2 = zeros(s_maxIter,1); err3 = zeros(s_maxIter,1);
            err4= zeros(s_maxIter,1);
            s_it = 1; b_conv = 0;
            
            while ~b_conv && s_it < s_maxIter
                
                m_A0_new =m_temp_prod+ m_tmp_inv*(  - m_U0 + s_rho*m_S0)-m_temp_cross_prod0*m_A1; % update for A
                m_A1_new =m_temp_prod_Prev+ m_tmp_invPrev*(  - m_U1 + s_rho*m_S1)-m_temp_cross_prod1*m_A0_new; %
                %update for m_S, soft-thresholding
                % why this code had +1 ?
                m_J0 = max(0,abs(m_A0_new + m_U0./s_rho) - s_const0).*(sign(m_A0_new + m_U0./s_rho)...+1
                    )/2; 
                m_J1 = max(0,abs(m_A1_new + m_U1./s_rho) - s_const1).*(sign(m_A1_new + m_U1./s_rho)...+1
                    )/2;
                %wthresh
                %%SOS M_S1 non zero diagonal
                m_S0_new = m_J0 - diag(diag(m_J0));
                m_S1_new = m_J1; %- diag(diag(m_J1));
                %Auxiliary variable update
                m_U0 = m_U0 + s_rho*(m_A0_new - m_S0_new); 
                m_U1 = m_U1 + s_rho*(m_A1_new - m_S1_new); 
                
                %convergence checks 
                err1(s_it) = norm(m_S0_new-m_A0_new,'fro'); err2(s_it) = norm(m_S1_new-m_A1_new,'fro');
                err3(s_it) = norm(m_A0 - m_A0_new,'fro');
                err4(s_it) = norm(m_A1-m_A1_new,'fro');
                if err1(s_it) <= s_tol && err2(s_it) <= s_tol && err3(s_it) <= s_tol && err4(s_it)L<=s_tol
                    b_conv = 1;
                end
                m_A0 = m_A0_new;  m_S0 = m_S0_new;
                m_A1 = m_A1_new;  m_S1 = m_S1_new;
                s_it = s_it+1; %update iteration counter
            end
            
            
            if b_display
                disp(['EnSEM ADMM finished after ',num2str(s_it),' iterations']);
                figure;
                hold all;
                plot(err1(1:s_it-1));
                plot(err2(1:s_it-1));
                plot(err3(1:s_it-1));
                plot(err4(1:s_it-1));
                legend({'err1','err2','err3','err4'});
            end
            
        end
    end
    
end
