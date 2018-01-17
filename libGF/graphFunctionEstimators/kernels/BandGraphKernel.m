classdef BandGraphKernel < GraphKernel
    %
    properties(Constant)
    end
    
    properties % Required by superclass Parameter
        c_parsToPrint    = {};
        c_stringToPrint  = {};
        c_patternToPrint = {};
    end
    
    properties
        ch_name = 'Band Kernel';
        m_laplacian; %the laplacian associated with the specific Kernel
        s_band1;     %The sigma parameter of the difussion process
        s_band2;
        s_beta;
    end
    
    methods
        
        function obj = BandGraphKernel(varargin)
            obj@GraphKernel(varargin{:});
            obj.m_kernels=obj.generateKernelMatrix(); %the kernel is generated when the object is constructed
        end
        
        
        
    end
    
    methods
        function m_kernels=generateKernelMatrix(obj)
            [m_eigenvectors,m_eigenvalues] = eig(obj.m_laplacian);%,size(obj.m_laplacian,1));
            v_eigenvalues=diag(m_eigenvalues);
            v_eigenvalues(v_eigenvalues == 0) = eps;
            s_band1=obj.s_band1;
            s_band2=obj.s_band2;
            s_beta=obj.s_beta;
            v_eigenvalues=s_beta*ones(size(v_eigenvalues));
            v_eigenvalues(1:s_band1)=1/s_beta;
            v_eigenvalues(end-s_band2:end)=1/s_beta;
            %m_kernels=m_eigenvectors*diag(1./((obj.s_sigma^2*(v_eigenvalues).^2)))*m_eigenvectors';
            m_kernels=m_eigenvectors*diag(1./v_eigenvalues)*m_eigenvectors';

        end

    end
    
end
