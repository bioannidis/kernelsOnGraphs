classdef DiffusionGraphKernel < GraphKernel
    %
    properties(Constant)
    end
    
    properties % Required by superclass Parameter
        c_parsToPrint    = {};
        c_stringToPrint  = {};
        c_patternToPrint = {};
    end
    
    properties
        ch_name = 'Diffusion Kernel';
        m_laplacian; %the laplacian associated with the specific Kernel
        s_sigma;     %The sigma parameter of the difussion process
    end
    
    methods
        
        function obj = DiffusionGraphKernel(varargin)
            obj@GraphKernel(varargin{:});
            obj.m_kernels=obj.generateKernelMatrix(); %the kernel is generated when the object is constructed
        end
        
        
        
    end
    
    methods
        function m_kernels=generateKernelMatrix(obj)
            [m_eigenvectors,m_eigenvalues] = eig(obj.m_laplacian);%,size(obj.m_laplacian,1));
            v_eigenvalues=diag(m_eigenvalues);
            v_eigenvalues(v_eigenvalues == 0) = eps;
            m_kernels=m_eigenvectors*diag(1./(exp(obj.s_sigma^2*v_eigenvalues/2)))*m_eigenvectors';
            
        end
        function m_kernels=generateKernelMatrixFromNormLaplacian(obj)
            graph=Graph('m_adjacency',Graph.createAdjacencyFromLaplacian(obj.m_laplacian));
            m_normLaplacian=graph.getNormalizedLaplacian();
            [m_eigenvectors,m_eigenvalues] = eig(m_normLaplacian);%,size(obj.m_laplacian,1));
            v_eigenvalues=diag(m_eigenvalues);
            v_eigenvalues(v_eigenvalues == 0) = eps;
            m_kernels=m_eigenvectors*diag(1./(exp(obj.s_sigma^2*v_eigenvalues/2)))*m_eigenvectors';
        
        end
        function m_kernels=earlyStopEigVecGenerateKernelMatrix(obj,s_val)
            
            %% different method
            OPTS.tol=s_val;
            
            [m_eigenvectors,m_eigenvalues] = eigs(obj.m_laplacian,size(obj.m_laplacian,1),'sm' ,OPTS);
            
            v_eigenvalues=diag(m_eigenvalues);
            v_eigenvalues(v_eigenvalues == 0) = eps;
            m_kernels=m_eigenvectors*diag(1./(exp(obj.s_sigma^2*v_eigenvalues/2)))*m_eigenvectors';
            
        end
        
        function m_kernel = getNewKernelMatrix(obj,graph)
            obj.m_laplacian = graph.getLaplacian();
            %obj.m_laplacian = graph.getNormalizedLaplacian();
            m_kernel = obj.generateKernelMatrix();
        end
        
    end
    
end
