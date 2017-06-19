classdef TestMKLEstimator < matlab.unittest.TestCase  
    % test multi-kernel estimator
    
    % Test to be performed
    % 1. illegal parameter: kernel or mu is not set
    % 2. illegal paramter: sample and position not consistent
    % 3. illegal paramter: position out of range
    % 4. test output
    % 5. vector handling  
    
    properties
        mkrEstimator
        graph
        bandwidth = 5;
        N = 10;     % number of vertices
        p = 0.3;    % edge probability
        graphSignal
        samples
        positions
    end
    
    methods(TestMethodSetup)
        function createGraph(tc)
            ERGraphGenerator = ErdosRenyiGraphGenerator();
            ERGraphGenerator.s_numberOfVertices = tc.N;
            ERGraphGenerator.s_edgeProbability = tc.p;
            tc.graph = ERGraphGenerator.realization();
        end
        
        function createGraphSignal(tc)
            V = tc.graph.getLaplacianEigenvectors();
            alpha = zeros(tc.N,1);
            alpha(1:tc.bandwidth) = sort(rand(tc.bandwidth,1));
            tc.graphSignal = V*alpha;
        end
              
        function createMkrGraphFunctionEstimator(tc)
            tc.mkrEstimator = MkrGraphFunctionEstimator();
        end
        function createSampleAndPosition(tc)
            S = floor(tc.N*2/3);
            tc.positions = randi(tc.N,S,1);
            tc.samples = tc.graphSignal(tc.positions);
        end
    end
    
    methods(Test)
        % trivial tests
        function testGraphNotEmpty(tc)
            tc.verifyNotEmpty(tc.graph);
        end
        function testGraphSignalNotEmpty(tc)
            tc.verifyNotEmpty(tc.graphSignal);
        end
        % test kernel or mu not set
        function testParameterNotSet(tc)
            tc.verifyError(@() tc.mkrEstimator.estimate(tc.samples, ...
                tc.positions), 'MkrGraphFunctionEstimator:notEnoughInfo');
        end
        % test sample and position not consistent
        function testParameterInconsistent(tc)
            tc.samples(end) = [];
            tc.mkrEstimator.s_mu = 1e-5;
            tc.mkrEstimator.m_kernel = magic(tc.N);
            tc.verifyError(@() tc.mkrEstimator.estimate(tc.samples,...
                tc.positions), ...
                'MkrGraphFunctionEstimator:inconsistentParameter');
        end
        % test position out of range
        function testPositionOutOfRange(tc)
            tc.positions(1) = tc.N+1;
            tc.mkrEstimator.s_mu = 1e-5;
            tc.mkrEstimator.m_kernel = magic(tc.N);
            tc.verifyError(@() tc.mkrEstimator.estimate(tc.samples, ...
                tc.positions), 'MkrGraphFunctionEsitmator:outOfBound');
        end
    end
    
end

