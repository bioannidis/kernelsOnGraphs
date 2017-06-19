classdef TestExpDecayFG < matlab.unittest.TestCase
    
    properties
        expFG
        N = 100;
    end
    
    methods(TestMethodSetup)
        function createExpFG(tc)
            N = tc.N;
            A = randn(N,N);
            A = double( ( A + A' ) > 0.5 );
            graph = Graph('m_adjacency',A);
            BW = floor(N/2);
            tc.expFG = ExponentiallyDecayingFunctioinGenerator('graph',graph, ...
                's_bandwidth', BW);
        end
    end
    
    methods(Test)
        function testOutputSize(tc)
            nRealization = randi(10);
            x = tc.expFG.realization(nRealization);
            tc.verifySize(x, [tc.N nRealization]);
        end
        
        function testMean(tc)
            x = tc.expFG.realization(1);
            tc.verifyLessThan( mean( x(1:floor(tc.N/2)) ), 1 );
        end
        
        function testVariance(tc)
            x = tc.expFG.realization(1);
            L = tc.expFG.graph.getLaplacian();
            [V,~] = eig(L);
            xhat = V'*x;
            xx = xhat(1:floor(tc.N/2));
            tc.verifyLessThan( var(xx), 0.3 );
        end
        
        function testExpDecaying(tc)
            x = tc.expFG.realization(1);
            
            L = tc.expFG.graph.getLaplacian();
            [V,D] = eig(L);
            xhat = V'*x;
            
            h = stem(xhat);
            
            tc.verifyNotEmpty(h);
        end
    end
    
end

