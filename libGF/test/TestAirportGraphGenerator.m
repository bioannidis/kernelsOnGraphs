classdef TestAirportGraphGenerator < matlab.unittest.TestCase
    
    properties
        filepath = 'testdatafile.txt';
    end
    
    methods(Test)
        function testFilepath(tc)
            tc.verifyTrue(strcmp(tc.filepath,'testdatafile.txt'));
        end
        % NaN values
        function testNanValues(tc)
            apg = AirportGraphGenerator();
            apg.ch_filepath = tc.filepath;
            graph = apg.realization();
        end
    end
    
end

