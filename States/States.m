classdef States
    enumeration
        FORWARD
        BACKWARD
        UPDATE
        LOSS
        DERIVATIVE
    end
    
    methods (Static)
        function check_state(state, desired_state)
            if (state ~= desired_state)
                error(['Current state: ' state.char '; Required state: ' desired_state.char])
            end
        end
    end
end