classdef ReLULayer < Layer
    methods
        function rll = ReLULayer
           rll.state = States.FORWARD; 
        end
        
        function forward_pass(self, X)
            States.check_state(self.state, States.FORWARD)
            
            self.X = X;
            self.Y = X .* (X > 0);
            
            self.state = States.BACKWARD;
        end
        
        function backward_pass(self, dL_dY)
            States.check_state(self.state, States.BACKWARD)
            
            self.dL_dX = dL_dY .* (self.X > 0);
            
            self.X = [];
            self.Y = [];
            self.state = States.UPDATE;
        end
        
        function update(self, ~)
            States.check_state(self.state, States.UPDATE)
            
            self.state = States.FORWARD;
        end
    end
end