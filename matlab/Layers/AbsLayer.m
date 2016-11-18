classdef AbsLayer < Layer
    methods
        function al = AbsLayer
            al.state = States.FORWARD;
        end
        
        function forward_pass(self, X)
            States.check_state(self.state, States.FORWARD)
            
            self.X = X;
            self.Y = abs(X);
            
            self.state = States.BACKWARD;
        end
        
        function backward_pass(self, dL_dY)
            States.check_state(self.state, States.BACKWARD)
            
            self.dL_dX = dL_dY .* sign(self.X);
            
            self.X = [];
            self.Y = [];
            self.state = States.UPDATE;
        end
        
        function update(self, ~)
            States.check_state(self.state, States.UPDATE)
            
            self.dL_dX = [];
            self.state = States.FORWARD;
        end
    end
end