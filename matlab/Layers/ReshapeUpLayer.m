classdef ReshapeUpLayer < Layer
    properties
        reshaped_size
    end
    
    methods
        function rul = ReshapeUpLayer(reshaped_size)
            rul.reshaped_size = reshaped_size;
            rul.state = States.FORWARD;
        end

        function forward_pass(self, X)
            States.check_state(self.state, States.FORWARD)

            self.X = X;
            self.Y = [X; zeros(self.reshaped_size-size(X, 1), size(X, 2))];

            self.state = States.BACKWARD;
        end

        function backward_pass(self, dL_dY)
            States.check_state(self.state, States.BACKWARD)

            self.dL_dX = dL_dY(1:size(self.X, 1), 1:size(self.X, 2));

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