classdef DizzyLayer < Layer
    
    properties
        W
        dL_dW
        alpha_scale
    end
    
    methods
        function dzyl = DizzyLayer(n, alpha_scale)
            dzyl.W = 2 * pi * rand(1,n*(n-1)/2);
            dzyl.dL_dW = zeros(1, n*(n-1)/2);
            dzyl.alpha_scale = alpha_scale;
            dzyl.state = States.FORWARD;
        end
        
        function forward_pass(self, X)
            States.check_state(self.state, States.FORWARD)
            
            Y = X;
            n = size(X,1);
            idx = 1;
            for i = 1:n-1
               for j = i+1:n
                   t = self.W(idx);
                   idx = idx + 1;
                   R = [cos(t), sin(t); ...
                       -sin(t), cos(t)];
                   Rows = [Y(i,:); ...
                           Y(j,:)];
                   Rows = R*Rows;
                   Y(i,:) = Rows(1,:);
                   Y(j,:) = Rows(2,:);
               end
            end
            self.Y = Y;
            
            self.state = States.BACKWARD;
        end
        
        function backward_pass(self, dL_dY)
            States.check_state(self.state, States.BACKWARD)
            
            [n,m] = size(dL_dY);
            x = self.Y;
            self.dL_dX = dL_dY;
            
            idx = size(self.W,2);
            for i = n-1:-1:1
                for j = n:-1:i+1
                    
                    t = self.W(idx);
                    
                    R = [cos(t), sin(t); ...
                        -sin(t), cos(t)];
                    
                    dR_dt = [-sin(t),  cos(t); ...
                             -cos(t), -sin(t)];
                    
                    x_Rows = [x(i,:); ...
                              x(j,:)];
                    
                    dL_dX_Rows = [self.dL_dX(i,:); ...
                                  self.dL_dX(j,:)];
                    dL_dt = dR_dt * R' * x_Rows;
                    self.dL_dW(idx) = sum(sum(dL_dX_Rows .* dL_dt))/m;
                    
                    dL_dX_Rows = R'*dL_dX_Rows;
                    self.dL_dX(i,:) = dL_dX_Rows(1,:);
                    self.dL_dX(j,:) = dL_dX_Rows(2,:);
                    
                    x_Rows = R' * x_Rows;
                    x(i,:) = x_Rows(1,:);
                    x(j,:) = x_Rows(2,:);
                    
                    idx = idx - 1;
                    
                end
            end
            
            self.X = [];
            self.Y = [];
            self.state = States.UPDATE;
        end
        
        function update(self, alpha)
            States.check_state(self.state, States.UPDATE)
            
            self.W = self.W - alpha * self.alpha_scale * self.dL_dW;
            
            self.dL_dW = [];
            self.dL_dX = [];
            self.state = States.FORWARD;
        end
    end
end

