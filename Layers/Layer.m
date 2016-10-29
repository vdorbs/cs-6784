classdef (Abstract) Layer < matlab.mixin.Heterogeneous & handle
    properties
        X
        Y
        dL_dX
        state
    end
    
    methods
        forward_pass(self, X)
        backward_pass(self, dL_dY)
        update(self, alpha)
    end
end