classdef (Abstract) Loss < matlab.mixin.Heterogeneous & handle
    properties
        X
        Y
        L
        dL_dX
        state
    end
    
    methods
        compute_loss(self, X, Y)
        differentiate_loss(self)
    end
end