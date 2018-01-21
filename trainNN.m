function [e delta_W0 delta_W1] = trainNN(X, d, W0, W1, delta_W0, delta_W1)
    etta=0.6;alpha=1;
    VK0 = W0'*X;
    Y0 = 1./(1+exp(-VK0));
    VK1 = W1'*Y0;
    Y1 = 1./(1+exp(-VK1));
    error = d - Y1;
    sq = error.*error;
    e = sqrt(sum(sq(:)));
    del = error.*Y1.*(1-Y1);
    delta_W1= alpha.*delta_W1 + etta.*(Y0*del'); 
    d_star = (W1*del).*Y0.*(1-Y0);
    delta_W0=alpha*delta_W0 + etta*X*d_star';
end