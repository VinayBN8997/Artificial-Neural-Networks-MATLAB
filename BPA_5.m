function BPA(X, d, hidden_no)
    [a,b] = size(X);
    [m,n] = size(d);

    W0 = rand(a,hidden_no); % Weight matrix from Input to Hidden
    W1 = rand(hidden_no,m); % Weight matrix from Hidden to Output
    delta_W0 = zeros(size(W0));
    delta_W1 = zeros(size(W1));

    count = 0;
    [e delta_W0 delta_W1] = trainNN(X,d,W0,W1,delta_W0,delta_W1);

    while e > 0.001
        count = count + 1;
        Error(count)=e;
        W0=W0+delta_W0;
        W1=W1+delta_W1;
        [e delta_W0 delta_W1]=trainNN(X,d,W0,W1,delta_W0,delta_W1);
    end
    plot([1:count],Error);
    title("Error vs iteration count");

    disp("Weight matrix from Input to Hidden");
    disp(W0);
    disp("Weight matrix from Hidden to Output");
    disp(W1);
    
    VK0 = W0'*X;
    Y0 = 1./(1+exp(-VK0));
    VK1 = W1'*Y0;
    Y1 = 1./(1+exp(-VK1));
    
    disp("Output");
    disp(Y1);

    
end

