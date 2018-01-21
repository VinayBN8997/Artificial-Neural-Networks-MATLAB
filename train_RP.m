function w = train_RP(x,d)
    [rows,cols] = size(x);
    x = [ones(rows,1) x];
    neta = 0.1;
    w = [1 zeros(1,cols)];
    E = 0;
    while(1)
        for i =1:rows
            y = calc_RP(w,x(i,:));
            w = w + neta*(d(i)-y)*x(i,:);
            E = E +((d(i) - y)^2)/2;
        end
        if(E == 0)
            break;
        else
            E=0;
        end
    end
end
    
    