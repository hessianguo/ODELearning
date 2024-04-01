function f = genforce(W,customconv,d,comb)
    if d ==1
        f = @(x,t) 0*x;
        if ~isempty(customconv)
            for i=1:length(W)
                if W(i)~=0
                    f = @(x,t) f(x,t) + W(i)*customconv{i}{2}(x,t);
                end
            end
        end
        f = {f};
    elseif d==2
        fx = @(x,y,t) 0*x+0*y;
        fy = @(x,y,t) 0*x+0*y;
        if ~isempty(customconv)
            for i=1:length(W)
                if ~comb
                    if W(i)~=0
                        if mod(i,2)==1
                            fx = @(x,y,t) fx(x,y,t) + W(i)*customconv{i}{2}(x,y,t);
                        else
                            fy = @(x,y,t) fy(x,y,t) + W(i)*customconv{i}{2}(x,y,t);
                        end
                    end
                else
                    if W(i)~=0
                        fx = @(x,y,t) fx(x,y,t) + W(i)*customconv{2*i-1}{2}(x,y,t);
                        fy = @(x,y,t) fy(x,y,t) + W(i)*customconv{2*i}{2}(x,y,t);
                    end
                end
            end
        end
        f = {fx,fy};
    end
end
           
