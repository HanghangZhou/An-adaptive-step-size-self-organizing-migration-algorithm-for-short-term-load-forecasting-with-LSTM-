function x=C_Initpop(N,D,lu)
for j=1:D
    log_set=Logistic_map(N,4);
    for i=1:N
        x(i,j)=log_set(i)*(lu(2,j)-lu(1,j));
    end
end
end
 function log_set=Logistic_map(N,u)
log_set(1)=rand;
for i=2:N
    log_set(i)=u*log_set(i-1)*(1-log_set(i-1));
end 
 end






    
