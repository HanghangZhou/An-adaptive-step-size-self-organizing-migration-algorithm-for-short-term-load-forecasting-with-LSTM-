function [ox,x]=O_Initpop(N,D,lu)
% x=repmat(lu(1,:),N,1)+rand(N,D).*(repmat(lu(2,:)-lu(1,:),N,1));
for i=1:N
    for j=1:D
        x(i,j)=lu(1,j)+rand*(lu(2,j)-lu(1,j));
    end
end

for i=1:N
    for j=1:D
        ox(i,j)=lu(2,j)+lu(1,j)-x(i,j);
    end
end



    
