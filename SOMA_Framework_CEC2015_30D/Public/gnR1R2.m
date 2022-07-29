function [r1, r2] = gnR1R2(NP1, NP2, r0)

% gnA1A2 generate two column vectors r1 and r2 of size NP1 & NP2, respectively
%    r1's elements are choosen from {1, 2, ..., NP1} & r1(i) ~= r0(i)
%    r2's elements are choosen from {1, 2, ..., NP2} & r2(i) ~= r1(i) & r2(i) ~= r0(i)
%
% Call:
%    [r1 r2 ...] = gnA1A2(NP1)   % r0 is set to be (1:NP1)'
%    [r1 r2 ...] = gnA1A2(NP1, r0) % r0 should be of length NP1
%
% Version: 1  Date: 2020/08/11
% Written by zhenyu wang (tsingke123@163.com)

    NP0 = length(r0);
    for i = 1:NP0
        temp = randperm(NP2);
        r2(i) = temp(1);
        k=2;
        while temp(k) > NP1
            k = k+1;
        end
        r1(i) = temp(k);
        if(r2(i) == r0(i))
            r2(i) = temp(k+1);
        end
        if(r1(i) == r0(i))
            k=k+1;
            while temp(k) > NP1
            k = k+1;
            end
            r1(i) = temp(k);
        end
    end       
end