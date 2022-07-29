%% The original SOMA
%problem: the serial number of testing function recorded in "Public\benchmark_func.m"
%N: the population size
%runmax: the number of the algorithm runs
%RunResult: the  optimal value produced by each algorithm runs
%RunOptimization: the optimal value produced by reach algorithm runs
%RunValue: the fitness of optimal value produced by each 10000 FES
%RunParameter:the optimal value produced by each 10000 FES
%RunTime: the time spent by each algorithm runs
%RunFES: the FES required to satisfy the conditions
function [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=SOMA(problem,N,runmax)
'HBSOMA'
D=Dim(problem);%13-16行的意思参考CEP
lu=Boundary(problem,D);
tempTEV=Error(D);
TEV = tempTEV(problem);
FESMAX=D*10000;
RunOptimization=zeros(runmax,D);
for run=1:runmax
    step = 0.21; %step size
    pathLength = 2.1; % path length
    prt = 0.3; % prt 摄动参数
    max_iter = 10; %max iterations最大迭代
    TimeFlag=0;
    TempFES=FESMAX;
    t1=clock;
    x=Initpop(N,D,lu);%种群初始化，参考CEP
    fitness=benchmark_func(x,problem);%计算每一个个体的函数值，参考CEP
    %%挑出领导者
    lead_pos = 1;
    Count = ceil(pathLength/step);
    
    mini = inf;
    for i = 1:N
        if fitness(i)<mini
            mini = fitness(i);
            lead_pos = i;
        end
    end
    
    %     min_cost= [zeros(max_iter,1)];
    
    FES=N;%当前的函数评价次数，即函数已计算的次数
    k=1;
    while FES<=FESMAX
        for i = 1:N
            
            if i==lead_pos
%                 if lead_pos~=N %关系运算符号:不等于。表示当左右两端表达式不相等时,结果为1。
%                     i=i+1;
%                 else
               continue;
%                 end
            end
            
            strt_pop=x(i,:);
            
            leader = x(lead_pos,:);%将种群的第lead_pos个个体作为领导者
            
            for t=0:step:pathLength
                %% 生成prtvector摄动向量
                prtntzero= true;
                while(prtntzero)
                    for j=1:D
                        if rand<prt
                            prt_vector(j) = 1;
                            prtntzero = false;
                        else
                            prt_vector(j) = 0;
                        end
                    end
                end
                %%
                epsilon = rand;
                if epsilon<rand
                    for j=1:D
                        v(i,j) = strt_pop(j) + ( leader(j) - strt_pop(j) ) * t * prt_vector(j);
                    end
                else
                    for j=1:D
                        indexSet=1:N;%生成一个1,2,3，。。。N的序列
                        indexSet(i)=[];%去掉向量indexSet第i个元素，即把i删除
                        temp=floor(rand*(N-1))+1;%在1到N-1之间随机取一个整数
                        r(1)=indexSet(temp);%生成差分变异的第一个个体角标
                        indexSet(temp)=[];%去掉向量indexSet第temp个元素
                        temp=floor(rand*(N-2))+1;%在1到N-2之间随机取一个整数
                        r(2)=indexSet(temp);%生成差分变异的第二个个体角标
                        v(i,j)=strt_pop(j) + (( x(r(1),(j)) - x(r(2),(j)) ) * t * prt_vector(j))';
                    end
                end
                %% 边界处理
                for j = 1:D
                    if  v(i,j)>lu(2,j)
                        v(i,j)=max(lu(1,j),2*lu(2,j)-v(i,j));%超出上界处理，参考CEP
                    end
                    if  v(i,j)<lu(1,j)
                        v(i,j)=min(lu(2,j),2*lu(1,j)-v(i,j));%超出下界处理，参考CEP
                    end
                end
                
                fnew(i)=benchmark_func(v(i,:),problem);%计算生成的试验向量newx(i,:)的目标函数值
                if fnew(i) < fitness(i)
                    fitness(i) = fnew(i);   % store better CV and position存储更好的简历和职位
                    x(i,:) = v(i,:);
                end
                step=0.1+0.99*(step-0.1);
            end
        end
        
        mini = inf;
        for i = 1:N
            if fitness(i)<mini
                mini = fitness(i);
                lead_pos = i;
            end
        end
        
        for i=1:N
            FES=FES+Count;
            if FES==10000*0.1||mod(FES,10000)==0
                [kk,ll]=min(fitness);
                RunValue(run,k)=kk;
                Para(k,:)=x(ll,:);
                k=k+1;
                fprintf('Algorithm:%s problemIndex:%d Run:%d FES:%d Best:%g\n','HBSOMA',problem,run,FES,kk);
            end
            if TimeFlag==0
                if min(fitness)<=TEV
                    TempFES=FES;
                    TimeFlag=1;
                end
            end
        end
        
    end
    % [kk,ll]=min(benchmark_func(x,problem));
    

    [kk,ll]=min(fitness);
    gbest=x(ll,:);
    t2=clock;
    RunTime(run)=etime(t2,t1);
    RunResult(run)=kk;
    RunFES(run)=TempFES;
    RunOptimization(run,1:D)=gbest;
    RunParameter{run}=Para;
end
end