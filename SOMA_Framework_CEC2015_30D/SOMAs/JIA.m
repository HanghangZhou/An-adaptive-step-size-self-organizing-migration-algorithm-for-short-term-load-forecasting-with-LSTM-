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
function [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=JIA(problem,N,runmax)
'JIA'
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
%     Count = ceil(pathLength/step);
    Goodstep=[0,step];
    c=0.5;
    ustep=0.1;
    
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
%         original_list = randperm(N);
%         spiral_list = original_list(1:50);
        for i = 1:N
            
            strt_pop=x(i,:);
            leader = x(lead_pos,:);%将种群的第lead_pos个个体作为领导者              
            
            if i==lead_pos
               continue;
            end
            
            t = 0;
            while t<pathLength
                t=t+step;
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
%                 if ismember(i, spiral_list) == 0
                for j=1:D
                    v(i,j) = strt_pop(j) + ( leader(j) - strt_pop(j) ) * t * prt_vector(j);
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
                FES = FES+1;
                if fnew(i) < fitness(i)
                    fitness(i) = fnew(i);   % store better CV and position存储更好的简历和职位
                    x(i,:) = v(i,:);
                    Goodstep = [Goodstep,step];
                end

                num=size(Goodstep,2);
                good_ustep=sum(Goodstep)/(num-1);
                ustep = (1-c)*ustep+c*good_ustep;
                step=abs(normrnd(ustep,0.01));

                if FES==10000*0.1||mod(FES,10000)==0
                    [kk,ll]=min(fitness);
                    RunValue(run,k)=kk;
                    Para(k,:)=x(ll,:);
                    k=k+1;
                    fprintf('Algorithm:%s problemIndex:%d Run:%d FES:%d Best:%g\n','JIA',problem,run,FES,kk);
                end
                if TimeFlag==0
                    if min(fitness)<=TEV
                        TempFES=FES;
                        TimeFlag=1;
                    end
                end
            end
        end      
        mini = inf;
        for i = 1:N
            if fitness(i)<mini
                mini = fitness(i);
                lead_pos = i;
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