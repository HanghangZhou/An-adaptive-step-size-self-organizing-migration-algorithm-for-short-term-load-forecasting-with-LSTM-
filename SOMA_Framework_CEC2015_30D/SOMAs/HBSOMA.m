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
D=Dim(problem);%13-16�е���˼�ο�CEP
lu=Boundary(problem,D);
tempTEV=Error(D);
TEV = tempTEV(problem);
FESMAX=D*10000;
RunOptimization=zeros(runmax,D);
for run=1:runmax
    step = 0.21; %step size
    pathLength = 2.1; % path length
    prt = 0.3; % prt �㶯����
    max_iter = 10; %max iterations������
    TimeFlag=0;
    TempFES=FESMAX;
    t1=clock;
    x=Initpop(N,D,lu);%��Ⱥ��ʼ�����ο�CEP
    fitness=benchmark_func(x,problem);%����ÿһ������ĺ���ֵ���ο�CEP
    %%�����쵼��
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
    
    FES=N;%��ǰ�ĺ������۴������������Ѽ���Ĵ���
    k=1;
    while FES<=FESMAX
        for i = 1:N
            
            if i==lead_pos
%                 if lead_pos~=N %��ϵ�������:�����ڡ���ʾ���������˱��ʽ�����ʱ,���Ϊ1��
%                     i=i+1;
%                 else
               continue;
%                 end
            end
            
            strt_pop=x(i,:);
            
            leader = x(lead_pos,:);%����Ⱥ�ĵ�lead_pos��������Ϊ�쵼��
            
            for t=0:step:pathLength
                %% ����prtvector�㶯����
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
                        indexSet=1:N;%����һ��1,2,3��������N������
                        indexSet(i)=[];%ȥ������indexSet��i��Ԫ�أ�����iɾ��
                        temp=floor(rand*(N-1))+1;%��1��N-1֮�����ȡһ������
                        r(1)=indexSet(temp);%���ɲ�ֱ���ĵ�һ������Ǳ�
                        indexSet(temp)=[];%ȥ������indexSet��temp��Ԫ��
                        temp=floor(rand*(N-2))+1;%��1��N-2֮�����ȡһ������
                        r(2)=indexSet(temp);%���ɲ�ֱ���ĵڶ�������Ǳ�
                        v(i,j)=strt_pop(j) + (( x(r(1),(j)) - x(r(2),(j)) ) * t * prt_vector(j))';
                    end
                end
                %% �߽紦��
                for j = 1:D
                    if  v(i,j)>lu(2,j)
                        v(i,j)=max(lu(1,j),2*lu(2,j)-v(i,j));%�����Ͻ紦���ο�CEP
                    end
                    if  v(i,j)<lu(1,j)
                        v(i,j)=min(lu(2,j),2*lu(1,j)-v(i,j));%�����½紦���ο�CEP
                    end
                end
                
                fnew(i)=benchmark_func(v(i,:),problem);%�������ɵ���������newx(i,:)��Ŀ�꺯��ֵ
                if fnew(i) < fitness(i)
                    fitness(i) = fnew(i);   % store better CV and position�洢���õļ�����ְλ
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