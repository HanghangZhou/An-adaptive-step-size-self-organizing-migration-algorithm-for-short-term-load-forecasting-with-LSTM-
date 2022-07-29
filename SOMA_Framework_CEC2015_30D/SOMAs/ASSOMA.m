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
function [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=ASSOMA(problem,N,runmax)
'ASSOMA'
D=Dim(problem);%13-16�е���˼�ο�CEP
lu=Boundary(problem,D);
tempTEV=Error(D);
TEV = tempTEV(problem);
FESMAX=D*10000;
RunOptimization=zeros(runmax,D);
for run=1:runmax
%     step = 0.21; %step size
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
%     Count = ceil(pathLength/step);
    Goodstep=[0.21];
    c=0.5;
    ustep=0.1;
    step=[];
    
    step_list=[];
    step_rank=[];
    decrease_list=[];
    rank_list=[];
    
    mini = inf;
    for i = 1:N
        if fitness(i)<mini
            mini = fitness(i);
            lead_pos = i;
        end
    end
    
    fitness_record = fitness;
    
    FES=N;%��ǰ�ĺ������۴������������Ѽ���Ĵ���
    k=1;
    iteration = 1;
    while FES<=FESMAX
        
        num=size(Goodstep,2);
        good_ustep=sum(Goodstep)/num;
        ustep = (1-c)*ustep+c*good_ustep;
        for i =1:N
            temp_step = abs(normrnd(ustep,0.01));
            step_list = [step_list,temp_step];
        end
        step_rank = sort(step_list);         
   
        if length(Goodstep)==1
            for i = 1:N
                step(i) = step_list(i);
            end
        else
            for i = 1:N
                index = rank_list(i);
                step(i) = step_rank(index);
            end
        end
        
        Goodstep=[0.21];
        step_list=[];
        step_rank=[];
        decrease_list=[];
        rank_list=[];
        fitness_record=[];
        
        fitness_record = [fitness];
        for i = 1:N
            strt_pop=x(i,:);
            leader = x(lead_pos,:);%����Ⱥ�ĵ�lead_pos��������Ϊ�쵼�� 
            
            if i==lead_pos
               continue;
            end
            
            t = 0;
            while t<pathLength

                t=t+step(i);
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
%                 if ismember(i, spiral_list) == 0
                for j=1:D
                    v(i,j) = strt_pop(j) + ( leader(j) - strt_pop(j) ) * t * prt_vector(j);
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
                FES = FES+1;
                if fnew(i) < fitness(i)
                    fitness(i) = fnew(i);   % store better CV and position�洢���õļ�����ְλ
                    x(i,:) = v(i,:);
                    Goodstep = [Goodstep,step(i)];
                end

                if FES==10000*0.1||mod(FES,10000)==0
                    [kk,ll]=min(fitness);
                    RunValue(run,k)=kk;
                    Para(k,:)=x(ll,:);
                    k=k+1;
                    fprintf('Algorithm:%s problemIndex:%d Run:%d FES:%d Best:%g\n','ASSOMA',problem,run,FES,kk);
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
        fitness_record = [fitness_record,fitness];
        for i =1:N
            decrease = fitness_record(i,2) - fitness_record(i,1);
            decrease_list = [decrease_list, decrease];
        end
        sort_list = sort(decrease_list,'descend');
        for i = 1:N
            sort_index = find(sort_list==decrease_list(i));
            rank_list = [rank_list, sort_index];
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