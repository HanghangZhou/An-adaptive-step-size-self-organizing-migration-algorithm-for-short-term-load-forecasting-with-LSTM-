 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=CPSO(problem,N,runmax) 
'CPSO'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    global Initial_Flag
    D=Dim(problem);%3-6行的意思参考CEP
    lu=Boundary(problem,D);
    tempTEV=Error(D);
    TEV = tempTEV(problem);
    FESMAX=D*10000;
    RunOptimization=zeros(runmax,D);
    
    PopSize = N;
    DimSize = D;
    xmin    = lu(1,1);
    xmax    = lu(2,1);
%     VTR     = opt.VTR;
%     runmax  = opt.runmax;
%     FESMAX  = opt.maxFES;
    genmax  = round(FESMAX/N);
    refresh = 10000;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cc  = [1.49445 1.49445];%
    ps = PopSize;
    me=FESMAX/ps; %0.9-0.4
    iwt = 0.9 - (1 : me) * (0.4 / me);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mv   = 0.2*(xmax-xmin);
    xmin = repmat(xmin,PopSize,1); xmax = repmat(xmax,PopSize,1);
    Vmin = repmat(-mv,PopSize,1);  Vmax  = -Vmin;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for run=1:runmax
        Initial_Flag = 0; TimeFlag = 0; TempTime = inf; TempFES = inf; t1 = clock;  FES = 0; k = 1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Pop = C_Initpop(N,D,lu);


        for i = 1:PopSize
            PopFit(i,1) = benchmark_func(Pop(i,:),problem);  FES = FES + 1;
        end
        vel = Vmin+2.*Vmax.*rand(PopSize,DimSize);                      % initialize the velocity of the particles

        pBest = Pop;   
        pBestFit = PopFit;                               % initialize the pBest and the pBest's Fitness value

        [gBestFit,gBestid] = min(pBestFit);  
        gBest = pBest(gBestid,:);  % initialize the gBest and the gBest's Fitness value
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Output
        gen = 1;  RunBest(run,gen) = gBestFit;
        if ((gen==1)||(mod(FES,refresh) == 0)||(FES==FESMAX))
            fprintf('EA:%s ObjFun:%d PopSize:%d Run:%d Gen:%d FES:%d Best:%g\n','PSO',problem,PopSize,run,gen,FES,gBestFit);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for gen = 2:genmax
            for i = 1:PopSize 

                vel(i,:) = iwt(gen).*vel(i,:)+cc(1).*rand(1,DimSize).*(pBest(i,:)-Pop(i,:))+cc(2).*rand(1,DimSize).*(gBest-Pop(i,:)); 

                vel(i,:) = (vel(i,:)>mv).*mv+(vel(i,:) <= mv).*vel(i,:);   
                vel(i,:) = (vel(i,:)<(-mv)).*(-mv)+(vel(i,:) >= (-mv)).*vel(i,:);

                Pop(i,:) = Pop(i,:)+vel(i,:); 

                Pop(i,:) =  (Pop(i,:)>xmax(i,:)).*xmax(i,:)+(Pop(i,:)<=xmax(i,:)).*Pop(i,:);
                Pop(i,:) =  (Pop(i,:)<xmin(i,:)).*xmin(i,:)+(Pop(i,:)>=xmin(i,:)).*Pop(i,:);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                PopFit(i,1) = benchmark_func(Pop(i,:),problem);  FES = FES + 1;

                if  PopFit(i,1)<pBestFit(i,1)              
                    pBest(i,:) = Pop(i,:);  
                    pBestFit(i) = PopFit(i); % update the pBest
                end

                if pBestFit(i,1)<gBestFit
                    gBest = pBest(i,:);  
                    gBestFit = pBestFit(i);   % update the gBest
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Output
            RunBest(run,gen) = gBestFit;
            if FES == 10000 * 0.1 || (mod(FES,refresh) == 0)
                    RunValue(run,k)=gBestFit;
                    Para(k,:)=gBest;
                    k=k+1;
                fprintf('EA:%s ObjFun:%d PopSize:%d Run:%d Gen:%d FES:%d Best:%g\n','CPSO',problem,PopSize,run,gen,FES,gBestFit);
            end

            % Store the accepted running time
            if TimeFlag == 0
                if gBestFit <= TEV
                    t2 = clock; t = etime(t2,t1); TempTime = t; TempFES = FES;  TimeFlag = 1;
                end
            end

        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        t3 = clock; t = etime(t3,t1); 
        RunTime(run,1) = t;    RunFES(run) = min(TempFES,FES);  RunResult(run) = gBestFit;
        RunOptimization(run,1:D)=gBest;
        RunParameter=Para;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
