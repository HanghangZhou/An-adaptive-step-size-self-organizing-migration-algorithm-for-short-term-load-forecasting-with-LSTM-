function t_test()  
    %% DE
    load("ASOMA.mat"); ASOMA = TestFitness;
    load("PSO.mat"); data1 = TestFitness;
    H_Result=[];
    P_Result=[];
    for i = 1:15
        [h,p] = ttest(ASOMA(i,:)',data1(i,:)');
        H_Result = [H_Result;h];
        P_Result = [P_Result;p];
    end
    temp = [P_Result, H_Result];
    xlswrite("ASOMA-PSO-results.xls",temp);
    %% JDE
    load("ASOMA.mat"); ASOMA = TestFitness;
    load("CPSO.mat"); data2 = TestFitness;
    H_Result=[];
    P_Result=[];
    for i = 1:15
        [h,p] = ttest(ASOMA(i,:)',data2(i,:)');
        H_Result = [H_Result;h];
        P_Result = [P_Result;p];
    end
    temp = [P_Result, H_Result];
    xlswrite("ASOMA_CPSO-ttest-results.xls",temp);
    %% CODE
    load("ASOMA.mat"); ASOMA = TestFitness;
    load("SOMA.mat"); data3 = TestFitness;
    H_Result=[];
    P_Result=[];
    for i = 1:15
        [h,p] = ttest(ASOMA(i,:)',data3(i,:)');
        H_Result = [H_Result;h];
        P_Result = [P_Result;p];
    end
    temp = [P_Result, H_Result];
    xlswrite("ASOMA_SOMA-ttest-results.xls",temp);
    %% EPSDE
    load("ASOMA.mat"); ASOMA = TestFitness;
    load("LSOMA.mat"); data4 = TestFitness;
    H_Result=[];
    P_Result=[];
    for i = 1:15
        [h,p] = ttest(ASOMA(i,:)',data4(i,:)');
        H_Result = [H_Result;h];
        P_Result = [P_Result;p];
    end
    temp = [P_Result, H_Result];
    xlswrite("ASOMA_LSOMA-ttest-results.xls",temp);
    %% SADE
    load("ASOMA.mat"); ASOMA = TestFitness;
    load("OSOMA.mat"); data5 = TestFitness;
    H_Result=[];
    P_Result=[];
    for i = 1:15
        [h,p] = ttest(ASOMA(i,:)',data5(i,:)');
        H_Result = [H_Result;h];
        P_Result = [P_Result;p];
    end
    temp = [P_Result, H_Result];
    xlswrite("ASOMA_OSOMA-ttest-results.xls",temp);
    %% JADE
    load("ASOMA.mat"); ASOMA = TestFitness;
    load("HBSOMA.mat"); data6 = TestFitness;
    H_Result=[];
    P_Result=[];
    for i = 1:15
        [h,p] = ttest(ASOMA(i,:)',data6(i,:)');
        H_Result = [H_Result;h];
        P_Result = [P_Result;p];
    end
    temp = [P_Result, H_Result];
    xlswrite("ASOMA_HBSOMA-ttest-results.xls",temp);
%     %% RLDE
%     load("ASOMA.mat"); ASOMA = TestFitness;
%     load("RLDE.mat"); data6 = TestFitness;
%     H_Result=[];
%     P_Result=[];
%     for i = 1:15
%         [h,p] = ttest(ASOMA(i,:)',data6(i,:)');
%         H_Result = [H_Result;h];
%         P_Result = [P_Result;p];
%     end
%     temp = [P_Result, H_Result];
%     xlswrite("ASOMA_RLDE-ttest-results.xls",temp);
% end

