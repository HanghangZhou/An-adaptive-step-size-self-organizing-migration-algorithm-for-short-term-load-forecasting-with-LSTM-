function f=benchmark_func(x,func_num)
    bias = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500];
    f = cec15_func(x',func_num) - bias(func_num);
    f = f';
end