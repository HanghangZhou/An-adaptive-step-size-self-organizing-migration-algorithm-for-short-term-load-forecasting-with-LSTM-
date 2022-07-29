function y=Error(D)
    if (D == 10)
        y=[1E-6 1E-6 1E-6 1E-6 5.00 1E-6 1E-6 10.0 5.00 0.05 6E+2 1E-4 1E-5 1E-6 1E-6 1E-6 1E-6 1E-6 10.0 2.00 10.0 10.0 0.20 4E+3 4E+3 1E-8 1E-8 1E-8 1E-8 1E-8];      % Dim=10;
    end
    if (D == 30)
        y =[1E-2 1E-5 1E-6 1E+3 20.0 1E-6 1E-6 50.0 50.0 0.05 4E+3 1E-5 1E-6 1E-6 1E-6 1E-6 1E-6 50.0 50.0 20.0 20.0 50.0 0.01 1.3E+4 1.3E+4 1E-8 1E-8 1E-8 1E-8 1E-8];    % Dim=30;
    end
    if (D == 50||D == 100)
        y =[1E-6 0.30 1E-6 1E+3 50.0 1E-6 3E-5 100.0 20.0 0.05 8E+3 1E-5 1E-6 1E-6 1E-6 1E-6 1E-6 3E-4 1E+2 5.00 1E+2 100.0 0.01 2E+4 4E+3 1E-8 1E-8 1E-8 1E-8 1E-8];      % Dim=50;
    end
    if (D == 1000)
        for i=1:D
            y(i) = 1E-3;
        end
    end
end
