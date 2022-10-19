function optLambda = getOptLamda(X, Y, setPara)
% Get the optimal lamda
%
% INPUTS:
%   X(MxN) : trData(i,j) is the i-th feature from the j-th trial
%   Y(Nx1): trData(j) is the label of the j-th trial (1 or -1)
%   setPara : Initialized parameters
%            setPara.t      
%            setPara.beta   
%            setPara.Tmax   
%            setPara.tol    
%            setPara.W      
%            setPara.C      
%
% OUTPUTS:
%   optiLamda: Optimal lamda value 

beta = setPara.beta;
Tmax = setPara.Tmax;
tol = setPara.tol;
W = setPara.W;
C = setPara.C;
best = 0;
optLambda = 0;
for lambda = [0.01 1 30 100 300 1000 3000 10000]
    correct_num = zeros(5,1);
    for j=0:4
        t = setPara.t;
        [curr_train, curr_test] = devide_set(X(:,1:100), X(:,101:200), j, 5);
        train_y = [ones(80,1);ones(80,1)*-1];
        test_y = [ones(20,1);ones(20,1)*-1];
        zeta = 1.001*ones(160,1);
        init_Z.W = W; init_Z.C = C; init_Z.zeta = zeta;
        while (t <= Tmax)
            [optSolution, err] = solveOptProb_NM(@costFcn,init_Z,tol, curr_train, train_y, lambda,t);
            init_Z = optSolution;
            t = t*beta;
        end
        opt_W = optSolution.W;
        opt_C = optSolution.C;
        est_y = opt_W'*curr_test+opt_C;
        correct_num(j+1) = sum(est_y(1:20)>0)+sum(est_y(21:40)<0);        
    end
    average_correct = mean(correct_num);
    if (average_correct>best)
        best = average_correct;
        optLambda = lambda;
    end
end
