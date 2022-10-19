tic;
clear;
Elmg = load('feaSubEOvert.mat'); %% 'feaSubEImg.mat'
Elmg1 = Elmg.class{1,1};
Elmg2 = Elmg.class{1,2};
tol = 0.000001;
Tmax = 1000000;
correct_num = zeros(6,1);
lambda_list = zeros(6,1);
for i = 0:5 %%%%%
    % 1st class has y of 1, 2nd class has y of -1
    [train_set, test_set] = devide_set(Elmg1, Elmg2, i, 6);
    train_y = [ones(100,1);ones(100,1)*-1];
    test_y = [ones(20,1);ones(20,1)*-1];
    %%%%
    setPara.t = 1000; setPara.beta = 15; setPara.Tmax=1000000;   
    setPara.tol = 0.000001; setPara.W = zeros(204,1); setPara.C = 0; 
    optLambda = getOptLamda(train_set, train_y, setPara); 
    lambda_list(i+1)=optLambda;
    init_Z.W = zeros(204,1);
    init_Z.C = 0;
    init_Z.zeta = 1.001*ones(200,1);
    %%%%
    t=1000;
    beta = 15;
    while (t <= Tmax)
        [opt, err] = solveOptProb_NM(@costFcn,init_Z,tol, train_set, train_y, optLambda,t);
        %disp(["t:" num2str(t)]);
        init_Z = opt;
        t=t*beta;
    end
    est_y = opt.W'*test_set + opt.C;
    correct_num(i+1) = sum(est_y(1:20)>0)+sum(est_y(21:40)<0); 
end
Ac = correct_num/40;
meanAc = mean(Ac);
stdAc = std(Ac);
toc;