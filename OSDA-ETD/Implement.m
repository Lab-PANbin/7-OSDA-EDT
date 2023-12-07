
clear;close all;

% The files are the MATLAB source code for the papers
% Jiao Liu et al. "An Open Set Domain Adaptation Algorithm via Exploring Transferability and Discriminability  for Remote Sensing Image Scene Classification", 
% IEEE Transactions on Geoscience and Remote Sensing, 2021. 
% The code has not been well organized. Please contact me if you meet any problems.
% Email: liujiao.hebut@hotmail.com


% Set algorithm parameters
options.p = 10;      %%%%manifold neighbour
   
options.rho = 1;     %%%%regularization
    
options.lambda = 50; %%%%%transferability  regularization

options.mu=1;  %%%%%mainfold regularization
   
options.T = 10;      %%%%iterations%%%%%
    
options.gamma = 0.4;%%%%%%open set parameters
    
options.sigma = 0.35; %%%%%%open set parameters

options.eta = 1.5;%discriminability  regularization

results=[];     
source = {'ucm'};
target = {'nwpu'};
src = char(source);
tgt = char(target);
options.data = strcat(src,'-vs-',tgt);
fprintf('Data=%s \n',options.data);

 % load  data  
load(['E:\matlab_liujiaocode\ours\resnet\feature_source_ucm\' src '.mat']);
Xs = fts; 
Ys = labels;

load(['E:\matlab_liujiaocode\ours\resnet\feature_source_ucm\' tgt '.mat']);
Xt = fts; 
Yt = labels;

%choice known classes and unknown classes
[Xs,Xt,Ys,Yt]=datachoice(Xs,Xt,Ys,Yt,9,9);
  
[OS,OS_star,ALL,ALL_star,CA] = OSDA_ETD(Xs,Ys,Xt,Yt,options);

disp('OS');  disp(OS);
disp('OS*'); disp(OS_star);
disp('ALL');  disp(ALL)
disp('ALL*'); disp(ALL_star);
disp('CA');  disp(CA);

fprintf('OSDA-ETD ends!\n');    


