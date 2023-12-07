
function  [OS,OS_star,ALL,ALL_star,CA] = OSDA_ETD(Xs,Ys,Xt,Yt,options)

%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim
%%% Yt      : Target domain label matrix, m * 1 (only used for testing accuracy)
%%% options : algorithm options:


%% Outputs:
%%%% OS      :  all classes average accuracy
%%%% OS_star :  known classes average accuracy
%%%% ALL     :  all classes overall accuracy
%%%% ALL_star:  known classes overall accuracy
%%%% CA      :  classes accuracy


%% Algorithm starts here
    fprintf('OSDA-ETD starts...\n');
    
    %% Load algorithm options
    if ~isfield(options,'p')
        options.p = 10;
    end
    if ~isfield(options,'rho')
        options.rho = 1;
    end
    if ~isfield(options,'lambda')
        options.lambda = 50;
    end
    if ~isfield(options,'mu')
        options.mu = 1.0;
    end
    if ~isfield(options,'T')
        options.T = 10;
    end
    if ~isfield(options,'gamma')
        options.alpha = 0.40;
    end
    if ~isfield(options,'sigma')
        options.beta = 0.35;
    end
     if ~isfield(options,'eta')
        options.eta = 1.5;
     end
     
    Xs = double(Xs');
    Xt = double(Xt');
    X = [Xs,Xt];
    ns = size(Xs,2);
    nt = size(Xt,2);
    C = length(unique(Ys));
    acc_iter = [];
    YY = [];
    for c = 1 : (C+1)
        YY = [YY,Ys==c];
    end
    YY = [YY;[zeros(nt,C),ones(nt,1)]];
    YY2= [[zeros(ns,C),ones(ns,1)];zeros(nt,C+1)];

    %% Data normalization
    X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));

    %% Construct graph Laplacian
    if options.rho > 0
        manifold.k = options.p;
        manifold.Metric = 'Cosine';
        manifold.NeighborMode = 'KNN';
        manifold.WeightMode = 'Cosine';
        W = lapgraph(X',manifold);
        Dw = diag(sparse(sqrt(1 ./ sum(W))));
        L = eye(ns + nt) - Dw * W * Dw;
    else
        L = 0;
    end
    
    % Generate soft labels for the target domain
    Cls =OSNN_cv(Xs',Ys,Xt',Yt);
    Cls_check_convergence=Cls;
    

    % Construct kernel
    K = kernel_meda('rbf',X,sqrt(sum(sum(X .^ 2).^0.5)/(ns + nt)));
    nummax = max(nt,ns);
    
    E = diag(sparse([nummax*ones(ns,1)/ns;options.gamma*nummax*ones(nt,1)/nt]));
    E2 = diag(sparse([options.sigma*nummax*ones(ns,1)/ns;zeros(nt,1)/nt]));

    for t = 1 : options.T
        
        % Estimate alpha
        known_position = find(Cls<(C+1));
        temp0 = Xt';
        Xt_known = temp0(known_position,:);
        Cls_known= Cls(known_position,1);
        alpha= estimate_alpha(Xs',Ys,Xt_known,Cls_known); 
    
        % Transferability
        KL=zeros(nt,1);
        KL(known_position,1)=1;
        ntk = sum(KL);
        e = [1 / ns * ones(ns,1); -1 / ntk * KL];
        M0 = e * e' * length(unique(Ys));
        As=[];
        At=[];
        for i=1:C
             temp1=onehot(Ys,unique(Ys));
             Ass=1/length(find(Ys == i))*temp1(:,i);
             As=[As,Ass];
             temp2=onehot(Cls,unique(Ys));
             Att=1/length(find(Cls == i))*temp2(:,i);
             At=[At,Att];        
        end
        M1=[As*As',-As*At';-At*As',At*At'];
        M = (1 - alpha) * M0 + alpha * M1;
        M = M / norm(M,'fro');

        %Discriminability
        Fs=[];
        Ft=[];
        for j=1:C        
            Fs=[Fs,repmat(As(:,j),1,C-1)];
            idx=1:C;
            idx(j)=[];
            Ft=[Ft,At(:,idx)];
        end
        F=[Fs*Fs',-Fs*Ft';-Ft*Fs',Ft*Ft'];
        F = F / norm(F,'fro');
         
        % Compute coefficients vector Theta
        Theta = ((E-E2 + options.lambda * M + options.mu * L-options.eta*F) * K + options.rho * speye(ns + nt,ns + nt)) \ (E * YY-E2*YY2);
        g = K * Theta;
        [~,Cls] = max(g,[],2);
         Cls = Cls(ns+1:end);

       if Cls== Cls_check_convergence
           break;
       else
           Cls_check_convergence=Cls;
       end
        
    end
       %% Compute accuracy
       
       OS_temp=0;
       OS_star_temp=0;
       CA=[];
       correct_number=0;
       correctstar_number=0;
       number_k=0;
    
       for j=1:(C+1)
         LL=find(Yt==j);
         CA(j)=length(find(Cls(LL,1)==Yt(LL,1)))/length(Yt(LL,1));
         correct_number=correct_number+length(find(Cls(LL,1)==Yt(LL,1)));
         OS_temp=OS_temp+length(find(Cls(LL,1)==Yt(LL,1)))/length(Yt(LL,1));
       end
         ALL=correct_number/nt;
         OS=OS_temp/(C+1);
         
       for j=1:C
         LL=find(Yt==j);
         correctstar_number=correctstar_number+length(find(Cls(LL,1)==Yt(LL,1)));
         number_k=number_k+length(Yt(LL,1));
         OS_star_temp=OS_star_temp+length(find(Cls(LL,1)==Yt(LL,1)))/length(Yt(LL,1));
        end
        OS_star=OS_star_temp/C;
        ALL_star=correctstar_number/number_k;
        
end

function K = kernel_meda(ker,X,sigma)
    switch ker
        case 'linear'
            K = X' * X;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        
        case 'sam'
            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));
        otherwise
            error(['Unsupported kernel ' ker])
    end
end
function y_onehot=onehot(y,class)
    % Encode label to onehot form
    % Input:
    % y: label vector, N*1
    % Output:
    % y_onehot: onehot label matrix, N*C  

    nc=length(class);
    y_onehot=zeros(length(y), nc);
    for i=1:length(y)
        y_onehot(i, class==y(i))=1;
    end
end
