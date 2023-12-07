function [alpha,adist_m,adist_c] = estimate_alpha(Xs,Ys,Xt,Yt)
    C = length(unique(Ys));
    list_adist_c = [];
    epsilon = 1e-3;
    for i = 1 : C
        index_i = Ys == i;
        Xsi = Xs(index_i,:);
        index_j = Yt == i;
        Xtj = Xt(index_j,:);
        adist_i = adist(Xsi,Xtj);
        list_adist_c = [list_adist_c;adist_i];
    end
    adist_c = mean(list_adist_c);
    
    adist_m = adist(Xs,Xt);
    alpha = adist_c / (adist_c + adist_m);
    if alpha > 1    % Theoretically mu <= 1, but calculation may be over 1
        alpha = 1;
    elseif alpha <= epsilon
        alpha = 0;  
    end
end

function dist = adist(Xs,Xt)
    Yss = ones(size(Xs,1),1);
    Ytt = ones(size(Xt,1),1) * 2;
    
    % The results of fitclinear() may vary in a very small range, since Matlab uses SGD to optimize SVM.
    % The fluctuation is very small, ignore it
    model_linear = fitclinear([Xs;Xt],[Yss;Ytt],'learner','svm');
    ypred = model_linear.predict([Xs;Xt]);
    error = mae([Yss;Ytt],ypred);
    dist = 2 * (1 - 2 * error);
end