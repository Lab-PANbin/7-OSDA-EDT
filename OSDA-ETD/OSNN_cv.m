function [Cls]=OSNN_cv(Xs,Ys,Xt,Yt)
%number of Classes for Source domain
c=length(unique(Ys));
%number of Classes for target doamin
C=c+1;
%label begin
Cls=[];
%Compute the distance matrix
Dist=pdist2(Xs,Xt);
%dimension of features and number of samples
[mt,nt]=size(Xt);
%choice two cloest samples for target samples
[B,I]=mink(Dist,2,1);
%label every target sample
for i=1:mt
    %if two closet samples have same label
    if Ys(I(1,i))==Ys(I(2,i))
        Cls=[Cls;Ys(I(1,i))];
    else
        Cls=[Cls;c+1];
    end
end
        
      
             
    
    
    
  
    
  