
function data0_Q=division( Mod,m1,I,data0)

data0_Q={};

     for  M=1:Mod
               
         IM=I(((M-1)*m1+1):(m1*M),:);
         data0_M=data0(IM,:);
         data0_Q=[data0_Q,data0_M];
           
     end


end



