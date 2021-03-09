function [x,xt,xt_res]= model_proj(xx,xxt_res,aa,annmax,nlevel,xa,nelin,next1,next2,lc,lstart,lperiod,data_ext,inorm,std_r)
	[npc,length,nlevel] = size(xx);
	x=xx;
	xt_res = xxt_res;
	nmax=npc;
%        MAIN LOOP
	for nl=1:nlevel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	xt = zeros(length,nmax);
%	xt_res = zeros(length,nmax);
		for n=1:nmax
		for l=lstart:length-1
		xt(l,n)=x(n,l+1,nl)-x(n,l,nl);
		end
%	        xt(lc,n)=x(n,lc,nl)-x(n,lc-1,nl);
	        xt(length,n)=x(n,length,nl)-x(n,length-1,nl);
		end
      nc=annmax(nl);
      svd=zeros(length,nc);
%
      for l=lstart:length 
%       
      if nl==1 
      count=1;
      xa(count,nl)=1;                            
%       
      for n=1:nmax 
      count=count+1;
          xa(count,nl)=x(n,l,nl);
      end                                     
	if nelin > 1 
      for n=1:nmax
      for n_1=1:n
        count=count+1;
          xa(count,nl)=x(n,l,nl)*x(n_1,l,nl);
      end
      end                                    
	end
	if nelin > 2
      for n=1:nmax
      for n_1=1:n
      for n_2=1:n_1
        count=count+1;
          xa(count,1)=x(n,l,1)*x(n_1,l,1)*x(n_2,l,1);
      end
      end                                    
      end                                    
	end
	if nelin > 3 
      for n=1:nmax
      for n_1=1:n
      for n_2=1:n_1
      for n_3=1:n_2
        count=count+1;
          xa(count,1)=x(n,l,1)*x(n_1,l,1)*x(n_2,l,1)*x(n_3,l,1);
      end
      end                                    
      end                                    
      end                                    
	end

		if next1 ~= 0  
        
		for i=1:2*lperiod
        count = count+1;
        xa(count,1)= data_ext(l,i);
		if next1 > 1*lperiod  
	         for n=1:nmax
	         count = count+1;
	         xa(count,1) = data_ext(l,i)*x(n,l,1);
	         end
		end
        end
 
		end


	else
      count=1;
      xa(count,nl)=1;
     
      for nl_2=1:nl
      for n=1:nmax
       count=count+1;
       xa(count,nl)=x(n,l,nl_2);
        end         
        end      


	end  % of if nl==1

	for n_1=1:count
	svd(l,n_1)=xa(n_1,nl);
	end
	end   % of l=1,length loop
%%% output of the correlation matrix
bb=zeros(nc,npc); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:npc
  for n_1=1:annmax(nl)
  bb(n_1,n)=aa(n,n_1,nl);
  end
xt_res(lstart:length,n)=xt(lstart:length,n)-svd(lstart:length,:)*bb(:,n);
end
        if nl ~= nlevel 

        for l=lstart:length
        for n=1:nmax
        if inorm ~=1
        x(n,l,nl+1)=xt_res(l,n);
        else
        x(n,l,nl+1)=xt_res(l,n)/std_r(nl,n);
        end

        end
        end

        end

 	end	%of nl=1,NLEVEL		




	return
