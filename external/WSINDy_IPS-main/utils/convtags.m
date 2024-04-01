function tags = convtags(dimx,varargin)
   defaultExp = [];
   defaultMon = [];
   defaultSing = [];
   defaultGauss = [];
   defaultSingeps = 10^-2;
   defaultsvdtol = [];
   defaultutagin = [];
   defaultutagout = [];
   defaultpsitags = [];

   p = inputParser;
   validScalarPosNum = @(x) isnumeric(x) && all(x >= 0);
   addRequired(p,'dimx',validScalarPosNum);
   addParameter(p,'Exp',defaultExp,validScalarPosNum);
   addParameter(p,'Mon',defaultMon,validScalarPosNum);
   addParameter(p,'Sing',defaultSing,@(x)validScalarPosNum(-x));
   addParameter(p,'Gauss',defaultGauss);
   addParameter(p,'Singeps',defaultSingeps,validScalarPosNum);
   addParameter(p,'svdtol',defaultsvdtol);
   addParameter(p,'utagin',defaultutagin);
   addParameter(p,'utagout',defaultutagout);
   addParameter(p,'psitags',defaultpsitags);
   parse(p,varargin{:});
   
   dimx=p.Results.dimx;
   utagin = p.Results.utagin;
   utagout = p.Results.utagout;
   psitags = p.Results.psitags;  
   
   [utagin,utagout,psitags] = ndgrid(utagin,utagout,psitags);
   utags = [utagin(:) utagout(:) psitags(:)];
   
   
   tags = {};
   if ~isempty(utags)
       for nn = size(utags,1)
           utagin = utags(nn,1);
           utagout = utags(nn,2);
           psitags = [eye(dimx)*utags(nn,3) zeros(dimx,1)];
           Expparams = p.Results.Exp;
           ExpL = length(Expparams);
           Expcell = cell(dimx*ExpL,1);
           Monparams = p.Results.Mon;
           MonL = length(Monparams); 
           Moncell = cell(dimx*MonL,1);
           Singparams = p.Results.Sing;
           SingL = length(Singparams);
           Gaussparams = p.Results.Gauss;
           GaussL = size(Gaussparams,2);
           Gausscell = cell(dimx*GaussL,1);
           Singcell = cell(dimx*SingL,1);
           svdtol = p.Results.svdtol;

           f = @(h,r) h(r)./max(r,eps); 

           for i=1:ExpL
               if dimx == 1
                   Expcell{i} = {utagin, @(x,t) -Expparams(i)*f(@(x)exp(-Expparams(i)*x),abs(x)).*x,psitags(1,:),utagout,svdtol,['convex',num2str(Expparams(i))]};
               elseif dimx == 2
                   Expcell{2*i-1} = {utagin, @(x,y,t) -Expparams(i)*f(@(x)exp(-Expparams(i)*x),hypot(x,y)).*x,psitags(1,:),utagout,svdtol,['convex',num2str(Expparams(i))]};
                   Expcell{2*i} = {utagin, @(x,y,t) -Expparams(i)*f(@(x)exp(-Expparams(i)*x),hypot(x,y)).*y,psitags(2,:),utagout,svdtol,['convey',num2str(Expparams(i))]};
               end
           end

           for i=1:MonL
               if dimx == 1
                   Moncell{i} = {utagin, @(x,t) f(@(x)x.^Monparams(i),abs(x)).*x,psitags(1,:),utagout,svdtol,['convmonx',num2str(Monparams(i))]};
               elseif dimx == 2
                   Moncell{2*i-1} = {utagin, @(x,y,t) f(@(r)r.^Monparams(i),hypot(x,y)).*x,psitags(1,:),utagout,svdtol,['convmonx',num2str(Monparams(i))]};
                   Moncell{2*i} = {utagin, @(x,y,t) f(@(r)r.^Monparams(i),hypot(x,y)).*y,psitags(2,:),utagout,svdtol,['convmony',num2str(Monparams(i))]};
               end
           end

          for i=1:GaussL
               if dimx == 1
                   Gausscell{i} = {utagin, @(x,t) f(@(x)exp(-(x-Gaussparams(1,i)).^2/2/Gaussparams(2,i)^2),abs(x)).*x,psitags(1,:),utagout,svdtol,['convgaussx',num2str(Gaussparams(1,i))]};
               elseif dimx == 2
                   Gausscell{2*i-1} = {utagin, @(x,y,t) f(@(r)exp(-(r-Gaussparams(1,i)).^2/2/Gaussparams(2,i)^2),hypot(x,y)).*x,psitags(1,:),utagout,svdtol,['convgaussx',num2str(Gaussparams(1,i))]};
                   Gausscell{2*i} = {utagin, @(x,y,t) f(@(r)exp(-(r-Gaussparams(1,i)).^2/2/Gaussparams(2,i)^2),hypot(x,y)).*y,psitags(1,:),utagout,svdtol,['convgaussy',num2str(Gaussparams(1,i))]};
               end
           end

           
           for i=1:SingL
               if dimx == 1
                   if Singparams(i)<0
                       Singcell{i} = {utagin, @(x,t) f(@(x)(max(x,p.Results.Singeps)).^(Singparams(i)),abs(x)).*x,psitags(1,:),utagout,svdtol,['convsingx',num2str(Singparams(i))]};
                   elseif Singparams(i)==0
                       Singcell{i} = {utagin, @(x,t) f(@(x)log(max(x,p.Results.Singeps)),abs(x)).*x,psitags(1,:),utagout,svdtol,['convsingx',num2str(Singparams(i))]};
                   end
               elseif dimx == 2
                   if Singparams(i)<0
                       Singcell{2*i-1} = {utagin, @(x,y,t) f(@(r)(max(r,p.Results.Singeps)).^(Singparams(i)),hypot(x,y)).*x,psitags(1,:),utagout,svdtol,['convsingx',num2str(Singparams(i))]};
                       Singcell{2*i} = {utagin, @(x,y,t) f(@(r)(max(r,p.Results.Singeps)).^(Singparams(i)),hypot(x,y)).*y,psitags(2,:),utagout,svdtol,['convsingy',num2str(Singparams(i))]};
                   elseif Singparams(i)==0
                       Singcell{2*i-1} = {utagin, @(x,y,t) f(@(r)log(max(r,p.Results.Singeps)),hypot(x,y)).*x,psitags(1,:),utagout,svdtol,['convsingx',num2str(Singparams(i))]};
                       Singcell{2*i} = {utagin, @(x,y,t) f(@(r)log(max(r,p.Results.Singeps)),hypot(x,y)).*y,psitags(2,:),utagout,svdtol,['convsingy',num2str(Singparams(i))]};
                   end
               end
           end 

           tags = [tags;[Expcell;Moncell;Singcell;Gausscell]];
       end
   end   
end

