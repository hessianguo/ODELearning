function axi = str2axi(true_nz_weight_tags,tags_pde_G)
   if ~isempty(true_nz_weight_tags)
      neq = length(true_nz_weight_tags);
      m = length(tags_pde_G);
      axi = zeros(m,neq);
      for k=1:neq
          [~,loc] = ismember(true_nz_weight_tags{k}{2},tags_pde_G);
          if any(loc==0)
              axi=[];
              return
          else
              axi(loc,k) = true_nz_weight_tags{k}{3};
          end
      end
   else
       axi=[];
   end
end
