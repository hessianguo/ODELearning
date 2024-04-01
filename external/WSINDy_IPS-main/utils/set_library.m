tic,
drifttags = get_drifttags(driftpolys,drifttrigs,diffpolys,difftrigs,dim,crossdrift);
[tags_pde_0,lib_list_0,~,lhs_ind_0,max_dx,max_dt,polys,customf,customconv]  = get_lib_tags(n,dim,lhs,max_dx,max_dt,polys,trigs,use_cross_dx,use_all_dt,custom_remove,custom_add,drifttags,convargs,true_nz_weights);
ET_build_lib = toc;
fprintf(1,'ET_build_lib = %4.4f \n',ET_build_lib);
