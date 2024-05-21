f_data_h5 = 'DAUGAARD_AVG.h5';
f_prior_data ='PRIOR_Daugaard_N2000000_TX07_20230731_2x4_RC20-33_Nh280_Nf12.h5';
f_post_h5 = 'POST_DAUGAARD_AVG_PRIOR_Daugaard_N2000000_TX07_20230731_2x4_RC20-33_Nh280_Nf12_Nu100000_aT1.h5';




%% plot profile
close all

[X, Y, ELE, LINE] = integrate_get_geometry(f_data_h5);
iline0=find([1,diff(LINE)>1]);
nline =diff(iline0);
iline = find(nline==max(nline));iline=iline(1);
ii=[0:1:(nline(iline)-1)]+iline0(iline);

integrate_plot_profile(f_post_h5,ii)


%%
is = ceil(median(ii));
%%
figure;clf;
integrate_plot_sounding(f_post_h5, is)
ppp(11,8,8,2,1)
%%
figure;clf;
integrate_plot_sounding_model(f_post_h5, is)