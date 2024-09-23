clear all;close all;
set(0, 'DefaultFigureColor', 'white');

i=0
i=i+1;f_post_h5_all{i}='POST_DAUGAARD_AVG_prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5';
i=i+1;f_post_h5_all{i}='POST_DAUGAARD_AVG_prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_N500000_Nh280_Nf12_Nu500000_aT1.h5';
i=i+1;f_post_h5_all{i}='POST_DAUGAARD_AVG_prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_N10000_Nh280_Nf12_Nu10000_aT1.h5';
i=i+1;f_post_h5_all{i}='POST_DAUGAARD_AVG_PRIOR_WB30_N500000_log-uniform_R10_500_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu500000_aT1.h5';
i=i+1;f_post_h5_all{i}='POST_DAUGAARD_AVG_PRIOR_UNIFORM_NL_1-12_log-uniform_N500000_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu500000_aT1.h5';
i=i+1;f_post_h5_all{i}='POST_DAUGAARD_AVG_PRIOR_UNIFORM_NL_1-12_log-uniform_N500000_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5';

ele_arr = 70:-1:-60;

for i=1:length(f_post_h5_all);
    f_post_h5 = f_post_h5_all{i}
    [p,f]=fileparts(f_post_h5);
    figure(1);
    subfigure(2,3,4)
    useLog=1;
    integrate_plot_elevation_movie(f_post_h5,'/M1','Median',ele_arr,useLog)
    try;
        figure(2);
        subfigure(2,3,6)
        useLog=0;
        integrate_plot_elevation_movie(f_post_h5,'/M2','Mode',ele_arr,useLog);
    end

    %%
    figure;
    ele = 5;
    figure;
    subfigure(2,3,1)
    useLog=1;
    integrate_plot_2d_elevation(f_post_h5,'/M1','Median',ele,useLog)
    print_mul(sprintf('%s_m1_ele%d',f,ele))
    try
        figure;
        subfigure(2,3,3)
        useLog=0;  
        integrate_plot_2d_elevation(f_post_h5,'/M2','Mode',ele,useLog)
        print_mul(sprintf('%s_m2_ele%d',f,ele))

    end
end