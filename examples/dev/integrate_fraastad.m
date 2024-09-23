
f_post_h5 = 'POST_Fra20200930_202001001_1_AVG_export_PRIOR_UNIFORM_NL_1-4_log-uniform_N1000000_fraastad_ttem_Nh280_Nf12_Nu100000_aT1.h5';
integrate_posterior_stats(f_post_h5);
%f_post_h5 = 'POST_Fra20200930_202001001_1_AVG_export_PRIOR_UNIFORM_NL_1-4_log-uniform_N1000000_fraastad_ttem_Nh280_Nf12_Nu50000_aT1.h5';
%f_post_h5 = 'POST_Fra20200930_202001001_1_AVG_export_PRIOR_UNIFORM_NL_1-4_log-uniform_N1000000_fraastad_ttem_Nh280_Nf12_Nu10000_aT1.h5';


%%

integrate_plot_data_TEV(f_post_h5,'plotT',1)

%%
is=100;
integrate_plot_sounding(f_post_h5, is)

%%
ii_plot=1:3:8000;
integrate_plot_profile(f_post_h5,ii_plot);


%%
integrate_plot_3d_continuous(f_post_h5);


%%
filename = 'Fraastad_video.mp4';
writerObj = VideoWriter(filename);
writerObj.FrameRate = 10;  % You can set the frame rate as needed
open(writerObj);

ele_arr = 30:-1:-90;
for i=1:length(ele_arr)
    ele =ele_arr(i);
    integrate_plot_2d_elevation(f_post_h5,'/M1','Median',ele);
    caxis([1 100])
    drawnow;

    % Capture the plot as a frame
    frame = getframe(gcf);  % Capture the current figure as a frame
    writeVideo(writerObj, frame);

end

close(writerObj);

