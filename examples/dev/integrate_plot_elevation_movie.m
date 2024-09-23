function vname_out = integrate_plot_elevation_movie(f_post_h5,DataGroup,ProbGroup,ele_array,useLog)
if nargin<1, f_post_h5 = 'POST_DAUGAARD_AVG_PRIOR_UNIFORM_NL_1-12_log-uniform_N500000_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu500000_aT1.h5'; end
if nargin<2, DataGroup='/M1';end
if nargin<3, ProbGroup='Median';end
if nargin<4, ele_array = 80:-1:-60;end
if nargin<4, useLog=0;end

[p,f]=fileparts(f_post_h5);

vname=sprintf('%s_%s_%s_elevation',f,DataGroup(2:end),ProbGroup);
FrameRate = 15;
Quality = 99;


writerObj = VideoWriter(vname);
%writerObj = VideoWriter(vname,'MPEG-4'); % Awful quality ?
writerObj.FrameRate=FrameRate;
writerObj.Quality=Quality;
open(writerObj);

i=0;
for ele = ele_array
    i=i+1;
    integrate_plot_2d_elevation(f_post_h5,DataGroup,ProbGroup,ele,useLog);drawnow;
    frame = getframe(gcf);
    writeVideo(writerObj,frame);
end
vname_out = writerObj.Filename;
close(writerObj);
