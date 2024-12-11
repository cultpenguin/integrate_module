clear all

f_post_h5 = 'POST_DAUGAARD_AVG_PRIOR_CHI2_NF_5_log-uniform_N500_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu1000000_aT1.h5';
f_post_h5 = 'POST_DAUGAARD_AVG_prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu200000_aT1.h5';
f_post_h5 = 'POST_DAUGAARD_AVG_prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu200000_aT1.h5';
%f_post_h5 = 'POST_HALD_AVG_PRIOR_WB_CHI2_NF_5_log-uniform_N200000_TX07_20230731_2x4_RC20-33_Nh280_Nf12_Nu100000000000_aT1.h5'
%f_post_h5 = 'POST_HALD_AVG_PRIOR_WB_CHI2_NF_5_log-uniform_N200000_TX07_20230731_2x4_RC20-33_Nh280_Nf12_Nu100000000000_aT1.h5';
%f_post_h5 = 'POST_DAUGAARD_AVG_PRIOR_CHI2_NF_5_log-uniform_N1000000_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu1000000_aT1.h5';

ktype = 'sk';
ktype = 'ok';

[p,f,ext]=fileparts(f_post_h5);

% Load '/i_use' from f_post_h5


f_prior_h5 = h5readatt(f_post_h5,'/','f5_prior');
f_data_h5 = h5readatt(f_post_h5,'/','f5_data');
X = h5read(f_data_h5,'/UTMX');
Y = h5read(f_data_h5,'/UTMY');
ELE = h5read(f_data_h5,'/ELEVATION');

%h5disp(f_post_h5)
i_use = h5read(f_post_h5, '/i_use');

M1 = h5read(f_post_h5, '/M1/LogMean');
M1std = h5read(f_post_h5, '/M1/Std');
M1std_log = M1std./M1;

M1 = real(log10(h5read(f_post_h5, '/M1/LogMean')));
M1std = M1std_log;

nz=size(M1,1);

nx=60;
ny=61;
x = linspace(min(X)-100,min(X)+1000,nx);
y = 500+linspace(min(Y)-100,min(Y)+1000,ny);

x = linspace(min(X)-100,max(X)+100,nx);
y = linspace(min(Y)-100,max(Y)+100,ny);
dx=10;
x = [(min(X)-100):dx:(max(X)+100)];nx=length(x);
y = [(min(Y)-100):dx:(max(Y)+100)];ny=length(y);

[xx,yy]=meshgrid(x,y);
pos_est=[xx(:),yy(:)];

sill=(0.1)^2;
range=100;
V=sprintf('%d Nug(0) + %f Gau(%3.1f)',0.01*sill,sill,range);
options.max=20;

M1_3D=zeros(ny,nx,nz);
M1_3D_std=zeros(ny,nx,nz);

%
for iz=1:1:nz;
    val_known = [M1(iz,:)' M1std(iz,:)'];
    pos_known = [X(:), Y(:)];
    if strcmp(ktype,'sk')
        options.mean=mean(val_known(:,1)); % Simple Kriging
    end

    [d_est,d_var]=krig(pos_known,val_known,pos_est,V,options);
    M1_3D(:,:,iz) = reshape(d_est,ny,nx);
    M1_3D_std(:,:,iz) = reshape(sqrt(d_var),ny,nx);

    figure_focus(1);
    subplot(1,2,1)
    imagesc(x,y,M1_3D(:,:,iz));
    A=M1_3D_std(:,:,1)./sqrt(sill);
    alpha(1-A)
    axis image
    set(gca,'ydir','normal')
    caxis([log10(10),log10(500)])
    colormap(cmap_geosoft)
    colorbar
    title(sprintf('iz=%d',iz))
    
   
    subplot(1,2,2)
    %plot(X,Y,'k.','MarkerSize',6)
    %hold on
    scatter(X,Y,8,val_known(:,1),'filled')
    hold off
    set(gca,'ydir','normal')
    axis image
    caxis([log10(10),log10(500)])
    colormap(cmap_geosoft)
    colorbar
    title(sprintf('iz=%d',iz))
    box on
    drawnow

end

%%
write_eas_matrix(sprintf('%s_%s_est.eas',ktype,f),M1_3D,'Estimate');
write_eas_matrix(sprintf('%s_%s_std.eas',ktype,f),M1_3D_std,'Estimate');

%%
M1_3D(M1_3D_std>(0.9*sqrt(sill)))=NaN;
v3d(M1_3D,1,[log10(1),log10(500)],cmap_geosoft)
v3d(M1_3D_std,1,[0,sqrt(sill)],hot)
 

