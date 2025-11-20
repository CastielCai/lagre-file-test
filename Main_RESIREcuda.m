
projections   = importdata('proj_PMN30PT.mat');
angles        = importdata('angle_PMN30PT.mat' );
defocus_param = importdata('Defocus_parameters.mat');
size(projections)
 
addpath('./src/')

%% input  
rotation       = 'ZYX';  % Euler angles setting ZYZ
dtype          = 'single';
projections    = cast(projections,dtype);
angles         = cast(angles,dtype);
defocus_param  = cast(defocus_param,dtype);



% compute normal vector of rotation matrix
matR = zeros(3,3);
if length(rotation)~=3
    disp('rotation not recognized. Set rotation = ZYX\n'); rotation = 'ZYX';
end
for i=1:3
    switch rotation(i)
        case 'X',   matR(:,i) = [1;0;0];
        case 'Y',   matR(:,i) = [0;1;0];
        case 'Z',   matR(:,i) = [0;0;1];
        otherwise,  matR = [0,0,1;
                0,1,0;
                1,0,0];
            disp('Rotation not recognized. Set rotation = ZYX');
            break
    end
end
vec1 = matR(:,1); vec2 = matR(:,2); vec3 = matR(:,3);

% extract size of projections & num of projections
[dimx, dimy, Num_pj] = size(projections)

%% rotation matrix
Rs = zeros(3,3,Num_pj, dtype);
for k = 1:Num_pj
    phi   = angles(k,1);
    theta = angles(k,2);
    psi   = angles(k,3);
    
    % compute rotation matrix R w.r.t euler angles {phi,theta,psi}
    rotmat1 = MatrixQuaternionRot(vec1,phi);
    rotmat2 = MatrixQuaternionRot(vec2,theta);
    rotmat3 = MatrixQuaternionRot(vec3,psi);
    R =  single(rotmat1*rotmat2*rotmat3)';
    Rs(:,:,k) = R;
end

%% parameters
step_size      = 2.0;  %step_size <=1 but can be larger is sparse
iterations     = 300

dimz           = dimx

positivity     = 1;         % 1 means true, 0 means false
l2_regularizer = 0.00;     % a small positive number
is_avg_on_y    = 1;         % 1 means averaging in the y-direction, 0 means no

defocus_step   = 0; %50;    
semi_angle     = 22.0e-3;    
Voltage        = 300*10^3;
pixelsize      = 0.3079;                            
nr             = 200; 
defocus_scale  = 0.0;%0.6;

%projections_2 = single( My_paddzero(projections, [dimx,dimy,71]) );
projections = single(projections(:,:,:) );

% provide dimz or exension dimension (dimx_ext,dimy_ext,dimz)
dim_ext = [dimx,dimy,dimz];

% make sure these following parameters are double
GD_info      = [iterations, step_size]
defocus_info = [Voltage, pixelsize, nr, semi_angle, defocus_step, defocus_scale];
constraints  = [positivity, is_avg_on_y, l2_regularizer];

% only projections and Rs are single

%% iteration: minimize ||Au-b||^2 by gradient descent: run reconstruction the first time
% syntax 1: no initial rec
%tic
%[rec, cal_proj4] = RT3_defocus_1GPU( (projections), (Rs), (dimz), GD_info , (constraints),  ...
%    defocus_info, defocus_param);
%toc

tic
[rec, cal_proj4, probe, pj_shift, pj_fft, pj_ifft] = ...
    RT3_defocus_film_1GPU_2( (projections), (Rs), (dim_ext), GD_info , (constraints),  ...
    defocus_info, defocus_param);
toc

% calculated projections are multipled 4 times -> needs normalize
cal_proj = cal_proj4/4;

%%


%% calculate Rfactor
        for i=1:Num_pj
            pj = projections(:,:,i);
            resi_i=projections(:,:,i)-cal_proj(:,:,i);
            Rarr(i) = sum(abs(resi_i(:)))/ sum(abs(pj(:)));
        end
        R1=mean(Rarr);
%% Output

OBJ.InputProjections=projections;
OBJ.InputAngles=angles;
OBJ.Calprojection=cal_proj;
OBJ.reconstruction=rec;
OBJ.step_size=step_size;
OBJ.Dimx = dimx;
OBJ.Dimy = dimy;
OBJ.Dimz = dimz;
OBJ.numIterations=iterations;
OBJ.defocus_step=defocus_step;
OBJ.semi_ang=semi_angle;
OBJ.InputDefocusparas=defocus_param;
OBJ.Inputdefocus_info=defocus_info;
OBJ.Inputconstraints=constraints;
OBJ.Voltage=Voltage;
OBJ.pixelsize=pixelsize;
OBJ.nr=nr;
OBJ.Rarr=Rarr;
OBJ.R1=R1;


save("Resire_Dataset1_multislice_N300_noDefocus.mat","OBJ",'-v7.3')


%exit







%return
%% defocus 1 with initial rec
step_size=3;
iterations=50;
GD_info      = [iterations, step_size];
constraints  = [positivity, is_avg_on_y, 0.005];

tic
[rec1, cal_proj4] = RT3_defocus_1GPU( (projections), (Rs), (dimz), GD_info , (constraints),  ...
    defocus_info, defocus_param, rec);
toc
cal_proj = cal_proj4/4;

%% show results

figure(1); img(rec1,'RESIRE rec' , 'caxis',[0,max(rec1(:))]);
figure(2); img(projections,'measured projections', cal_proj,'calculated projection', projections-cal_proj,'residual')    




