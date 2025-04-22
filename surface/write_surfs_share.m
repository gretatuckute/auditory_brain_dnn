function params_out = write_surfs_share(varargin)
% This is a function that will write surfaces on the fsaverage surface.
%
% Specifically, it will go through the specified folder directory 
% (specified by ROOTDIR, MODEL_IDENTIFIER, and SUBFOLDER) and writes the surfaces.
% And it writes them to .../written_surfaces/.
% 
% The required input is a .mat file with the following fields:
% vals: the values you want to plot
% x_ras and y_ras: voxel coordinates, from the neural data meta.
%
% <varargin determines whether <do_nearest_neighbor> is true or false
%     which determines whether you use linear interpolation or nearest-neighbor
%     interpolation
% 
% Note: You must have freesurfer installed for this to work.
% It also requires a voxel_matrix_for_alex.mat to be in the ROOTDIR.
% Function by Alex Kell, smaller edits by Greta Tuckute.
%
p=inputParser();
addParameter(p, 'MODEL_IDENTIFIER', 'Kell2018word');
addParameter(p, 'ROOTDIR', '/Users/gt/Documents/GitHub/aud-dnn/data/fsavg_surf/'); % OM_transfer_20220127/
addParameter(p, 'SUBFOLDER', 'TYPE=subj-argmax_METRIC=median_r2_test_c')
parse(p, varargin{:});
params = p.Results;
params_out = params;


%% PARAMS %%
do_nearest_neighbor = true;

default_val = nan;
% default_val = -2; % for difference volumes

if ~isnan(default_val)
  fprintf('\n\nNOTE YOUR DEFAULT VALUE HERE IS **NOT**: NaN\n');
  fprintf('  INSTEAD IT IS: %d\n\n',default_val);
end

if do_nearest_neighbor
  fprintf('\nNote that you are using:\n  **NEAREST NEIGHBOR** interpolation.\n\n')
end

% because you will forget:
% sam put RH first and LH second in these cell arrays 
idxs = struct(); idxs.rh = 1; idxs.lh = 2;
grid_ras_steps = 2;

% determine which dataset you want to run for 
mats_for_writing_d = [params.ROOTDIR filesep params.MODEL_IDENTIFIER filesep params.SUBFOLDER '/'];
mats_dump_dir = [mats_for_writing_d filesep 'written'];
output_d = [mats_for_writing_d filesep 'written_surfaces'];

voxel_info_fp = [params.ROOTDIR filesep 'voxel_matrix_for_alex.mat'];
hemis = {'lh','rh'};

% get the filenames for the mats
d = dir(mats_for_writing_d); %mats_for_writing_d if running a new one!! otherwise mat_dump_dir, because they will be copied over in that folder!
fnames = {d.name}'; 
% short circuit and so won't fail if not long enough
comp_func = @(s)(length(s)>length('.mat') && strcmp(s(end-3:end),'.mat'));
is_mat_file = cellfun(comp_func,fnames);
fnames = fnames(is_mat_file);

% create the necessary subdirs if not there 
if ~exist(output_d,'dir')
  mkdir(output_d)
end
if ~exist(mats_dump_dir)
  mkdir(mats_dump_dir)
end

% load the coordinate info
coord_info = load(voxel_info_fp);
% fields of interest:
%   ras_coords_fsaverage:    {[28729x2 double]  [28495x2 double]}
%   voxel_indices_fsaverage: {[28729x1 int32]   [28495x1 int32]}

% load the necessary fsaverage info
% [~,fs_home] = unix('echo $FREESURFER_HOME');
% fs_home = strtrim(fs_home); % often has a trailing newline -- get rid of it 
fs_home = '/Applications/freesurfer';
addpath([fs_home filesep 'matlab']);

mri_sts = struct();

% Get the fsaverage surface %
for hemi_ii = 1:2
  hemi = hemis{hemi_ii};
  % just an arbitrary filename
  tmp = MRIread([fs_home filesep 'subjects/fsaverage/surf/' hemi '.orig.avg.area.mgh']);
  tmp.vol = default_val * tmp.vol; 
  mri_sts.(hemi) = tmp;
end

% interpolate data and write the surface
n_fnames = length(fnames);

for f_ii = 1:n_fnames
  fname = fnames{f_ii};
  fprintf('\n  %d of %d: %s\n',f_ii,n_fnames,fname);
  splits = strsplit(fname,'.');
  hemi = splits{1}(end-1:end);
  sub_id = splits{1}(end-3);
  
  mat_dump = load([mats_for_writing_d filesep fname]);
  
  % set up grids (with the values in the mat_dump file), so that each point
  % in the grid has a value of interest to be written
  % use: mat_dump.x_ras, mat_dump.y_ras, mat_dump.vals
  min_x = min(mat_dump.x_ras); max_x = max(mat_dump.x_ras);
  min_y = min(mat_dump.y_ras); max_y = max(mat_dump.y_ras);
  x_vec = double(min_x:grid_ras_steps:max_x);
  y_vec = double(min_y:grid_ras_steps:max_y);
  [grid_x,grid_y] = meshgrid(x_vec,y_vec);
  grid_vals = zeros(size(grid_x));
  for val_ii = 1:length(mat_dump.vals)
    val = mat_dump.vals(val_ii);
    x_idx = mat_dump.x_ras(val_ii); y_idx = mat_dump.y_ras(val_ii);
    grid_vals( grid_x==x_idx & grid_y==y_idx ) = val;
  end

  % interp2(x, y, z, target_x, target_y)
  % wants X and Y to be monotonic and plaid
  % interpolate the grids with the values of interest to the fsaverage
  % space?
  if do_nearest_neighbor
    vals_at_targets = interp2( grid_x,...
                               grid_y, ...
                               grid_vals, ...
                               coord_info.ras_coords_fsaverage{idxs.(hemi)}(:,1), ... % ras_coords_fsaverage has a coordinate for each x and y val
                               coord_info.ras_coords_fsaverage{idxs.(hemi)}(:,2), ...
                               'nearest');
  else
    vals_at_targets = interp2( grid_x,...
                               grid_y, ...
                               grid_vals, ...
                               coord_info.ras_coords_fsaverage{idxs.(hemi)}(:,1), ...
                               coord_info.ras_coords_fsaverage{idxs.(hemi)}(:,2) );
  end

  % if nans, let 'em be nans, but give yourself the option to do something different
  %   -- freeview will make nans zero which can be very distracting
  vals_at_targets(isnan(vals_at_targets)) = default_val; 
  vals_at_targets( abs(vals_at_targets)<1e-10 ) = default_val; 

  % vals_at_targets will be in the fsaverage space (a vector, because the grid was mapped to the coordinates in ras_coords_fsaverage).
  % then adds those to the surface loaded (mri_st)
  
  % write it
  mri_st = mri_sts.(hemi);
  if do_nearest_neighbor
    mri_fname = [fname(1:end-4) '.nearest.mgh'];
  else
    mri_fname = [fname(1:end-4) '.mgh'];
  end
  % write the values of interest to the surface files loaded from
  % freesurfer (transforming the grid to the indices that match fsaverage
  % surface?)
  % +1 because the fsaverage indices are listed as zero indexed
  mri_st.vol(:,coord_info.voxel_indices_fsaverage{idxs.(hemi)}+1) = vals_at_targets; % mri_st is the temp fsaverage surface
  mri_st.fspec = mri_fname;
  MRIwrite(mri_st,[output_d filesep mri_fname]); 

  % move the mat to the written dir
  movefile([mats_for_writing_d filesep fname],[mats_dump_dir filesep fname]);
end

end
