% Builds L-UQ.mltbx, the installable MATLAB toolbox package.
% Run from the repository root:  matlab -batch package_toolbox
%
% The toolbox files are first copied to a clean staging folder:
% ToolboxOptions(folder, ...) catalogs EVERYTHING under the folder, so
% pointing it at the repository root would make it scan python/.venv
% (tens of thousands of files) and .git.
root = fileparts(mfilename('fullpath'));
names = {'lmom.m','LegendreShiftPoly.m','Identify_dist.m', ...
    'Parameter_estimation.m','parameter_identify.m','fit_best.m', ...
    'PDF_l.m','CDF_l.m','Random_l.m','KLDiv.m','JSDiv.m', ...
    'demo_example.m','README.md','LICENSE','CHANGELOG.md','CITATION.cff'};
stage = fullfile(tempdir, 'L-UQ-toolbox-stage');
if exist(stage, 'dir'), rmdir(stage, 's'); end
mkdir(stage); mkdir(fullfile(stage, 'tests'));
for i = 1:numel(names)
    src = fullfile(root, names{i});
    if isfile(src), copyfile(src, fullfile(stage, names{i})); end
end
copyfile(fullfile(root,'tests','test_uq_matlab.m'), ...
    fullfile(stage,'tests','test_uq_matlab.m'));
copyfile(fullfile(root,'tests','octave_verify.m'), ...
    fullfile(stage,'tests','octave_verify.m'));

opts = matlab.addons.toolbox.ToolboxOptions(stage, ...
    'a3d8f2c1-7b4e-4d9a-b1c6-2e5f8a0d3c7b');
opts.ToolboxName = 'L-UQ';
opts.ToolboxVersion = '1.0.0';
opts.Summary = ['L-moments based uncertainty quantification from ', ...
    'scarce samples including extremes'];
opts.Description = ['Identify a distribution family from the L-moment ', ...
    'ratio diagram, estimate parameters in closed form with domain ', ...
    'guards and ranked fallback (fit_best), evaluate PDF/CDF/random ', ...
    'sampling, and compare fit quality via KL/JS divergence. ', ...
    'Companion Python package: lmoments-uq (PyPI).'];
opts.AuthorName = 'Deepan Jayaraman';
opts.AuthorEmail = 'deepanjayram@gmail.com';
opts.AuthorCompany = 'Indian Institute of Technology Madras';
opts.ToolboxMatlabPath = stage;   % staged files are the toolbox root
opts.MinimumMatlabRelease = 'R2018b';
opts.OutputFile = fullfile(root, 'L-UQ.mltbx');
matlab.addons.toolbox.packageToolbox(opts);
fprintf('packaged: %s\n', opts.OutputFile);
