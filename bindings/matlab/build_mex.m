function build_mex(targets)
% build_mex  Compile the n4m MEX shims against libn4m.
%
% build_mex({'n4m_role_pipeline_mex', 'n4m_version_mex'}) builds a subset.
% Set N4M_INCLUDE_DIR, N4M_GENERATED_DIR and N4M_LIB_DIR before
% running, or pass them as setenv calls before invoking this script.
% The headers and library must belong to the same release. With no overrides,
% falls back to the dev-release CMake build inside the repo.

matlab_root = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(matlab_root));
inc_dir = getenv("N4M_INCLUDE_DIR");
if isempty(inc_dir)
    inc_dir = fullfile(repo_root, "cpp", "include");
end
gen_dir = getenv("N4M_GENERATED_DIR");
if isempty(gen_dir)
    gen_dir = fullfile(repo_root, "build", "dev-release", "generated");
end
lib_dir = getenv("N4M_LIB_DIR");
if isempty(lib_dir)
    lib_dir = fullfile(repo_root, "build", "dev-release", "cpp", "src");
end

src_files = {
    fullfile(matlab_root, "mex", "n4m_spectral_mex.c"), ...
    fullfile(matlab_root, "mex", "n4m_preprocess_mex.c"), ...
    fullfile(matlab_root, "mex", "n4m_split_mex.c"), ...
    fullfile(matlab_root, "mex", "n4m_method_fit_mex.c"), ...
    fullfile(matlab_root, "mex", "n4m_model_fit_mex.c"), ...
    fullfile(matlab_root, "mex", "n4m_pls_fit_mex.c"), ...
    fullfile(matlab_root, "mex", "n4m_optimizer_mex.cpp"), ...
    fullfile(matlab_root, "mex", "n4m_role_pipeline_mex.cpp"), ...
    fullfile(matlab_root, "mex", "n4m_multimodal_pipeline_mex.cpp"), ...
    fullfile(matlab_root, "mex", "n4m_version_mex.c") ...
};

if nargin > 0
    if ischar(targets)
        targets = {targets};
    end
    if ~iscell(targets) || isempty(targets) || ~isvector(targets)
        error('n4m:build_mex', 'targets must be a nonempty cell vector of MEX names');
    end
    selected = cell(1, numel(targets));
    for k = 1:numel(targets)
        if ~ischar(targets{k}) || size(targets{k}, 1) ~= 1
            error('n4m:build_mex', 'each target must be a character row string');
        end
        found = false;
        for j = 1:numel(src_files)
            [~, name, ~] = fileparts(src_files{j});
            if strcmp(targets{k}, name)
                selected{k} = src_files{j};
                found = true;
                break;
            end
        end
        if ~found
            error('n4m:build_mex', 'unknown MEX target: %s', targets{k});
        end
    end
    src_files = selected;
end

out_dir = fullfile(matlab_root, "+n4m");

for k = 1:length(src_files)
    src = src_files{k};
    [~, out_name, ~] = fileparts(src);
    out_base = fullfile(out_dir, out_name);
    fprintf("Compiling %s ...\n", src);
    if exist("OCTAVE_VERSION", "builtin")
        % Octave path.
        cmd = sprintf('mkoctfile --mex -o %s %s %s %s %s -ln4m %s', ...
            shell_quote(out_base), shell_quote(src), ...
            shell_quote(['-I' inc_dir]), shell_quote(['-I' gen_dir]), ...
            shell_quote(['-L' lib_dir]), shell_quote(['-Wl,-rpath,' lib_dir]));
        status = system(cmd);
        if status ~= 0
            error("mkoctfile failed for %s", src);
        end
    else
        % MATLAB.
        mex('-output', out_base, ['-I' inc_dir], ['-I' gen_dir], ...
            ['-L' lib_dir], '-ln4m', ['-Wl,-rpath,' lib_dir], src);
    end
end
fprintf("Done. MEX files installed under %s\n", out_dir);
end

function quoted = shell_quote(value)
% POSIX shell argument quoting preserves paths, including spaces and quotes.
quote = char(39);
escape = [quote char(34) quote char(34) quote];
quoted = [quote strrep(value, quote, escape) quote];
end
