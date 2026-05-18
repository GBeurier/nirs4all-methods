function build_mex()
% build_mex  Compile the pls4all MEX shims against libp4a.
%
% Set the env vars PLS4ALL_INCLUDE_DIR, PLS4ALL_LIB_DIR before
% running, or pass them as setenv calls before invoking this script.
% Falls back to the dev-release CMake build inside the repo.

matlab_root = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(matlab_root));
inc_dir = getenv("PLS4ALL_INCLUDE_DIR");
if isempty(inc_dir)
    inc_dir = fullfile(repo_root, "cpp", "include");
end
gen_dir = getenv("PLS4ALL_GENERATED_DIR");
if isempty(gen_dir)
    gen_dir = fullfile(repo_root, "build", "dev-release", "generated");
end
lib_dir = getenv("PLS4ALL_LIB_DIR");
if isempty(lib_dir)
    lib_dir = fullfile(repo_root, "build", "dev-release", "cpp", "src");
end

src_files = {
    fullfile(matlab_root, "mex", "p4a_method_fit_mex.c"), ...
    fullfile(matlab_root, "mex", "p4a_model_fit_mex.c"), ...
    fullfile(matlab_root, "mex", "p4a_pls_fit_mex.c"), ...
    fullfile(matlab_root, "mex", "p4a_version_mex.c") ...
};

out_dir = fullfile(matlab_root, "+pls4all");

cflags = sprintf("-I%s -I%s", inc_dir, gen_dir);
ldflags = sprintf("-L%s -lp4a -Wl,-rpath,%s", lib_dir, lib_dir);

for k = 1:length(src_files)
    src = src_files{k};
    [~, out_name, ~] = fileparts(src);
    out_base = fullfile(out_dir, out_name);
    fprintf("Compiling %s ...\n", src);
    if exist("OCTAVE_VERSION", "builtin")
        % Octave path.
        cmd = sprintf("mkoctfile --mex -o %s %s %s %s", ...
                       out_base, src, cflags, ldflags);
        status = system(cmd);
        if status ~= 0
            error("mkoctfile failed for %s", src);
        end
    else
        % MATLAB.
        mex("-output", out_base, ...
            cflags, ldflags, src);
    end
end
fprintf("Done. MEX files installed under %s\n", out_dir);
end
