function pls4all_bench_emit(stats, last_preds, pred_path, versions)
% Persist predictions as .npy and emit a JSON record (last stdout line).
dir_ = fileparts(pred_path);
if ~isempty(dir_) && exist(dir_, "dir") == 0
    mkdir(dir_);
end
write_npy_f64(last_preds, pred_path);

% Hand-roll JSON (Octave/MATLAB lack a uniform jsonencode pre-R2017).
function s = jenc(v)
    if islogical(v)
        if v, s = "true"; else, s = "false"; end
    elseif isnumeric(v) && isscalar(v)
        s = sprintf("%.15g", v);
    elseif ischar(v) || (isstring(v) && isscalar(v))
        s = sprintf("\"%s\"", strrep(char(v), '"', '\"'));
    elseif iscell(v) || (isnumeric(v) && ~isscalar(v))
        parts = arrayfun(@(x) jenc(x), v, "UniformOutput", false);
        s = sprintf("[%s]", strjoin(parts, ","));
    elseif isstruct(v)
        fn = fieldnames(v);
        parts = cell(1, length(fn));
        for k = 1:length(fn)
            parts{k} = sprintf("\"%s\":%s", fn{k}, jenc(v.(fn{k})));
        end
        s = sprintf("{%s}", strjoin(parts, ","));
    else
        s = "null";
    end
end

rec = stats;
rec.predictions_path = pred_path;
rec.versions = versions;
disp(jenc(rec))
end
