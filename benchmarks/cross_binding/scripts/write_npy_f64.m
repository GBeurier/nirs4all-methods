function write_npy_f64(arr, path)
% Tiny numpy .npy v1.0 writer for float64. Honours matrix shape so that
% a downstream `np.load(path)` reshapes to the row-major (C-order) view
% Python emits — MATLAB / Octave store matrices column-major, so for
% 2-D inputs we transpose before flattening.
if ndims(arr) > 2
    error("write_npy_f64: only 1-D / 2-D arrays supported (got %s)", ...
        mat2str(size(arr)));
end
if isvector(arr)
    flat = double(arr(:));
    shape_str = sprintf("(%d,)", numel(flat));
else
    rows = size(arr, 1);
    cols = size(arr, 2);
    flat = double(reshape(arr.', [], 1));
    shape_str = sprintf("(%d, %d)", rows, cols);
end
header_body = sprintf("{'descr': '<f8', 'fortran_order': False, 'shape': %s, }", ...
                       shape_str);
magic = uint8([147 78 85 77 80 89]);   % \x93NUMPY
version = uint8([1 0]);
base_len = numel(magic) + numel(version) + 2 + length(char(header_body)) + 1;
pad = mod(64 - mod(base_len, 64), 64);
header_full = sprintf("%s%s\n", char(header_body), repmat(" ", 1, pad));
hl = length(header_full);
header_len_le = uint8([mod(hl, 256), floor(hl / 256)]);

fid = fopen(path, "wb");
if fid < 0
    error("write_npy_f64: cannot open %s", path);
end
fwrite(fid, magic, "uint8");
fwrite(fid, version, "uint8");
fwrite(fid, header_len_le, "uint8");
fwrite(fid, uint8(double(header_full)), "uint8");
fwrite(fid, flat, "double", 0, "ieee-le");
fclose(fid);
end
