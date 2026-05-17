function write_npy_f64(arr, path)
% Tiny numpy .npy v1.0 writer for 1-D float64 arrays. Compatible with
% numpy.load(path).
arr = double(arr(:));
header_body = sprintf("{'descr': '<f8', 'fortran_order': False, 'shape': (%d,), }", ...
                       numel(arr));
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
fwrite(fid, arr, "double", 0, "ieee-le");
fclose(fid);
end
