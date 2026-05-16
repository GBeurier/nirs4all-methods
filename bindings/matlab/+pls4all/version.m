function v = version()
% pls4all.version  Return the libp4a runtime version string.
%   "0.85.0+abi.1.13.0" for the snapshot that shipped this binding.
%
% The string is read by the p4a_version_mex shim which calls
% p4a_get_version_string() on the loaded libp4a.

v = pls4all.p4a_version_mex();
end
