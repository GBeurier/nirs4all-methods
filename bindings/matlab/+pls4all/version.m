function v = version()
% pls4all.version  Return the libn4m runtime version string.
%   "0.85.0+abi.1.13.0" for the snapshot that shipped this binding.
%
% The string is read by the n4m_version_mex shim which calls
% n4m_get_version_string() on the loaded libn4m.

v = pls4all.n4m_version_mex();
end
