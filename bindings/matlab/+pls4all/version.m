function v = version()
% pls4all.version  Return the libn4m runtime version string.
%   "1.0.2+abi.2.0.0" for the snapshot that shipped this binding.
%
% The string is read by the n4m_version_mex shim which calls
% n4m_get_version_string() on the loaded libn4m.

v = pls4all.n4m_version_mex();
end
