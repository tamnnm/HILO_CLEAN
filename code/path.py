import os
import sys

module_path = "/work/users/student6/tam/code/pkg_twcr"
sys.path.append(module_path)
print(sys.path)
if 'pkg_twcr' in sys.modules:
	print('Module load')
else:
	from pkg_twcr import pi
