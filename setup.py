# # -*- coding: utf-8 -*-
# from setuptools import setup

# packages = \
# ['casino']

# package_data = \
# {'': ['*']}

# install_requires = \
# ['cython>=0.29.35,<0.30.0', 'numba>=0.57.0,<0.58.0', 'numpy>=1.24.3,<2.0.0']

# setup_kwargs = {
#     'name': 'casino',
#     'version': '0.1.0',
#     'description': 'Casino enables fast probabilistic modelling',
#     'long_description': '# Casino Probabilistic Modelling Library\n\nThis is awesome.\n',
#     'author': 'Adolph Vogel',
#     'author_email': 'ajvogel@gmail.com',
#     'maintainer': 'None',
#     'maintainer_email': 'None',
#     'url': 'None',
#     'packages': packages,
#     'package_data': package_data,
#     'install_requires': install_requires,
#     'python_requires': '>=3.11,<3.12',
# }
# from build import *
# build(setup_kwargs)

# setup(**setup_kwargs)


from Cython.Build import cythonize
from setuptools import setup

setup(
    name="ranvar",
    ext_modules=cythonize(
        ["ranvar/vm.py", "ranvar/digest.py"],
        include_path=["ranvar/"],
        force=True,
        annotate=True,
    ),
)
