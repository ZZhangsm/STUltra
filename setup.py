from setuptools import Command, find_packages, setup

__lib_name__ = "STIntg"
__lib_version__ = "0.1.0"
__description__ = "Integrating spatial transcriptomics data across different conditions, technologies, and developmental stages"
__url__ = "https://github.com/ZZhangsm/STIntg"
__author__ = "Songming Zhang"
__author_email__ = "sm.zhang@smail.nju.edu.cn"
__license__ = "MIT"
__keywords__ = ["spatial transcriptomics", "data integration", "Graph autoencoder", "spatial domain", "batch correction"]
__requires__ = ["requests",]

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['STIntg'],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
)