from setuptools import find_packages, setup

setup(
    name='Lucid Torch',
    version='1.0.0',
    description='Feature visualization for pytorch',
    url='https://github.com/HealthML/lucid-torch',
    author='Martin Graf, Matthias Kirchler',
    packages=find_packages(),
    python_requires='==3.8.*',
    install_requires=['torch==1.5.0',
                      "numpy==1.18.1",
                      "pandas==1.0.1",
                      "pillow==8.1.1",
                      "scipy==1.4.1",
                      "torchvision==0.6.0",
                      "tqdm==4.42.1",
                      "scikit-image==0.16.2",
                      "matplotlib==3.1.3",
                      "kornia==0.3.*",
                      "moviepy==1.0.1",
                      "ipython==7.12.0"
                      ]
)
