from setuptools import setup, find_packages

setup(
    name='Lucid Torch',
    version='1.0.0',
    description='Feature visualization and attribution for pytorch',
    url='https://github.com/mkirchler/feature-visualization',
    author='Hasso Plattner Institute Chair of Digital Health - Machine Learning',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=['torch>=1.5', "numpy>=1.18.1", "pandas>=1.0.1", "pillow>=7.0.0",
                      "scipy>=1.4.1", "torchvision>=0.6.0", "tqdm>=4.42.1",
                      "scikit-image>=0.16.2", "matplotlib>=3.1.3", "kornia==0.3.*",
                      "moviepy>=1.0.1", "ipython>=7.12.0", "pytest>=5.3.5"],
)
