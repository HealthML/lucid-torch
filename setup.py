from setuptools import setup, find_packages

setup(
    name='Lucid Torch',
    version='1.0.0',
    description='Feature visualization and attribution for pytorch',
    url='https://github.com/mkirchler/feature-visualization',
    author='Hasso Plattner Institute Chair of Digital Health - Machine Learning',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=['torch>=1.5', "numpy", "pandas", "pillow",
                      "scipy", "torchvision", "tqdm", "scikit-image",
                      "matplotlib", "kornia==0.3.*", "moviepy",
                      "ipython", "pytest"],
)
