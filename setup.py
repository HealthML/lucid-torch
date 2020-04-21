from setuptools import setup, find_packages

setup(
    name='lucid torch',
    version='1.0.0',
    description='Feature visualization and attribution for pytorch',
    url='https://github.com/mkirchler/feature-visualization',
    author='Hasso Plattner Institute Chair of Digital Health - Machine Learning',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=['torch', "numpy", "pandas", "pillow",
                      "scipy", "torchvision", "tqdm", "scikit-image", "matplotlib"],
)
