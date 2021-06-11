from setuptools import setup, find_packages, version


setup(
    name='koras',
    version='0.0.0',
    packages=find_packages(exclude=['koras_examples']),
    install_requires=[
        'matplotlib==3.4.2',
        'numpy~=1.19.2',
        'scikit-learn==0.24.2',
        'tensorflow==2.5.0',
        'torch==1.8.1',
        'torchvision==0.9.1',
        'tqdm==4.61.0'
    ]
)
