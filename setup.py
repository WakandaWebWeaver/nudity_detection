from setuptools import setup, find_packages

setup(
    name='nudity_detection',
    version='0.1',
    description='A package for detecting nudity in images',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/WakandaWebWeaver',
    author='Esvin Joshua',
    author_email='Joshua.Esvin312@gmail.com',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'tensorflow',
        'numpy',
        'opencv-python',
        'huggingface_hub'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
