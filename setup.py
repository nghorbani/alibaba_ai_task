# -*- coding: utf-8 -*-
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2021.11.19
from setuptools import setup, find_packages
from glob import glob

setup(name='alibaba_ai_task',
      version='0.1.0',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      include_package_data=True,
      data_files=[
          ('soma/support_data', glob('support_data/*.*')),
          ('soma/support_data/github_files', glob('support_data/github_files/*.*')),
          ('soma/support_data/tests', glob('support_data/tests/*.*')),
          ('soma/support_data/conf', glob('support_data/conf/*.*'))
      ],

      author='Nima Ghorbani',
      author_email='nima.gbani@gmail.com',
      maintainer='Nima Ghorbani',
      maintainer_email='nima.gbani@gmail.com',
      url='https://github.com/nghorbani/alibaba_ai_task',
      description='Future 7-Days Flight Price Prediction',
      license='See LICENSE.txt',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      install_requires=[],
      dependency_links=[],
      classifiers=[
          "Intended Audience :: Research",
          "Natural Language :: English",
          "Operating System :: POSIX",
          "Operating System :: POSIX :: BSD",
          "Operating System :: POSIX :: Linux",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.7", ],
      )
