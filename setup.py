from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name='edwar',         # How you named your package folder (MyLib)
  packages=find_packages(),   # Chose the same as "name"
  version='1.0',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license
  include_package_data=False,
  description='signal Error Detection for WBSN And data1 Recovery',   # Give a short description about your library
  long_description=long_description,
  long_description_content_type="text/markdown",
  author='Miguel Merino',                   # Type in your name
  author_email='miguel_dreimal_30@hotmail.com',      # Type in your E-Mail
  url='https://github.com/greenlsi/edwar',   # Provide either the link to your github or to your website
  keywords=['EDA', 'RECOVERY', 'SIGNAL'],   # Keywords that define your package best
  install_requires=[
          'cvxopt == 1.2.6', 
          'configparser==6.0.1',
          'datetime==5.4',
          'joblib==1.3.2'
          'ledapy==1.2.1',
          'matplotlib==3.7.5',
          'mysql-connector-python==8.3.0',
          'numpy==1.24.4',
          'pandas==2.0.3',
          'pycryptodome==3.20.0',
          'PyWavelets==1.4.1',
          'scipy==1.10.1',          
          'scikit-learn==1.3.2',
  ], # 'mysql' used to be neceessary
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the
                                            # current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      # Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
