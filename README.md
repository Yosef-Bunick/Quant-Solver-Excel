3 WORKS
4 IS FIRST STEP INTO C++ COMPILE, 4 STEP ONE IS CONNECTED BUT VERY BADLY NEEDS FIXING, YOU CAN REMOVE THEM ONCE YOU CREATE A NEW ONE THAT WORKS
5 is 12 seconds long to solve 3
6 is 8 secconds lightning speed




PS C:\Users\yosef> pip show pyinstaller cython setuptools scipy numpy openpyxl pycel | findstr /R "^Name ^Version"
Name: pyinstaller
Version: 6.20.0
Name: Cython
Version: 3.2.4
Name: setuptools
Version: 80.9.0
Name: scipy
Version: 1.16.3
Name: numpy
Version: 2.2.6
Name: openpyxl
Version: 3.1.5
Name: pycel
Version: 1.0b30
python ==3.12
PS C:\Users\yosef>




Runtime dependencies
    numpy>=1.24
    scipy>=1.10
    openpyxl>=3.1
    pycel==1.0b30     # exact — newer may break pickle/cache behavior

    Build only (for FastSolver.exe via PyInstaller)
    pyinstaller>=6.6


    Stdlib (no install needed): json, os, sys, time, traceback, pathlib, hashlib, tempfile, subprocess, shutil

    The pycel==1.0b30 pin matters 
