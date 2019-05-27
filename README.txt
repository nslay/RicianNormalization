# 
# Copyright (c) 2019 Nathan Lay (enslay@gmail.com)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#
# Nathan Lay
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# National Institutes of Health
# March 2017
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#######################################################################
# Introduction                                                        #
#######################################################################
RicianNormalization is a tool that reads T2W images (as DICOM or Nifti, 
MetaIO, etc...) and potentially other magnitude MR images, fits a Rice 
distribution to the pixel intensities and normalizes the pixels based 
on the standard score using the fit Rice mean and standard deviation.

The method is loosely based on this work:
Lemaître, Guillaume, et al. "Normalization of t2w-mri prostate images 
using rician a priori." Medical Imaging 2016: Computer-Aided Diagnosis. 
Vol. 9785. International Society for Optics and Photonics, 2016.

Though the implementation is vastly different employing a maximum log
likelihood scheme to fit the Rice distribution.

The source code is partly based on ComputeBValue from which this README 
is also partly based.
https://github.com/nslay/ComputeBValue/

#######################################################################
# Installing                                                          #
#######################################################################
If a precompiled version is available for your operating system, either
extract the archive where it best suits you, or copy the executable to
the desired location.

Once installed, the path to RicianNormalization should be added to PATH.

Windows: Right-click on "Computer", select "Properties", then select
"Advanced system settings." In the "System Properties" window, look
toward the bottom and click the "Environment Variables" button. Under
the "System variables" list, look for "Path" and select "Edit." Append
the ";C:\Path\To\Folder" where "C:\Path\To\Folder\RicianNormalization.exe"
is the path to the executable. Click "OK" and you are done.

Linux: Use a text editor to open the file ~/.profile or ~/.bashrc
Add the line export PATH="${PATH}:/path/to/folder" where
/path/to/folder/RicianNormalization is the path to the executable. Save
the file and you are done.

RicianNormalization can also be compiled from source. Instructions are
given in the "Building from Source" section.

#######################################################################
# Usage                                                               #
#######################################################################
Once installed, RicianNormalization must be run from the command line. 
In Windows this is accomplished using Command Prompt or PowerShell.
Unix-like environments include terminals where commands may be issued.

WINDOWS TIP: Command Prompt can be launched conveniently in a folder
holding shift and right clicking in the folder's window and selecting
"Open command window here."

RicianNormalization reads in one input file or DICOM folder, normalizes 
it and then writes out the normalized image in an image file (e.g. Nifti 
or MetaIO) or DICOM series.

As a quickstart example:

RicianNormalization C:\Path\To\2-t2tsetraGrappa3-70574 normalized

will fit the Rice distribution to ProstateX-0191's T2W image, normalize
the image and then write a DICOM series to the folder 'normalized'

The program's output will look something like the following:

--- Begin snippet ---
Info: Normalizing ...
Solving for initial guess with bisection method...
fa = -327.663, fb = 24.5955
s = 140.581, fs = 46.9242
...
s = 34.0533, fs = -0.00556499
s = 34.0534, fs = -0.000763149

Initial nu = 394.553, sigma = 34.0534
Sample mean = 327.663, rice mean = 327.662
Sample 2nd moment = 157991, rice 2nd moment = 157991

Initial x = 394.553 34.0534
At X0         0 variables are exactly at the bounds
loss = 70.9392, x = 394.553 34.0534, grad = 0.24536 -5.82141
At iterate     0    f=       70.939    |proj g|=       5.8214
loss = 65.408, x = 394.511 35.0525, grad = 0.227606 -5.24555
loss = 48.1231, x = 394.342 39.049, grad = 0.170359 -3.52175
At iterate     1    f=       48.123    |proj g|=       3.5217
...
loss = 6.58556, x = 119.387 273.428, grad = 7.50053e-06 4.4826e-06
At iterate    20    f=       6.5856    |proj g|=   7.5005e-06
Iter   20, Eval   22: Best F =    6.58556
           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *
   N   Tit  Tnf  Tnint  Skip  Nact     Projg        F
    2   20   22     20    0     0     7.5e-06        6.59
F = 6.58556
            CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL
 Cauchy                time          0 seconds.
 Subspace minimization time          0 seconds.
 Line search           time          0 seconds.
 Total User time          0 seconds.

Final x = 119.387 273.428, rice mean = 358.833 rice std = 187.131
Info: Saving DICOM series to 'normalized' ...
--- End snippet ---

And this will produce a DICOM folder with the following file hierarchy

normalized/
+-- 1.dcm
+-- 2.dcm
+-- 3.dcm
+-- 4.dcm
...
+-- 18.dcm

If the output path includes a file extansion (e.g. .nii, .mha), then
RicianNormalization will write an image file. Otherwise, it will assume 
the output is a folder and write a DICOM series (and create the folder 
automatically).

Lastly, RicianNormalization provides the below usage message when
provided with the -h flag or no arguments. It has no additional
options yet.

Usage: RicianNormalization [-h] inputPath outputPath

#######################################################################
# Building from Source                                                #
#######################################################################
To build RicianNormalization from source, you will need a recent version 
of CMake, a C++11 compiler, and InsightToolkit version 4 or later.

First extract the source code somewhere. Next create a separate
directory elsewhere. This will serve as the build directory. Run CMake
and configure the source and build directories as chosen. More
specifically

On Windows:
- Run cmake-gui (Look in Start Menu) and proceed from there.

On Unix-like systems:
- From a terminal, change directory to the build directory and then
run:

ccmake /path/to/source/directory

In both cases, "Configure." If you encounter an error, set ITK_DIR
and then run "Configure" again. Then select "Generate." On Unix-like
systems, you may additionally want to set CMAKE_BUILD_TYPE to "Release"

NOTE: ITK_DIR should be set to the cmake folder in the ITK lib
folder. For example: /path/to/ITK/lib/cmake/ITK-4.13/

Visual Studio:
- Open the solution in the build directory and build RicianNormalization.
Make sure you select "Release" mode.

Unix-like systems:
- Run the "make" command.

RicianNormalization has been successfully built and tested with:
Microsoft Visual Studio 2017 on Windows 10 Professional
Clang 8.0.0 on FreeBSD 12.0-STABLE

using ITK versions:
ITK 4.13

#######################################################################
# Caveats                                                             #
#######################################################################
Using absolute paths to Windows shares (i.e. \\name\folder) could cause
problems since BaseName() and DirName() have not yet implemented
parsing these kinds of paths.

