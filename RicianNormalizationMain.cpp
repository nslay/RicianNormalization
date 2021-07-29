/*-
 * Copyright (c) 2019 Nathan Lay (enslay@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdlib>
#include <iostream>
#include <type_traits>
#include "Common.h"
#include "RicianNormalization.h"

#include "bsdgetopt.h"

#include "itkImageIOFactory.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"

#if ITK_VERSION_MAJOR > 4
using IOFileModeEnum = itk::IOFileModeEnum;
using IOComponentEnum = itk::IOComponentEnum;
using IOPixelEnum = itk::IOPixelEnum;
#else // ITK_VERSION_MAJOR <= 4
using IOFileModeEnum = itk::ImageIOFactory;
using IOComponentEnum = itk::ImageIOBase;
using IOPixelEnum = itk::ImageIOBase;
#endif // ITK_VERSION_MAJOR > 4

void Usage(const char *p_cArg0) {
  std::cerr << "Usage: " << p_cArg0 << " [-h] inputPath outputPath" << std::endl;
  exit(1);
}

template<typename PixelType, unsigned int Dimension>
bool NormalizeTemplate(const std::string &strImagePath, const std::string &strOutputPath);

bool Normalize(const std::string &strImagePath, const std::string &strOutputPath);

int main(int argc, char **argv) {
  const char * const p_cArg0 = argv[0];

  int c = 0;
  while ((c = getopt(argc, argv, "h")) != -1) {
    switch (c) {
      case 'h':
        Usage(p_cArg0);
        break;
      case '?':
      default:
        Usage(p_cArg0);
        break;
    }
  }

  argc -= optind;
  argv += optind;

  if (argc != 2)
    Usage(p_cArg0);

  if (!Normalize(argv[0], argv[1])) {
    std::cerr << "Error: Failed to normalize image." << std::endl;
    return -1;
  }

  return 0;
}

template<typename PixelType, unsigned int Dimension>
bool NormalizeTemplate(const std::string &strImagePath, const std::string &strOutputPath) {
  typedef itk::Image<PixelType, Dimension> ImageType;

  typename ImageType::Pointer p_clImage;

  constexpr const double dScale = std::is_integral<PixelType>::value ? 100.0 : 1.0;

  const bool bInputIsSeries = IsFolder(strImagePath);
  const bool bOutputIsSeries = GetExtension(strOutputPath).empty();

  if (!bInputIsSeries && bOutputIsSeries) {
    std::cerr << "Error: Cannot save DICOM series from non-DICOM image." << std::endl;
    return false;
  }

  if (bInputIsSeries)
    p_clImage = LoadDicomImage<PixelType, Dimension>(strImagePath);
  else
    p_clImage = LoadImg<PixelType, Dimension>(strImagePath);

  if (!p_clImage) {
    std::cerr << "Error: Failed to load image '" << strImagePath << "'." << std::endl;
    return false;
  }

  RicianNormalization<ImageType> clNormalizer;

  std::cout << "Info: Normalizing ..." << std::endl;

  if (!clNormalizer.Normalize(p_clImage, dScale)) {
    std::cerr << "Error: Failed to normalize image." << std::endl;
    return false;
  }

  if (bOutputIsSeries) {
    MkDir(strOutputPath);

    std::cout << "Info: Saving DICOM series to '" << strOutputPath << "' ..." << std::endl;

    return SaveDicomImage(p_clImage.GetPointer(), strOutputPath);
  }

  std::cout << "Info: Saving image to '" << strOutputPath << "' ..." << std::endl;

  return SaveImg(p_clImage.GetPointer(), strOutputPath);
}

bool Normalize(const std::string &strImagePath, const std::string &strOutputPath) {
  itk::ImageIOBase::Pointer p_clImageIO;
  unsigned int uiDimension = 0;
  auto pixelComponent = IOComponentEnum::UNKNOWNCOMPONENTTYPE;

  if (IsFolder(strImagePath)) {
    // DICOM
    itk::GDCMSeriesFileNames::Pointer p_clFileNames = itk::GDCMSeriesFileNames::New();

    p_clFileNames->SetDirectory(strImagePath);
    p_clFileNames->SetUseSeriesDetails(false); // No weird concatenation of crap
    const auto &vSeriesUIDs = p_clFileNames->GetSeriesUIDs();

    if (vSeriesUIDs.empty()) {
      std::cerr << "Error: No DICOM files found." << std::endl;
      return false;
    }

    const auto &vFileNames = p_clFileNames->GetFileNames(vSeriesUIDs[0]); // Use first series UID

    itk::GDCMImageIO::Pointer p_clDICOMImageIO = itk::GDCMImageIO::New();

    p_clDICOMImageIO->SetKeepOriginalUID(true);
    p_clDICOMImageIO->SetLoadPrivateTags(true);
    p_clDICOMImageIO->SetFileName(vFileNames[0]);

    p_clImageIO = p_clDICOMImageIO;

    try {
      p_clImageIO->ReadImageInformation();
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << "Error: Failed to read image meta data: " << e << std::endl;
      return false;
    }

    pixelComponent = p_clDICOMImageIO->GetInternalComponentType(); // Bypass all the fancy transformations...
    uiDimension = vFileNames.size() > 1 ? 3 : 2; // p_clDICOMImageIO->GetNumberOfDimensions() is probably 2
  }
  else {
    p_clImageIO = itk::ImageIOFactory::CreateImageIO(strImagePath.c_str(), IOFileModeEnum::ReadMode);

    if (!p_clImageIO) {
      std::cerr << "Error: Could not determine ImageIO for file path '" << strImagePath << "'." << std::endl;
      return false;
    }

    p_clImageIO->SetFileName(strImagePath);

    try {
      p_clImageIO->ReadImageInformation();
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << "Error: Failed to read image meta data: " << e << std::endl;
      return false;
    }

    pixelComponent = p_clImageIO->GetComponentType();
    uiDimension = p_clImageIO->GetNumberOfDimensions();
  }

  if (p_clImageIO->GetNumberOfComponents() != 1 || p_clImageIO->GetPixelType() != IOPixelEnum::SCALAR) {
    std::cerr << "Error: Cannot operate on images comprised of more than 1 pixel component." << std::endl;
    return false;
  }

  switch (uiDimension) {
  case 2:
    switch (pixelComponent) {
    case IOComponentEnum::UCHAR:
      return NormalizeTemplate<unsigned char, 2>(strImagePath, strOutputPath);
    case IOComponentEnum::CHAR:
      return NormalizeTemplate<char, 2>(strImagePath, strOutputPath);
    case IOComponentEnum::USHORT:
      return NormalizeTemplate<unsigned short, 2>(strImagePath, strOutputPath);
    case IOComponentEnum::SHORT:
      return NormalizeTemplate<short, 2>(strImagePath, strOutputPath);
    case IOComponentEnum::FLOAT:
      return NormalizeTemplate<float, 2>(strImagePath, strOutputPath);
    case IOComponentEnum::DOUBLE:
      return NormalizeTemplate<double, 2>(strImagePath, strOutputPath);
    default:
      std::cerr << "Error: Unsupported pixel type: " << (int)pixelComponent << '.' << std::endl;
      return false;
    }
    break;
  case 3:
    switch (pixelComponent) {
    case IOComponentEnum::UCHAR:
      return NormalizeTemplate<unsigned char, 3>(strImagePath, strOutputPath);
    case IOComponentEnum::CHAR:
      return NormalizeTemplate<char, 3>(strImagePath, strOutputPath);
    case IOComponentEnum::USHORT:
      return NormalizeTemplate<unsigned short, 3>(strImagePath, strOutputPath);
    case IOComponentEnum::SHORT:
      return NormalizeTemplate<short, 3>(strImagePath, strOutputPath);
    case IOComponentEnum::FLOAT:
      return NormalizeTemplate<float, 3>(strImagePath, strOutputPath);
    case IOComponentEnum::DOUBLE:
      return NormalizeTemplate<double, 3>(strImagePath, strOutputPath);
    default:
      std::cerr << "Error: Unsupported pixel type: " << (int)pixelComponent << '.' << std::endl;
      return false;
    }
    break;
  default:
    std::cerr << "Error: Cannot operate on " << uiDimension << "D images." << std::endl;
    return false;
  }

  return false; // Not reached
}
