/*-
 * Copyright (c) 2018 Nathan Lay (enslay@gmail.com)
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
 
/*-
 * Nathan Lay
 * Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
 * National Institutes of Health
 * March 2017
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

#ifndef COMMON_H
#define COMMON_H

#include <cstring>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkNumericSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "vnl/vnl_vector_fixed.h"
#include "vnl/vnl_matrix_fixed.h"
#include "vnl/vnl_cross.h"

#include "gdcmCSAHeader.h"
#include "gdcmCSAElement.h"
#include "gdcmUIDGenerator.h"

void Trim(std::string &strString);

template<typename ValueType>
ValueType FromString(const std::string &strValue, const ValueType &failValue);

template<>
std::string FromString<std::string>(const std::string &strValue, const std::string &);

template<typename ValueType>
std::vector<ValueType> SplitString(const std::string &strValue, const std::string &strDelim);

template<>
std::vector<std::string> SplitString<std::string>(const std::string &strValue, const std::string &strDelim);

bool ParseITKTag(const std::string &strKey, uint16_t &ui16Group, uint16_t &ui16Element);

void CopyStringMetaData(itk::MetaDataDictionary &clNew, const itk::MetaDataDictionary &clOriginal);

bool FileExists(const std::string &strPath);
bool IsFolder(const std::string &strPath);
bool RmDir(const std::string &strPath);
bool MkDir(const std::string &strPath);
bool Unlink(const std::string &strPath);
bool Copy(const std::string &strFrom, const std::string &strTo, bool bReplace = false);
bool Rename(const std::string &strFrom, const std::string &strTo, bool bReplace = false);
void USleep(unsigned int uiMicroSeconds);

std::string BaseName(std::string strPath);
std::string DirName(std::string strPath);
std::string GetExtension(const std::string &strPath);
std::string StripExtension(const std::string &strPath);
std::string StripTrailingDelimiters(std::string strPath);

// Wrap ITK itk::ExposeMetaData to do lookup AND conversion
template<typename ValueType>
bool ExposeStringMetaData(const itk::MetaDataDictionary &clTags, const std::string &strKey, ValueType &value);

template<>
bool ExposeStringMetaData<std::string>(const itk::MetaDataDictionary &clTags, const std::string &strKey, std::string &strValue);

// Delimits by backslash '\'
template<>
bool ExposeStringMetaData<std::vector<float>>(const itk::MetaDataDictionary &clTags, const std::string &strKey, std::vector<float> &vValues);

template<>
bool ExposeStringMetaData<std::vector<double>>(const itk::MetaDataDictionary &clTags, const std::string &strKey, std::vector<double> &vValues);

template<>
bool ExposeStringMetaData<gdcm::CSAHeader>(const itk::MetaDataDictionary &clTags, const std::string &strKey, gdcm::CSAHeader &clCSAHeader);

template<typename ValueType>
void EncapsulateStringMetaData(itk::MetaDataDictionary &clTags, const std::string &strKey, const ValueType &value);

template<>
void EncapsulateStringMetaData<std::string>(itk::MetaDataDictionary &clTags, const std::string &strKey, const std::string &strValue);

template<>
void EncapsulateStringMetaData<std::vector<float>>(itk::MetaDataDictionary &clTags, const std::string &strKey, const std::vector<float> &vValues);

template<>
void EncapsulateStringMetaData<std::vector<double>>(itk::MetaDataDictionary &clTags, const std::string &strKey, const std::vector<double> &vValues);

template<>
void EncapsulateStringMetaData<itk::Point<float, 3>>(itk::MetaDataDictionary &clTags, const std::string &strKey, const itk::Point<float, 3> &clPoint);

template<>
void EncapsulateStringMetaData<itk::Point<float, 2>>(itk::MetaDataDictionary &clTags, const std::string &strKey, const itk::Point<float, 2> &clPoint);

template<>
void EncapsulateStringMetaData<itk::Point<double, 3>>(itk::MetaDataDictionary &clTags, const std::string &strKey, const itk::Point<double, 3> &clPoint);

template<>
void EncapsulateStringMetaData<itk::Point<double, 2>>(itk::MetaDataDictionary &clTags, const std::string &strKey, const itk::Point<double, 2> &clPoint);

template<>
void EncapsulateStringMetaData<itk::Matrix<float, 3, 3>>(itk::MetaDataDictionary &clTags, const std::string &strKey, const itk::Matrix<float, 3, 3> &value);

template<>
void EncapsulateStringMetaData<itk::Matrix<double, 3, 3>>(itk::MetaDataDictionary &clTags, const std::string &strKey, const itk::Matrix<double, 3, 3> &value);

template<typename ValueType>
bool ExposeCSAMetaData(gdcm::CSAHeader &clHeader, const std::string &strKey, ValueType &value);

template<>
bool ExposeCSAMetaData<std::string>(gdcm::CSAHeader &clHeader, const std::string &strKey, std::string &strValue);

// Needed for sorting the B value dictionaries and writing DICOM
template<typename RealType>
bool GetOrientationMatrix(const itk::MetaDataDictionary &clDicomTags, vnl_matrix_fixed<RealType, 3, 3> &clOrientation);

template<typename RealType>
bool GetOrigin(const itk::MetaDataDictionary &clDicomTags, vnl_vector_fixed<RealType, 3> &clOrigin);

void SanitizeFileName(std::string &strFileName); // Does NOT operate on paths
void FindFiles(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFiles, bool bRecursive = false);
void FindFolders(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFolders, bool bRecursive = false);
void FindDicomFiles(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFiles, bool bRecursive = false);
void FindDicomFolders(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFolders, bool bRecursive = false);

// Use LoadImg since Windows #defines LoadImage ... lame
template<typename PixelType, unsigned int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer LoadImg(const std::string &strPath);

template<typename PixelType, unsigned int Dimension>
bool SaveImg(const itk::Image<PixelType, Dimension> *p_clImage, const std::string &strPath, bool bCompress = true);

// Append a singleton dimension and possibly set relevant 3D information for DICOM
template<typename PixelType>
typename itk::Image<PixelType, 3>::Pointer PromoteSlice(itk::Image<PixelType, 2> *p_clSlice, bool bDeepCopy = false);

// Save a DICOM slice and keep all UIDs and private tags
template<typename PixelType>
bool SaveDicomSlice(itk::Image<PixelType, 2> *p_clImage, const std::string &strPath, bool bCompress = false);

template<typename PixelType, unsigned int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer LoadDicomImage(const std::string &strPath, const std::string &strSeriesUID = std::string());

// The p_clImage must have a template MetaDataDictionary from a valid DICOM already set!
// NOTE: You are responsible for setting the series number 0020|0011 and series description 0008|103e as well as any other custom tags!
template<typename PixelType, unsigned int Dimension>
bool SaveDicomImage(itk::Image<PixelType, Dimension> *p_clImage, const std::string &strPath, bool bCompress = false);

template<typename ValueType>
ValueType FromString(const std::string &strValue, const ValueType &failValue) {
  std::stringstream valueStream;
  valueStream.str(strValue);

  ValueType value;
  if (!(valueStream >> value))
    return failValue;

  return value;
}

template<>
inline std::string FromString<std::string>(const std::string &strValue, const std::string &) {
  return strValue;
}

template<typename ValueType>
std::vector<ValueType> SplitString(const std::string &strValue, const std::string &strDelim) {
  std::vector<std::string> vTokens = SplitString<std::string>(strValue, strDelim);

  std::vector<ValueType> vValues;
  vValues.reserve(vTokens.size());

  std::stringstream valueStream;

  for (std::string &strToken : vTokens) {
    valueStream.clear();
    valueStream.str(strToken);

    ValueType value;

    if (!(valueStream >> value))
      return std::vector<ValueType>();

    vValues.emplace_back(value);
  }

  return vValues;
}

template<typename ValueType>
bool ExposeStringMetaData(const itk::MetaDataDictionary &clTags, const std::string &strKey, ValueType &value) {
  std::string strValue;
  if (!ExposeStringMetaData(clTags, strKey, strValue))
    return false;

  std::stringstream valueStream;
  valueStream.str(strValue);

  if (!(valueStream >> value))
    return false;

  return true;
}

template<>
inline bool ExposeStringMetaData<std::string>(const itk::MetaDataDictionary &clTags, const std::string &strKey, std::string &strValue) { 
  return itk::ExposeMetaData(clTags, strKey, strValue);
}

template<typename ValueType>
void EncapsulateStringMetaData(itk::MetaDataDictionary &clTags, const std::string &strKey, const ValueType &value) {
  std::stringstream valueStream;
  valueStream << value;
  EncapsulateStringMetaData(clTags, strKey, valueStream.str());
}

template<>
inline void EncapsulateStringMetaData<std::string>(itk::MetaDataDictionary &clTags, const std::string &strKey, const std::string &strValue) {
  return itk::EncapsulateMetaData(clTags, strKey, strValue);
}

template<typename ValueType>
bool ExposeCSAMetaData(gdcm::CSAHeader &clHeader, const std::string &strKey, ValueType &value) {
  if (!clHeader.FindCSAElementByName(strKey.c_str()))
    return false;

  const gdcm::CSAElement &clElement = clHeader.GetCSAElementByName(strKey.c_str());
  const gdcm::ByteValue * const p_clByteValue = clElement.GetByteValue();

  if (p_clByteValue == nullptr || p_clByteValue->GetLength() != sizeof(ValueType))
    return false;

  return p_clByteValue->GetBuffer((char *)&value, sizeof(ValueType));
}

template<typename RealType>
bool GetOrientationMatrix(const itk::MetaDataDictionary &clDicomTags, vnl_matrix_fixed<RealType, 3, 3> &clOrientation) {
  std::vector<RealType> vImageOrientationPatient; // 0020|0037

  if (!ExposeStringMetaData(clDicomTags, "0020|0037", vImageOrientationPatient) || vImageOrientationPatient.size() != 6)
    return false;

  vnl_vector_fixed<RealType, 3> clX, clY, clZ;

  clX.copy_in(vImageOrientationPatient.data());
  clY.copy_in(vImageOrientationPatient.data() + 3);
  clZ = vnl_cross_3d(clX, clY);

  clOrientation[0][0] = clX[0];
  clOrientation[1][0] = clX[1];
  clOrientation[2][0] = clX[2];

  clOrientation[0][1] = clY[0];
  clOrientation[1][1] = clY[1];
  clOrientation[2][1] = clY[2];

  clOrientation[0][2] = clZ[0];
  clOrientation[1][2] = clZ[1];
  clOrientation[2][2] = clZ[2];

  return true;
}

template<typename RealType>
bool GetOrigin(const itk::MetaDataDictionary &clDicomTags, vnl_vector_fixed<RealType, 3> &clOrigin) {
  std::vector<RealType> vImagePositionPatient; // 0020|0032

  if (!ExposeStringMetaData(clDicomTags, "0020|0032", vImagePositionPatient) || vImagePositionPatient.size() != clOrigin.size())
    return false;

  clOrigin.copy_in(vImagePositionPatient.data());

  return true;
}

template<typename PixelType, unsigned int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer LoadImg(const std::string &strPath) {
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::ImageFileReader<ImageType> ReaderType;

  typename ReaderType::Pointer p_clReader = ReaderType::New();

  p_clReader->SetFileName(strPath);

  try {
    p_clReader->Update();
  }
  catch (itk::ExceptionObject &e) {
    std::cerr << "Error: " << e << std::endl;
    return typename ImageType::Pointer();
  }

  return p_clReader->GetOutput();
}

template<typename PixelType, unsigned int Dimension>
bool SaveImg(const itk::Image<PixelType, Dimension> *p_clImage, const std::string &strPath, bool bCompress) {
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::ImageFileWriter<ImageType> WriterType;

  typename WriterType::Pointer p_clWriter = WriterType::New();

  p_clWriter->SetFileName(strPath);
  p_clWriter->SetUseCompression(bCompress);
  p_clWriter->SetInput(p_clImage);

  try {
    p_clWriter->Update();
  }
  catch (itk::ExceptionObject &e) {
    std::cerr << "Error: " << e << std::endl;
    return false;
  }

  return true;
}

template<typename PixelType>
typename itk::Image<PixelType, 3>::Pointer PromoteSlice(itk::Image<PixelType, 2> *p_clSlice, bool bDeepCopy) {
  typedef itk::Image<PixelType, 2> ImageType;
  typedef itk::Image<PixelType, 3> ImageType3D;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::SpacingType SpacingType;
  typedef typename ImageType::PointType PointType;
  typedef typename ImageType3D::SizeType SizeType3D;
  typedef typename ImageType3D::SpacingType SpacingType3D;
  typedef typename ImageType3D::PointType PointType3D;
  typedef typename ImageType3D::DirectionType DirectionType3D;
  
  if (!p_clSlice || !p_clSlice->GetBufferPointer())
    return typename ImageType3D::Pointer();
  
  const SizeType &clSize = p_clSlice->GetBufferedRegion().GetSize();
  const SpacingType &clSpacing = p_clSlice->GetSpacing();
  const PointType &clOrigin = p_clSlice->GetOrigin();
  
  SizeType3D clSize3D;
  clSize3D[0] = clSize[0];
  clSize3D[1] = clSize[1];
  clSize3D[2] = 1;
  
  SpacingType3D clSpacing3D;
  clSpacing3D[0] = clSpacing[0];
  clSpacing3D[1] = clSpacing[1];
  clSpacing3D[2] = 1; // Default
  
  PointType3D clOrigin3D;
  clOrigin3D[0] = clOrigin[0];
  clOrigin3D[1] = clOrigin[1];
  clOrigin3D[2] = 0; // Default
  
  itk::MetaDataDictionary clDicomTags = p_clSlice->GetMetaDataDictionary();

  float fSpacingBetweenSlices = 0.0f; // 0018|0088

  if (!ExposeStringMetaData(clDicomTags, "0018|0088", fSpacingBetweenSlices))
    return typename ImageType3D::Pointer(); // Not a DICOM, at least not one with Spacing Between Slices tag
  
  clSpacing3D[2] = fSpacingBetweenSlices;
  
  typename ImageType3D::Pointer p_clSlice3D = ImageType3D::New();
  
  p_clSlice3D->SetRegions(clSize3D);
  p_clSlice3D->SetSpacing(clSpacing3D);
  p_clSlice3D->SetOrigin(clOrigin3D);
  p_clSlice3D->SetMetaDataDictionary(clDicomTags);
  
  if (!bDeepCopy) {
    p_clSlice3D->GetPixelContainer()->SetImportPointer(p_clSlice->GetBufferPointer(), (size_t)clSize[0]*clSize[1], false);
  }
  else {
   const PixelType * const p_inBuffer = p_clSlice->GetBufferPointer();
   p_clSlice3D->Allocate();
   std::copy(p_inBuffer, p_inBuffer + (size_t)clSize[0]*clSize[1], p_clSlice3D->GetBufferPointer());
  }
  
  p_clSlice3D->SetSpacing(clSpacing3D);
  
  // Now pull out all the other 3D tags
  vnl_vector_fixed<float, 3> clVnlOrigin3D;

  if (!GetOrigin(clDicomTags, clVnlOrigin3D))
    return p_clSlice3D; // Uhh?
  
  clOrigin3D[0] = clVnlOrigin3D[0];
  clOrigin3D[1] = clVnlOrigin3D[1];
  clOrigin3D[2] = clVnlOrigin3D[2];
  
  p_clSlice3D->SetOrigin(clOrigin3D);

  vnl_matrix_fixed<float, 3, 3> clVnlR;

  if (!GetOrientationMatrix(clDicomTags, clVnlR))
    return p_clSlice3D; // Uhh?
  
  DirectionType3D clR;
  
  clR(0,0) = clVnlR[0][0];
  clR(1,0) = clVnlR[1][0];
  clR(2,0) = clVnlR[2][0];
  
  clR(0,1) = clVnlR[0][1];
  clR(1,1) = clVnlR[1][1];
  clR(2,1) = clVnlR[2][1];
  
  clR(0,2) = clVnlR[0][2];
  clR(1,2) = clVnlR[1][2];
  clR(2,2) = clVnlR[2][2];
  
  p_clSlice3D->SetDirection(clR);
  
  return p_clSlice3D;
}

template<typename PixelType>
bool SaveDicomSlice(itk::Image<PixelType, 2> *p_clSlice, const std::string &strFileName, bool bCompress) {
  typedef itk::Image<PixelType, 2> ImageType;
  typedef itk::Image<PixelType, 3> ImageType3D;
  typedef itk::GDCMImageIO ImageIOType;
  typedef itk::ImageFileWriter<ImageType> WriterType;
  typedef itk::ImageFileWriter<ImageType3D> WriterType3D;
  
  if (!p_clSlice)
    return false;
  
  typename ImageType3D::Pointer p_clSlice3D = PromoteSlice<PixelType>(p_clSlice);
  
  ImageIOType::Pointer p_clImageIO = ImageIOType::New();
  
  p_clImageIO->KeepOriginalUIDOn();
  p_clImageIO->LoadPrivateTagsOn();
  p_clImageIO->SetUseCompression(bCompress); // Maybe not needed?
  
  if (!p_clSlice3D) {
    // Normal 2D DICOM?
    typename WriterType::Pointer p_clWriter = WriterType::New();
    
    p_clWriter->SetImageIO(p_clImageIO);
    p_clWriter->SetUseCompression(bCompress);
    p_clWriter->SetFileName(strFileName);
    p_clWriter->SetInput(p_clSlice);
    
    try {
      p_clWriter->Update();
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << "Error: " << e << std::endl;
      return false;
    }
    
    return true;
  }
  
  typename WriterType3D::Pointer p_clWriter3D = WriterType3D::New();
  
  p_clWriter3D->SetImageIO(p_clImageIO);
  p_clWriter3D->SetUseCompression(bCompress);
  p_clWriter3D->SetFileName(strFileName);
  p_clWriter3D->SetInput(p_clSlice3D);
  
  try {
    p_clWriter3D->Update();
  }
  catch (itk::ExceptionObject &e) {
    std::cerr << "Error: " << e << std::endl;
    return false;
  }
  
  return true;
}

template<typename PixelType, unsigned int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer LoadDicomImage(const std::string &strPath, const std::string &strSeriesUID) {
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::GDCMImageIO ImageIOType;
  typedef itk::GDCMSeriesFileNames FileNameGeneratorType;

  if (!FileExists(strPath)) // File or folder must exist
    return typename ImageType::Pointer();

  ImageIOType::Pointer p_clImageIO = ImageIOType::New();

  p_clImageIO->LoadPrivateTagsOn();
  p_clImageIO->KeepOriginalUIDOn();

  if (Dimension == 2) {
    // Read a 2D image
    typedef itk::ImageFileReader<ImageType> ReaderType;

    if (IsFolder(strPath)) // Must be a file
      return typename ImageType::Pointer();
    
    if (!p_clImageIO->CanReadFile(strPath.c_str()))
      return typename ImageType::Pointer();

    typename ReaderType::Pointer p_clReader = ReaderType::New();

    p_clReader->SetImageIO(p_clImageIO);
    p_clReader->SetFileName(strPath);

    try {
      p_clReader->Update();
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << "Error: " << e << std::endl;
      return typename ImageType::Pointer();
    }

    typename itk::Image<PixelType, Dimension>::Pointer p_clImage = p_clReader->GetOutput();
    p_clImage->SetMetaDataDictionary(p_clImageIO->GetMetaDataDictionary());

    return p_clImage;
  }

  // Passed a file, read the series UID (ignore the one provided, if any)
  if (!IsFolder(strPath)) {

    if (!p_clImageIO->CanReadFile(strPath.c_str()))
      return typename ImageType::Pointer();

    p_clImageIO->SetFileName(strPath.c_str());

    try {
      p_clImageIO->ReadImageInformation();
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << "Error: " << e << std::endl;
      return typename ImageType::Pointer();
    }

    const itk::MetaDataDictionary &clDicomTags = p_clImageIO->GetMetaDataDictionary();

    std::string strTmpSeriesUID;
    if (!itk::ExposeMetaData(clDicomTags, "0020|000e", strTmpSeriesUID))
      return typename ImageType::Pointer();

    Trim(strTmpSeriesUID);

    return LoadDicomImage<PixelType, Dimension>(DirName(strPath), strTmpSeriesUID); // Call this function again
  }

  FileNameGeneratorType::Pointer p_clFileNameGenerator = FileNameGeneratorType::New();

  // Use the ACTUAL series UID ... not some custom ITK concatenations of lots of junk.
  p_clFileNameGenerator->SetUseSeriesDetails(false); 
  p_clFileNameGenerator->SetDirectory(strPath);

  if (strSeriesUID.empty()) {
    // Passed a folder but no series UID ... pick the first series UID
    const FileNameGeneratorType::SeriesUIDContainerType &vSeriesUIDs = p_clFileNameGenerator->GetSeriesUIDs();

    if (vSeriesUIDs.empty())
      return typename ImageType::Pointer();

    // Use first series UID
    return LoadDicomImage<PixelType, Dimension>(strPath, vSeriesUIDs[0]);
  }

  const FileNameGeneratorType::FileNamesContainerType &vDicomFiles = p_clFileNameGenerator->GetFileNames(strSeriesUID);

  if (vDicomFiles.empty())
    return typename ImageType::Pointer();

  // Read 3D or higher (but 4D probably doesn't work correctly)
  typedef itk::ImageSeriesReader<ImageType> ReaderType;

  typename ReaderType::Pointer p_clReader = ReaderType::New();

  p_clReader->SetImageIO(p_clImageIO);
  p_clReader->SetFileNames(vDicomFiles);

  try {
    p_clReader->Update();
  }
  catch (itk::ExceptionObject &e) {
    std::cerr << "Error: " << e << std::endl;
    return typename ImageType::Pointer();
  }

  typename itk::Image<PixelType, Dimension>::Pointer p_clImage = p_clReader->GetOutput();
  p_clImage->SetMetaDataDictionary(p_clImageIO->GetMetaDataDictionary());

  return p_clImage;
  //return p_clReader->GetOutput();
}

// For flexible support of 0028|0106 and 0028|0107
template<typename PixelType>
bool ComputePixelMinMax(const PixelType *p_begin, const PixelType *p_end, PixelType &minValue, PixelType &maxValue) {
  auto stPair = std::minmax_element(p_begin, p_end);
  minValue = *stPair.first;
  maxValue = *stPair.second;
  return true;
}

template<>
inline bool ComputePixelMinMax<itk::RGBPixel<uint8_t>>(const itk::RGBPixel<uint8_t> *, const itk::RGBPixel<uint8_t> *, itk::RGBPixel<uint8_t> &, itk::RGBPixel<uint8_t> &) {
  return false;
}

template<>
inline bool ComputePixelMinMax<itk::RGBAPixel<uint8_t>>(const itk::RGBAPixel<uint8_t> *, const itk::RGBAPixel<uint8_t> *, itk::RGBAPixel<uint8_t> &, itk::RGBAPixel<uint8_t> &) {
  return false;
}


template<typename PixelType>
bool SetDicomPixelInformation(itk::MetaDataDictionary &) { return false; }

template<>
bool SetDicomPixelInformation<int8_t>(itk::MetaDataDictionary &clTags);

template<>
bool SetDicomPixelInformation<uint8_t>(itk::MetaDataDictionary &clTags);

template<>
bool SetDicomPixelInformation<int16_t>(itk::MetaDataDictionary &clTags);

template<>
bool SetDicomPixelInformation<uint16_t>(itk::MetaDataDictionary &clTags);

template<>
bool SetDicomPixelInformation<int32_t>(itk::MetaDataDictionary &clTags);

template<>
bool SetDicomPixelInformation<uint32_t>(itk::MetaDataDictionary &clTags);

template<>
bool SetDicomPixelInformation<float>(itk::MetaDataDictionary &clTags);

template<>
bool SetDicomPixelInformation<double>(itk::MetaDataDictionary &clTags);

template<>
bool SetDicomPixelInformation<itk::RGBPixel<uint8_t>>(itk::MetaDataDictionary &clTags);

template<>
bool SetDicomPixelInformation<itk::RGBAPixel<uint8_t>>(itk::MetaDataDictionary &clTags);

template<typename PixelType, unsigned int Dimension>
bool SaveDicomImage(itk::Image<PixelType, Dimension> *p_clImage, const std::string &strPath, bool bCompress) {
  typedef itk::GDCMImageIO ImageIOType;
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::Image<PixelType, 3> VolumeType;
  typedef itk::Image<PixelType, 2> SliceType;
  typedef itk::ImageSeriesWriter<VolumeType, SliceType> WriterType;
  typedef itk::NumericSeriesFileNames FileNamesGeneratorType;
  typedef typename WriterType::DictionaryArrayType DictionaryArrayType;
  typedef typename WriterType::DictionaryRawPointer DictionaryRawPointer;

  static_assert((Dimension == 2 || Dimension == 3), "Only 2D or 3D images are supported.");

  if (!p_clImage || !SetDicomPixelInformation<PixelType>(p_clImage->GetMetaDataDictionary()))
    return false;

  if (Dimension == 2)
    return SaveDicomSlice<PixelType>((SliceType *)p_clImage, strPath, bCompress); // Cast to prevent compiler error... we NEVER reach this in 3D anyway

  itk::MetaDataDictionary clRefTags = p_clImage->GetMetaDataDictionary();

  {
    // Quick test to see if the reference tags are DICOM
    std::string strSeriesUID;
    if (!ExposeStringMetaData(clRefTags, "0020|000e", strSeriesUID)) {
      std::cerr << "Error: Reference meta data does not appear to be DICOM?" << std::endl;
      return false;
    }
  }

  typename ImageType::PointType clOrigin = p_clImage->GetOrigin();
  typename ImageType::SpacingType clSpacing = p_clImage->GetSpacing();
  typename ImageType::DirectionType clDirection = p_clImage->GetDirection();
  typename ImageType::SizeType clSize = p_clImage->GetBufferedRegion().GetSize();

  std::string strNewSeriesUID;

  {
    gdcm::UIDGenerator clGenUID;
    strNewSeriesUID = clGenUID.Generate();
  }

  std::vector<std::unique_ptr<itk::MetaDataDictionary>> vDictionaries;

  for (itk::IndexValueType z = 0; itk::SizeValueType(z) < clSize[2]; ++z) {
    std::unique_ptr<itk::MetaDataDictionary> p_clNewTags = std::make_unique<itk::MetaDataDictionary>();

    CopyStringMetaData(*p_clNewTags, clRefTags);

    std::string strNewSopInstanceUID;

    {
      gdcm::UIDGenerator clGenUID;
      strNewSopInstanceUID = clGenUID.Generate();
    }

    EncapsulateStringMetaData(*p_clNewTags, "0020|000e", strNewSeriesUID);
    EncapsulateStringMetaData(*p_clNewTags, "0008|0018", strNewSopInstanceUID);
    EncapsulateStringMetaData(*p_clNewTags, "0008|0003", strNewSopInstanceUID);

    // Image number
    EncapsulateStringMetaData(*p_clNewTags, "0020|0013", z+1);
    
    typename ImageType::IndexType clIndex;
    typename ImageType::PointType clPosition;
    clIndex[0] = clIndex[1] = 0;
    clIndex[2] = z;

    p_clImage->TransformIndexToPhysicalPoint(clIndex, clPosition);

    // Image position patient
    EncapsulateStringMetaData(*p_clNewTags, "0020|0032", clPosition);

    // Slice location
    EncapsulateStringMetaData(*p_clNewTags, "0020|1041", clPosition[2]);

    // Slice thickness and spacing between slices (probably not right, but from ITK example!)
    EncapsulateStringMetaData(*p_clNewTags, "0018|0050", clSpacing[2]);
    EncapsulateStringMetaData(*p_clNewTags, "0018|0088", clSpacing[2]);

    EncapsulateStringMetaData(*p_clNewTags, "0020|0037", clDirection);

    const PixelType * const p_begin = p_clImage->GetBufferPointer() + (z*clSize[0]*clSize[1]);
    const PixelType * const p_end = p_begin + (clSize[0]*clSize[1]);

    PixelType minValue = PixelType(), maxValue = PixelType();

    if (ComputePixelMinMax(p_begin, p_end, minValue, maxValue)) {
      EncapsulateStringMetaData(*p_clNewTags, "0028|0106", minValue);
      EncapsulateStringMetaData(*p_clNewTags, "0028|0107", maxValue);
    }
    else {
      p_clNewTags->Erase("0028|0106");
      p_clNewTags->Erase("0028|0107");
    }

    vDictionaries.emplace_back(std::move(p_clNewTags));
  }

  DictionaryArrayType clDictionaryArray(vDictionaries.size(), nullptr);

  std::transform(vDictionaries.begin(), vDictionaries.end(), clDictionaryArray.begin(),
    [](const std::unique_ptr<itk::MetaDataDictionary> &p_clTags) -> DictionaryRawPointer {
      return p_clTags.get();
    });

  MkDir(strPath);

  FileNamesGeneratorType::Pointer p_clFileNamesGenerator = FileNamesGeneratorType::New();

  if (strPath.find("%d") != std::string::npos)
    p_clFileNamesGenerator->SetSeriesFormat(strPath);
  else
    p_clFileNamesGenerator->SetSeriesFormat(strPath + "/%d.dcm");

  p_clFileNamesGenerator->SetStartIndex(1);
  p_clFileNamesGenerator->SetEndIndex(clSize[2]);

  ImageIOType::Pointer p_clImageIO = ImageIOType::New();

  p_clImageIO->KeepOriginalUIDOn();
  p_clImageIO->SetUseCompression(bCompress);

  typename WriterType::Pointer p_clWriter = WriterType::New();

  p_clWriter->SetImageIO(p_clImageIO);
  p_clWriter->SetFileNames(p_clFileNamesGenerator->GetFileNames());
  p_clWriter->SetMetaDataDictionaryArray(&clDictionaryArray);
  p_clWriter->SetUseCompression(bCompress);

  p_clWriter->SetInput((const VolumeType *)p_clImage); // Cast to prevent compiler error ... we NEVER reach this in 2D anyway

  try {
    p_clWriter->Update();
  }
  catch (itk::ExceptionObject &e) {
    std::cerr << "Error: " << e << std::endl;
    return false;
  }
  
  return true;
}

#endif // !COMMON_H
