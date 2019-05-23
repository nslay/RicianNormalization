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

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#elif defined(__unix__)
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <glob.h>
#else
#error "Not implemented."
#endif // _WIN32

#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include "Common.h"

#include "gdcmBase64.h"

void Trim(std::string &strString) {
  size_t p = strString.find_first_not_of(" \t\r\n");

  if (p != std::string::npos)
    strString.erase(0, p);

  p = strString.find_last_not_of(" \t\r\n");

  if (p != std::string::npos && p+1 < strString.size())
    strString.erase(p+1);
}

template<>
std::vector<std::string> SplitString<std::string>(const std::string &strValue, const std::string &strDelim) {
  std::vector<std::string> vTokens;
  std::string strToken;

  size_t p = 0;
  while (p < strValue.size()) {
    size_t q = strValue.find_first_of(strDelim, p);

    if (q != std::string::npos) {
      strToken = strValue.substr(p, q-p);
    }
    else {
      strToken = strValue.substr(p);
      q = strValue.size();
    }

    vTokens.push_back(strToken);

    p = q+1;
  }

  return vTokens;
}

bool ParseITKTag(const std::string &strKey, uint16_t &ui16Group, uint16_t &ui16Element) {
  if (strKey.empty())
    return false;

  char *p = nullptr;
  unsigned long ulGroup = strtoul(strKey.c_str(), &p, 16);

  if (*p != '|' || *(p+1) == '\0' || ulGroup > std::numeric_limits<uint16_t>::max())
    return false;

  unsigned long ulElement = strtoul(p+1, &p, 16);

  if (*p != '\0' || ulElement > std::numeric_limits<uint16_t>::max())
    return false;

  ui16Group = (uint16_t)ulGroup;
  ui16Element = (uint16_t)ulElement;

  return true;
}

void CopyStringMetaData(itk::MetaDataDictionary &clNew, const itk::MetaDataDictionary &clOriginal) {
  typedef itk::MetaDataObject<std::string> StringMetaDataType;

  clNew.Clear();

  for (auto itr = clOriginal.Begin(); itr != clOriginal.End(); ++itr) {
    const StringMetaDataType * const p_clStringMeta = dynamic_cast<const StringMetaDataType *>(itr->second.GetPointer());

    if (p_clStringMeta != nullptr) {
      StringMetaDataType::Pointer p_clNewStringMeta = StringMetaDataType::New();
      p_clNewStringMeta->SetMetaDataObjectValue(p_clStringMeta->GetMetaDataObjectValue());
      clNew[itr->first] = p_clNewStringMeta;
    }
  }
}

template<>
bool ExposeStringMetaData<std::vector<float>>(const itk::MetaDataDictionary &clTags, const std::string &strKey, std::vector<float> &vValues) {
  std::string strValue;

  if (!ExposeStringMetaData(clTags, strKey, strValue))
    return false;

  Trim(strValue);

  if (strValue.empty()) {
    vValues.clear();
    return true;
  }

  vValues = SplitString<float>(strValue, "\\");

  return vValues.size() > 0;
}

template<>
bool ExposeStringMetaData<std::vector<double>>(const itk::MetaDataDictionary &clTags, const std::string &strKey, std::vector<double> &vValues) {
  std::string strValue;

  if (!ExposeStringMetaData(clTags, strKey, strValue))
    return false;

  Trim(strValue);

  if (strValue.empty()) {
    vValues.clear();
    return true;
  }

  vValues = SplitString<double>(strValue, "\\");

  return vValues.size() > 0;
}

template<>
bool ExposeStringMetaData<gdcm::CSAHeader>(const itk::MetaDataDictionary &clTags, const std::string &strKey, gdcm::CSAHeader &clCSAHeader) {
  uint16_t ui16Group = 0, ui16Element = 0;

  clCSAHeader = gdcm::CSAHeader();

  if (!ParseITKTag(strKey, ui16Group, ui16Element))
    return false;

  std::string strValue;

  if (!ExposeStringMetaData(clTags, strKey, strValue))
    return false;

  const int iDecodeLength = gdcm::Base64::GetDecodeLength(strValue.c_str(), (int)strValue.size());

  if (iDecodeLength <= 0)
    return false;

  std::vector<char> vBuffer(iDecodeLength);

  if (gdcm::Base64::Decode(vBuffer.data(), vBuffer.size(), strValue.c_str(), strValue.size()) == 0)
    return false;

  gdcm::DataElement clDataElement;
  clDataElement.SetTag(gdcm::Tag(ui16Group, ui16Element));
  clDataElement.SetByteValue(vBuffer.data(), vBuffer.size());

  return clCSAHeader.LoadFromDataElement(clDataElement);
}

template<>
void EncapsulateStringMetaData<std::vector<float>>(itk::MetaDataDictionary &clTags, const std::string &strKey, const std::vector<float> &vValues) {
  if (vValues.empty())
    return;

  std::stringstream valueStream;
  valueStream << vValues[0];

  for (size_t i = 1; i < vValues.size(); ++i)
    valueStream << '\\' << vValues[i];

  EncapsulateStringMetaData(clTags, strKey, valueStream.str());
}

template<>
void EncapsulateStringMetaData<std::vector<double>>(itk::MetaDataDictionary &clTags, const std::string &strKey, const std::vector<double> &vValues) {
  if (vValues.empty())
    return;

  std::stringstream valueStream;
  valueStream << vValues[0];

  for (size_t i = 1; i < vValues.size(); ++i)
    valueStream << '\\' << vValues[i];

  EncapsulateStringMetaData(clTags, strKey, valueStream.str());
}

template<>
void EncapsulateStringMetaData<itk::Point<float, 3>>(itk::MetaDataDictionary &clTags, const std::string &strKey, const itk::Point<float, 3> &clPoint) {
  std::stringstream valueStream;
  valueStream << clPoint[0] << '\\' << clPoint[1] << '\\' << clPoint[2];
  EncapsulateStringMetaData(clTags, strKey, valueStream.str());
}

template<>
void EncapsulateStringMetaData<itk::Point<float, 2>>(itk::MetaDataDictionary &clTags, const std::string &strKey, const itk::Point<float, 2> &clPoint) {
  std::stringstream valueStream;
  valueStream << clPoint[0] << '\\' << clPoint[1];
  EncapsulateStringMetaData(clTags, strKey, valueStream.str());
}

template<>
void EncapsulateStringMetaData<itk::Point<double, 3>>(itk::MetaDataDictionary &clTags, const std::string &strKey, const itk::Point<double, 3> &clPoint) {
  std::stringstream valueStream;
  valueStream << clPoint[0] << '\\' << clPoint[1] << '\\' << clPoint[2];
  EncapsulateStringMetaData(clTags, strKey, valueStream.str());
}

template<>
void EncapsulateStringMetaData<itk::Point<double, 2>>(itk::MetaDataDictionary &clTags, const std::string &strKey, const itk::Point<double, 2> &clPoint) {
  std::stringstream valueStream;
  valueStream << clPoint[0] << '\\' << clPoint[1];
  EncapsulateStringMetaData(clTags, strKey, valueStream.str());
}

template<>
bool ExposeCSAMetaData<std::string>(gdcm::CSAHeader &clHeader, const std::string &strKey, std::string &strValue) {
  if (!clHeader.FindCSAElementByName(strKey.c_str()))
    return false;

  const gdcm::CSAElement &clElement = clHeader.GetCSAElementByName(strKey.c_str());
  const gdcm::ByteValue * const p_clByteValue = clElement.GetByteValue();

  if (p_clByteValue == nullptr)
    return false;

  std::vector<char> vBuffer(p_clByteValue->GetLength());
  if (!p_clByteValue->GetBuffer(vBuffer.data(), vBuffer.size()))
    return false;

  strValue.assign(vBuffer.begin(), vBuffer.end());

  return true;
}

template<>
bool SetDicomPixelInformation<int8_t>(itk::MetaDataDictionary &clTags) {
  std::string strPixelRep;
  ExposeStringMetaData(clTags, "0028|0004", strPixelRep);

  if (strPixelRep != "MONOCHROME2" && strPixelRep != "MONOCHROME1")
    EncapsulateStringMetaData(clTags, "0028|0004", std::string("MONOCHROME2"));

  EncapsulateStringMetaData(clTags, "0028|0002", 1); // Samples per pixel
  
  EncapsulateStringMetaData(clTags, "0028|0100", 8); // bits allocated
  EncapsulateStringMetaData(clTags, "0028|0101", 8); // bits stored
  EncapsulateStringMetaData(clTags, "0028|0102", 7); // high bit
  EncapsulateStringMetaData(clTags, "0028|0103", 1); // pixel rep

  return true;
}

template<>
bool SetDicomPixelInformation<uint8_t>(itk::MetaDataDictionary &clTags) {
  std::string strPixelRep;
  ExposeStringMetaData(clTags, "0028|0004", strPixelRep);

  if (strPixelRep != "MONOCHROME2" && strPixelRep != "MONOCHROME1")
    EncapsulateStringMetaData(clTags, "0028|0004", std::string("MONOCHROME2"));

  EncapsulateStringMetaData(clTags, "0028|0002", 1); // Samples per pixel
  
  EncapsulateStringMetaData(clTags, "0028|0100", 8); // bits allocated
  EncapsulateStringMetaData(clTags, "0028|0101", 8); // bits stored
  EncapsulateStringMetaData(clTags, "0028|0102", 7); // high bit
  EncapsulateStringMetaData(clTags, "0028|0103", 0); // pixel rep

  return true;
}

template<>
bool SetDicomPixelInformation<int16_t>(itk::MetaDataDictionary &clTags) {
  std::string strPixelRep;
  ExposeStringMetaData(clTags, "0028|0004", strPixelRep);

  if (strPixelRep != "MONOCHROME2" && strPixelRep != "MONOCHROME1")
    EncapsulateStringMetaData(clTags, "0028|0004", std::string("MONOCHROME2"));

  EncapsulateStringMetaData(clTags, "0028|0002", 1); // Samples per pixel
  
  EncapsulateStringMetaData(clTags, "0028|0100", 16); // bits allocated
  EncapsulateStringMetaData(clTags, "0028|0101", 16); // bits stored
  EncapsulateStringMetaData(clTags, "0028|0102", 15); // high bit
  EncapsulateStringMetaData(clTags, "0028|0103", 1); // pixel rep

  return true;
}

template<>
bool SetDicomPixelInformation<uint16_t>(itk::MetaDataDictionary &clTags) {
  std::string strPixelRep;
  ExposeStringMetaData(clTags, "0028|0004", strPixelRep);

  if (strPixelRep != "MONOCHROME2" && strPixelRep != "MONOCHROME1")
    EncapsulateStringMetaData(clTags, "0028|0004", std::string("MONOCHROME2"));

  EncapsulateStringMetaData(clTags, "0028|0002", 1); // Samples per pixel
  
  EncapsulateStringMetaData(clTags, "0028|0100", 16); // bits allocated
  EncapsulateStringMetaData(clTags, "0028|0101", 16); // bits stored
  EncapsulateStringMetaData(clTags, "0028|0102", 15); // high bit
  EncapsulateStringMetaData(clTags, "0028|0103", 0); // pixel rep

  return true;
}

template<>
bool SetDicomPixelInformation<int32_t>(itk::MetaDataDictionary &clTags) {
  std::string strPixelRep;
  ExposeStringMetaData(clTags, "0028|0004", strPixelRep);

  if (strPixelRep != "MONOCHROME2" && strPixelRep != "MONOCHROME1")
    EncapsulateStringMetaData(clTags, "0028|0004", std::string("MONOCHROME2"));

  EncapsulateStringMetaData(clTags, "0028|0002", 1); // Samples per pixel
  
  EncapsulateStringMetaData(clTags, "0028|0100", 32); // bits allocated
  EncapsulateStringMetaData(clTags, "0028|0101", 32); // bits stored
  EncapsulateStringMetaData(clTags, "0028|0102", 31); // high bit
  EncapsulateStringMetaData(clTags, "0028|0103", 1); // pixel rep

  return true;
}

template<>
bool SetDicomPixelInformation<uint32_t>(itk::MetaDataDictionary &clTags) {
  std::string strPixelRep;
  ExposeStringMetaData(clTags, "0028|0004", strPixelRep);

  if (strPixelRep != "MONOCHROME2" && strPixelRep != "MONOCHROME1")
    EncapsulateStringMetaData(clTags, "0028|0004", std::string("MONOCHROME2"));

  EncapsulateStringMetaData(clTags, "0028|0002", 1); // Samples per pixel
  
  EncapsulateStringMetaData(clTags, "0028|0100", 32); // bits allocated
  EncapsulateStringMetaData(clTags, "0028|0101", 32); // bits stored
  EncapsulateStringMetaData(clTags, "0028|0102", 31); // high bit
  EncapsulateStringMetaData(clTags, "0028|0103", 0); // pixel rep

  return true;
}

template<>
bool SetDicomPixelInformation<float>(itk::MetaDataDictionary &clTags) {
  EncapsulateStringMetaData(clTags, "0028|0004", std::string("MONOCHROME2"));

  EncapsulateStringMetaData(clTags, "0028|0002", 1); // Samples per pixel
  EncapsulateStringMetaData(clTags, "0028|0100", 32); // Bits allocated

  // Not relevant for float/double data
  clTags.Erase("0028|0101");
  clTags.Erase("0028|0102");
  clTags.Erase("0028|0103");

  return true;
}

template<>
bool SetDicomPixelInformation<double>(itk::MetaDataDictionary &clTags) {
  EncapsulateStringMetaData(clTags, "0028|0004", std::string("MONOCHROME2"));

  EncapsulateStringMetaData(clTags, "0028|0002", 1); // Samples per pixel
  EncapsulateStringMetaData(clTags, "0028|0100", 64); // Bits allocated

  // Not relevant for float/double data
  clTags.Erase("0028|0101");
  clTags.Erase("0028|0102");
  clTags.Erase("0028|0103");

  return true;
}

template<>
bool SetDicomPixelInformation<itk::RGBPixel<uint8_t>>(itk::MetaDataDictionary &clTags) {
  EncapsulateStringMetaData(clTags, "0028|0004", std::string("RGB"));

  EncapsulateStringMetaData(clTags, "0028|0002", 3); // Samples per pixel

  EncapsulateStringMetaData(clTags, "0028|0100", 8); // bits allocated
  EncapsulateStringMetaData(clTags, "0028|0101", 8); // bits stored
  EncapsulateStringMetaData(clTags, "0028|0102", 7); // high bit
  EncapsulateStringMetaData(clTags, "0028|0103", 0); // pixel rep

  return true;
}

template<>
bool SetDicomPixelInformation<itk::RGBAPixel<uint8_t>>(itk::MetaDataDictionary &clTags) {
  EncapsulateStringMetaData(clTags, "0028|0004", std::string("ARGB")); // XXX: Retired!

  EncapsulateStringMetaData(clTags, "0028|0002", 4); // Samples per pixel

  EncapsulateStringMetaData(clTags, "0028|0100", 8); // bits allocated
  EncapsulateStringMetaData(clTags, "0028|0101", 8); // bits stored
  EncapsulateStringMetaData(clTags, "0028|0102", 7); // high bit
  EncapsulateStringMetaData(clTags, "0028|0103", 0); // pixel rep

  return true;
}

#ifdef _WIN32
bool FileExists(const std::string &strPath) {
  return GetFileAttributes(strPath.c_str()) != INVALID_FILE_ATTRIBUTES;
}
#endif // _WIN32

#ifdef __unix__
bool FileExists(const std::string &strPath) {
  struct stat stBuff;
  memset(&stBuff, 0, sizeof(stBuff));
  return stat(strPath.c_str(), &stBuff) == 0;
}
#endif // __unix__

#ifdef _WIN32
bool IsFolder(const std::string &strPath) {
  const DWORD dwFlags = GetFileAttributes(strPath.c_str());
  return dwFlags != INVALID_FILE_ATTRIBUTES ? ((dwFlags & FILE_ATTRIBUTE_DIRECTORY) != 0) : false;
}
#endif // _WIN32

#ifdef __unix__
bool IsFolder(const std::string &strPath) {
  struct stat stBuff;
  memset(&stBuff, 0, sizeof(stBuff));

  if (stat(strPath.c_str(), &stBuff) != 0)
    return false;

  return S_ISDIR(stBuff.st_mode) != 0;
}
#endif // __unix__

#ifdef _WIN32
bool RmDir(const std::string &strPath) {
  return RemoveDirectory(strPath.c_str()) != 0;
}
#endif // _WIN32

#ifdef __unix__
bool RmDir(const std::string &strPath) {
  return rmdir(strPath.c_str()) == 0;
}
#endif // __unix__

#ifdef _WIN32
bool MkDir(const std::string &strPath) {
  return CreateDirectory(strPath.c_str(), nullptr) != 0;
}
#endif // _WIN32

#ifdef __unix__
bool MkDir(const std::string &strPath) {
  return mkdir(strPath.c_str(), 0777) == 0;
}
#endif // __unix__

#ifdef _WIN32
bool Unlink(const std::string &strPath) {
  return DeleteFile(strPath.c_str()) != 0;
}
#endif // _WIN32

#ifdef __unix__
bool Unlink(const std::string &strPath) {
  return unlink(strPath.c_str());
}
#endif // __unix__

#ifdef _WIN32
bool Copy(const std::string &strFrom, const std::string &strTo, bool bReplace) {
  return CopyFile(strFrom.c_str(), strTo.c_str(), bReplace ? FALSE : TRUE);
}
#endif // _WIN32

#ifdef __unix__
bool Copy(const std::string &strFrom, const std::string &strTo, bool bReplace) {
  int iFromFd = open(strFrom.c_str(), O_RDONLY);

  if (iFromFd == -1)
    return false;

  {
    struct stat stToStat, stFromStat;
    std::memset(&stToStat, 0, sizeof(stToStat));
    std::memset(&stFromStat, 0, sizeof(stFromStat));

    if (stat(strTo.c_str(), &stToStat) == 0 && fstat(iFromFd, &stFromStat) == 0 &&
      stToStat.st_dev == stFromStat.st_dev && stToStat.st_ino == stFromStat.st_ino) {
      // Same file, give up
      close(iFromFd);
      return false;
    }
  }

  int iFlags = O_TRUNC | O_CREAT | O_WRONLY;

  if (!bReplace)
    iFlags |= O_EXCL;

  int iToFd = open(strTo.c_str(), iFlags, 0777);

  if (iToFd == -1) {
    close(iFromFd);
    return false;
  }

  unsigned char a_ucBuff[4096];

  ssize_t sszSizeRead = 0, sszSizeWrote = 0;
  while ((sszSizeRead = read(iFromFd, a_ucBuff, sizeof(a_ucBuff))) > 0) {
    unsigned char *p = a_ucBuff;

    while (sszSizeRead > 0 && (sszSizeWrote = write(iToFd, p, sszSizeRead)) > 0) {
      p += sszSizeWrote;
      sszSizeRead -= sszSizeWrote;
    }

    if (sszSizeWrote == -1)
      break;
  }

  bool bSuccess = (sszSizeRead >= 0 && sszSizeWrote >= 0);

  close(iFromFd);
  close(iToFd);

  return bSuccess;
}
#endif // __unix__

#ifdef _WIN32
bool Rename(const std::string &strFrom, const std::string &strTo, bool bReplace) {
  DWORD dwFlags = (MOVEFILE_COPY_ALLOWED | MOVEFILE_WRITE_THROUGH | MOVEFILE_FAIL_IF_NOT_TRACKABLE);

  if (bReplace)
    dwFlags |= MOVEFILE_REPLACE_EXISTING;

  return MoveFileEx(strFrom.c_str(), strTo.c_str(), dwFlags) != 0;
}
#endif // _WIN32

#ifdef __unix__
bool Rename(const std::string &strFrom, const std::string &strTo, bool bReplace) {
  if (!bReplace && FileExists(strTo))
    return false;

  if (rename(strFrom.c_str(), strTo.c_str()) == 0)
    return true;

  if ((errno == EXDEV || errno == EPERM || errno == EACCES) && Copy(strFrom, strTo, bReplace)) {
    Unlink(strFrom);
    return true;
  }

  return false;
}
#endif // __unix__

#ifdef _WIN32
void USleep(unsigned int uiMicroSeconds) {
  DWORD dwMilliSeconds = uiMicroSeconds / 1000;

  if (uiMicroSeconds - dwMilliSeconds * 1000 >= 500)
    ++dwMilliSeconds;

  Sleep(dwMilliSeconds);
}
#endif // _WIN32

#ifdef __unix__
void USleep(unsigned int uiMicroSeconds) {
  unsigned int uiSeconds = uiMicroSeconds / 1000000;
  uiMicroSeconds -= uiSeconds * 1000000;

  sleep(uiSeconds);
  usleep(uiMicroSeconds);
}
#endif // __unix__

#ifdef _WIN32
std::string BaseName(std::string strPath) {
  if (strPath.empty())
    return std::string();

  while (strPath.size() > 1 && (strPath.back() == '/' || strPath.back() == '\\'))
    strPath.pop_back();

  if (strPath.size() == 1)
    return strPath;

  size_t p = strPath.find_last_of("/\\");

  return p != std::string::npos ? strPath.substr(p+1) : strPath;
}
#endif // _WIN32

#ifdef __unix__
std::string BaseName(std::string strPath) {
  if (strPath.empty())
    return std::string();

  while (strPath.size() > 1 && strPath.back() == '/')
    strPath.pop_back();

  if (strPath.size() == 1)
    return strPath;

  size_t p = strPath.find_last_of("/");

  return p != std::string::npos ? strPath.substr(p+1) : strPath;
}
#endif // __unix__

#ifdef _WIN32
std::string DirName(std::string strPath) {
  if (strPath.empty())
    return std::string(".");

  // XXX: Did not implement DirName for Windows shares

  while (strPath.size() > 1 && (strPath.back() == '/' || strPath.back() == '\\'))
    strPath.pop_back();

  if (strPath.size() == 2) {
    if (std::isalpha(strPath[0]) && strPath[1] == ':')
      return strPath;
  }

  if (strPath.size() == 1) {
    return (strPath == "/" || strPath == "\\") ? strPath : std::string(".");
  }

  size_t p = strPath.find_last_of("/\\");

  return p != std::string::npos ? strPath.substr(0, p) : std::string(".");
}
#endif // _WIN32

#ifdef __unix__
std::string DirName(std::string strPath) {
  if (strPath.empty())
    return std::string(".");

  while (strPath.size() > 1 && strPath.back() == '/')
    strPath.pop_back();

  if (strPath.size() == 1) {
    return strPath == "/" ? strPath : std::string(".");
  }

  size_t p = strPath.find_last_of("/");

  return p != std::string::npos ? strPath.substr(0, p) : std::string(".");
}
#endif // __unix__

#ifdef _WIN32
std::string GetExtension(const std::string &strPath) {
  if (strPath.empty() || strPath.back() == '/' || strPath.back() == '\\')
    return std::string();

  size_t p = strPath.find_last_of('.');

  if (p == std::string::npos)
    return std::string();

  size_t q = strPath.find_last_of("/\\");

  if (q != std::string::npos && p < q)
    return std::string();

  return strPath.substr(p);
}
#endif // _WIN32

#ifdef __unix__
std::string GetExtension(const std::string &strPath) {
  if (strPath.empty() || strPath.back() == '/')
    return std::string();

  size_t p = strPath.find_last_of('.');

  if (p == std::string::npos)
    return std::string();

  size_t q = strPath.find_last_of('/');

  if (q != std::string::npos && p < q)
    return std::string();

  return strPath.substr(p);
}
#endif // __unix__

#ifdef _WIN32
std::string StripExtension(const std::string &strPath) {
  if (strPath.empty() || strPath.back() == '/' || strPath.back() == '\\')
    return strPath;

  size_t p = strPath.find_last_of('.');

  if (p == std::string::npos)
    return strPath;

  size_t q = strPath.find_last_of("/\\");

  if (q != std::string::npos && p < q)
    return strPath;

  return strPath.substr(0, p);  
}
#endif // _WIN32

#ifdef __unix__
std::string StripExtension(const std::string &strPath) {
  if (strPath.empty() || strPath.back() == '/')
    return strPath;

  size_t p = strPath.find_last_of('.');

  if (p == std::string::npos)
    return strPath;

  size_t q = strPath.find_last_of('/');

  if (q != std::string::npos && p < q)
    return strPath;

  return strPath.substr(0, p);  
}
#endif // __unix__

#ifdef _WIN32
std::string StripTrailingDelimiters(std::string strPath) {
  while (strPath.size() > 0 && (strPath.back() == '/' || strPath.back() == '\\'))
    strPath.pop_back();

  return strPath;
}
#endif // _WIN32

#ifdef __unix__
std::string StripTrailingDelimiters(std::string strPath) {
  while (strPath.size() > 0 && strPath.back() == '/')
    strPath.pop_back();

  return strPath;
}
#endif // __unix__

#ifdef _WIN32

// https://msdn.microsoft.com/en-us/library/aa365247

void SanitizeFileName(std::string &strFileName) {
  for (size_t i = 0; i < strFileName.size(); ++i) {
    switch (strFileName[i]) {
    case '<':
    case '>':
    case ':':
    case '"':
    case '/':
    case '\\':
    case '|':
    case '?':
    case '*':
      strFileName[i] = '_';
      break;
    default:
      if (!std::isprint(strFileName[i]))
        strFileName[i] = '_';
      break;
    }
  }
}
#endif // _WIN32

#ifdef __unix__
void SanitizeFileName(std::string &strFileName) {
  for (size_t i = 0; i < strFileName.size(); ++i) {
    switch (strFileName[i]) {
    case '/':
      strFileName[i] = '_';
      break;
    default:
      if (!std::isprint(strFileName[i]))
        strFileName[i] = '_';
      break;
    }
  }
}
#endif // __unix__

#ifdef _WIN32
void FindFiles(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFiles, bool bRecursive) {
  std::string strPattern(p_cDir);
  strPattern += '\\';
  strPattern += p_cPattern;

  //std::cout << strPattern << std::endl;

  WIN32_FIND_DATA stFindData;

  memset(&stFindData, 0, sizeof(stFindData));
  
  HANDLE hFind = FindFirstFile(strPattern.c_str(), &stFindData);

  if (hFind != INVALID_HANDLE_VALUE) {
    do {
      if (!(stFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
        std::string strPath(p_cDir);
        strPath += '\\';
        strPath += stFindData.cFileName;

        vFiles.push_back(strPath);
      }
    } while (FindNextFile(hFind, &stFindData) != FALSE);

    FindClose(hFind);
  }

  if (bRecursive) {
    strPattern = p_cDir;
    strPattern += "\\*";

    memset(&stFindData, 0, sizeof(stFindData));

    hFind = FindFirstFile(strPattern.c_str(), &stFindData);

    if (hFind == INVALID_HANDLE_VALUE)
      return;

    do {
      if (strcmp(stFindData.cFileName, ".") == 0 || strcmp(stFindData.cFileName, "..") == 0)
        continue;

      if (stFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        std::string strPath(p_cDir);
        strPath += '\\';
        strPath += stFindData.cFileName;

        FindFiles(strPath.c_str(), p_cPattern, vFiles, bRecursive);
      }
    } while (FindNextFile(hFind, &stFindData) != FALSE);

    FindClose(hFind);
  }

  return;
}
#endif // _WIN32

#ifdef __unix__
void FindFiles(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFiles, bool bRecursive) {
  glob_t stGlob;

  std::memset(&stGlob, 0, sizeof(stGlob));

  std::string strPattern = p_cDir; 
  strPattern += '/';
  strPattern += p_cPattern;

  if (glob(strPattern.c_str(), GLOB_TILDE, nullptr, &stGlob) == 0) {
    for (size_t i = 0; i < stGlob.gl_pathc; ++i) {
      std::string strPath = stGlob.gl_pathv[i];
      if (!IsFolder(strPath))
        vFiles.push_back(std::move(strPath));
    }
  }

  globfree(&stGlob);

  if (bRecursive) {
    std::memset(&stGlob, 0, sizeof(stGlob));

    strPattern = p_cDir;
    strPattern += "/*";

    if (glob(strPattern.c_str(), GLOB_TILDE, nullptr, &stGlob) == 0) {
      for (size_t i = 0; i < stGlob.gl_pathc; ++i) {
        const std::string strPath = stGlob.gl_pathv[i];
        if (strPath != "." && strPath != ".." && IsFolder(strPath))
          FindFiles(strPath.c_str(), p_cPattern, vFiles, bRecursive);
      }
    }

    globfree(&stGlob);
  }
}
#endif // __unix__

#ifdef _WIN32
void FindFolders(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFolders, bool bRecursive) {
  std::string strPattern(p_cDir);
  strPattern += '\\';
  strPattern += p_cPattern;

  //std::cout << strPattern << std::endl;

  WIN32_FIND_DATA stFindData;

  memset(&stFindData, 0, sizeof(stFindData));

  HANDLE hFind = FindFirstFile(strPattern.c_str(), &stFindData);

  if (hFind != INVALID_HANDLE_VALUE) {
    do {
      if ((stFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && strcmp(stFindData.cFileName,".") != 0 && strcmp(stFindData.cFileName,"..") != 0) {
        std::string strPath(p_cDir);
        strPath += '\\';
        strPath += stFindData.cFileName;

        vFolders.push_back(strPath);
      }
    } while (FindNextFile(hFind, &stFindData) != FALSE);

    FindClose(hFind);
  }

  if (bRecursive) {

    strPattern = p_cDir;
    strPattern += "\\*";

    memset(&stFindData, 0, sizeof(stFindData));

    hFind = FindFirstFile(strPattern.c_str(), &stFindData);

    if (hFind == INVALID_HANDLE_VALUE)
      return;

    do {
      if (strcmp(stFindData.cFileName, ".") == 0 || strcmp(stFindData.cFileName, "..") == 0)
        continue;

      if (stFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        std::string strPath(p_cDir);
        strPath += '\\';
        strPath += stFindData.cFileName;

        FindFiles(strPath.c_str(), p_cPattern, vFolders, bRecursive);
      }
    } while (FindNextFile(hFind, &stFindData) != FALSE);

    FindClose(hFind);
  }

  return;
}
#endif // _WIN32

#ifdef __unix__

void FindFolders(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFolders, bool bRecursive) {
  glob_t stGlob;

  std::memset(&stGlob, 0, sizeof(stGlob));

  std::string strPattern = p_cDir; 
  strPattern += '/';
  strPattern += p_cPattern;

  if (glob(strPattern.c_str(), GLOB_TILDE, nullptr, &stGlob) == 0) {
    for (size_t i = 0; i < stGlob.gl_pathc; ++i) {
      std::string strPath = stGlob.gl_pathv[i];
      if (strPath != "." && strPath != ".." && IsFolder(strPath))
        vFolders.push_back(std::move(strPath));
    }
  }

  globfree(&stGlob);

  if (bRecursive) {
    std::memset(&stGlob, 0, sizeof(stGlob));

    strPattern = p_cDir;
    strPattern += "/*";

    if (glob(strPattern.c_str(), GLOB_TILDE, nullptr, &stGlob) == 0) {
      for (size_t i = 0; i < stGlob.gl_pathc; ++i) {
        const std::string strPath = stGlob.gl_pathv[i];
        if (strPath != "." && strPath != ".." && IsFolder(strPath))
          FindFolders(strPath.c_str(), p_cPattern, vFolders, bRecursive);
      }
    }

    globfree(&stGlob);
  }

}

#endif // __unix__

void FindDicomFolders(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFolders, bool bRecursive) {
  typedef itk::GDCMImageIO ImageIOType;

  ImageIOType::Pointer p_clImageIO = ImageIOType::New();

  std::vector<std::string> vTmpFolders, vTmpFiles;

  vTmpFolders.push_back(p_cDir); // Check base folder too

  FindFolders(p_cDir, p_cPattern, vTmpFolders, bRecursive);

  for (size_t i = 0; i < vTmpFolders.size(); ++i) {
    const std::string &strFolder = vTmpFolders[i];

    vTmpFiles.clear();

    FindFiles(strFolder.c_str(), "*", vTmpFiles, false);

    for (size_t j = 0; j < vTmpFiles.size(); ++j) {
      const std::string &strFile = vTmpFiles[j];

      if (p_clImageIO->CanReadFile(strFile.c_str())) {
        vFolders.push_back(strFolder);
        break;
      }
    }
  }
}

