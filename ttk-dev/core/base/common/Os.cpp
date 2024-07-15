#include <Debug.h>
#include <Os.h>

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include <cwchar>
#include <direct.h>
#include <stdint.h>

#elif defined(__unix__) || defined(__APPLE__)

#include <dirent.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <iostream>
#include <sstream>

namespace ttk {

  int OsCall::getCurrentDirectory(std::string &directoryPath) {
#ifdef _WIN32
    directoryPath = _getcwd(NULL, 0);
#else
    std::vector<char> cwdName(PATH_MAX);
    char *returnedString = getcwd(cwdName.data(), cwdName.size());
    directoryPath = std::string{returnedString};
#endif
    directoryPath += "/";

    return 0;
  }

  float OsCall::getMemoryInstantUsage() {
#ifdef __linux__
    // horrible hack since getrusage() doesn't seem to work well under
    // linux
    std::stringstream procFileName;
    procFileName << "/proc/" << getpid() << "/statm";

    std::ifstream procFile(procFileName.str().data(), std::ios::in);
    if(procFile) {
      float memoryUsage;
      procFile >> memoryUsage;
      procFile.close();
      return memoryUsage / 1024.0;
    }
#endif
    return 0;
  }

  float OsCall::getTotalMemoryUsage() {
    int max_use{0};
#ifdef __linux__
    int ru_maxrss;
    struct rusage use;
    getrusage(RUSAGE_SELF, &use);
    ru_maxrss = static_cast<int>(use.ru_maxrss);
#ifdef TTK_ENABLE_MPI
    if(ttk::hasInitializedMPI()) {
      MPI_Reduce(
        &ru_maxrss, &max_use, 1, MPI_INTEGER, MPI_MAX, 0, ttk::MPIcomm_);
    } else {
      max_use = ru_maxrss;
    }
#else
    max_use = ru_maxrss;
#endif // TTK_ENABLE_MPI
#endif // __linux__
    // In Kilo Bytes
    return (double)max_use;
  }

  int OsCall::getNumberOfCores() {
#ifdef TTK_ENABLE_OPENMP
    return omp_get_max_threads();
#endif
    return 1;
  }

  std::vector<std::string>
    OsCall::listFilesInDirectory(const std::string &directoryName,
                                 const std::string &extension) {

    std::vector<std::string> filesInDir;

#ifdef _WIN32

#ifdef UNICODE
    auto toWString = [](const std::string &str) {
      if(str.empty())
        return std::wstring();
      int wcharCount
        = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
      std::wstring wstr;
      wstr.resize(wcharCount);
      MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], wcharCount);
      return wstr;
    };
    auto toString = [](const WCHAR wstr[]) {
      int charCount = WideCharToMultiByte(
        CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
      std::string str;
      str.resize(charCount);
      WideCharToMultiByte(
        CP_UTF8, 0, wstr, -1, &str[0], charCount, nullptr, nullptr);
      return str;
    };
#else
    auto toWString = [](const std::string &str) { return str; };
    auto toString = [](const char *c) { return std::string(c); };
#endif

    WIN32_FIND_DATA FindFileData;
    char *buffer;
    buffer = _getcwd(NULL, 0);
    if((buffer = _getcwd(NULL, 0)) == NULL)
      perror("_getcwd error");
    else {
      free(buffer);
    }

    HANDLE hFind
      = FindFirstFile(toWString(directoryName).c_str(), &FindFileData);
    if(hFind == INVALID_HANDLE_VALUE) {
      std::string s;
      s = "Could not open directory `";
      s += directoryName;
      s += "'. Error: ";
      s += GetLastError();
      Debug d;
      d.printErr(s);
    } else {
      const std::string filename = toString(FindFileData.cFileName);

      std::string entryExtension(filename);
      entryExtension
        = entryExtension.substr(entryExtension.find_last_of('.') + 1);

      if(entryExtension == extension)
        filesInDir.push_back(filename);
      std::string dir = directoryName;
      dir.resize(dir.size() - 1);
      while(FindNextFile(hFind, &FindFileData)) {
        if(extension.size()) {
          std::string entryExtension(filename);
          entryExtension
            = entryExtension.substr(entryExtension.find_last_of('.') + 1);
          if(entryExtension == extension)
            filesInDir.push_back(dir + filename);
        } else {
          if((filename != ".") && (filename != "..")) {
            filesInDir.push_back(directoryName + "/" + filename);
          }
        }
      }
    }
    FindClose(hFind);
#else
    DIR *d = opendir((directoryName + "/").data());
    if(!d) {
      std::string msg;
      msg = "Could not open directory `";
      msg += directoryName;
      msg += "'...";
      const Debug dbg;
      dbg.printErr(msg);
    } else {
      struct dirent *dirEntry;
      while((dirEntry = readdir(d)) != nullptr) {
        if(extension.size()) {
          std::string entryExtension(dirEntry->d_name);
          entryExtension
            = entryExtension.substr(entryExtension.find_last_of('.') + 1);
          if(entryExtension == extension)
            filesInDir.push_back(directoryName + "/"
                                 + std::string(dirEntry->d_name));
        } else {
          if((std::string(dirEntry->d_name) != ".")
             && (std::string(dirEntry->d_name) != ".."))
            filesInDir.push_back(directoryName + "/"
                                 + std::string(dirEntry->d_name));
        }
      }
      closedir(d);
    }
#endif

    std::sort(filesInDir.begin(), filesInDir.end());

    return filesInDir;
  }

  int OsCall::mkDir(const std::string &directoryName) {

#ifdef _WIN32
    return _mkdir(directoryName.data());
#else
    return mkdir(directoryName.data(), 0777);
#endif
  }

  int OsCall::nearbyint(const double &x) {
    const double upperBound = ceil(x);
    const double lowerBound = floor(x);

    if(upperBound - x <= x - lowerBound)
      return (int)upperBound;
    else
      return (int)lowerBound;
  }

  int OsCall::rmDir(const std::string &directoryName) {
    return std::remove(directoryName.c_str());
  }

  int OsCall::rmFile(const std::string &fileName) {
    return std::remove(fileName.c_str());
  }

} // namespace ttk
