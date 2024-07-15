/// \author Julien Tierny <julien.tierny@sorbonne-universite.fr>.
/// \date February 2017.
///
/// \brief Command line program for critical point computation.

// TTK Includes
#include <CommandLineParser.h>
#include <ttkScalarFieldCriticalPoints.h>

// VTK Includes
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkXMLDataObjectWriter.h>
#include <vtkXMLGenericDataObjectReader.h>

int main(int argc, char **argv) {

  // ---------------------------------------------------------------------------
  // Program variables
  // ---------------------------------------------------------------------------
  std::vector<std::string> inputFilePaths;
  std::vector<std::string> inputArrayNames;
  std::string outputPathPrefix{"output"};
  bool listArrays{false};
  bool forceOffset{false};

  int backEnd = 0;
  int startingRL = 0;
  int stoppingRL = -1;
  double tl = 0.0;

  // ---------------------------------------------------------------------------
  // Set program variables based on command line arguments
  // ---------------------------------------------------------------------------
  {
    ttk::CommandLineParser parser;

    // -------------------------------------------------------------------------
    // Standard options and arguments
    // -------------------------------------------------------------------------
    parser.setArgument(
      "i", &inputFilePaths, "Input data-sets (*.vti, *vtu, *vtp)", false);
    parser.setArgument("a", &inputArrayNames, "Input array names", true);
    parser.setArgument(
      "o", &outputPathPrefix, "Output file prefix (no extension)", true);
    parser.setOption("l", &listArrays, "List available arrays");

    parser.setOption("F", &forceOffset, "Force custom offset field (array #1)");

    parser.setArgument("B", &backEnd, "Method (0:FTM, 1: progressive)", true);
    parser.setArgument("S", &startingRL,
                       "Starting Resolution Level for progressive "
                       "multiresolution scheme (-1: finest level)",
                       true);
    parser.setArgument("E", &stoppingRL,
                       "Stopping Resolution Level for progressive "
                       "multiresolution scheme (-1: finest level)",
                       true);
    parser.setArgument("T", &tl, "Time limit for progressive method", true);

    parser.parse(argc, argv);
  }

  // ---------------------------------------------------------------------------
  // Command line output messages.
  // ---------------------------------------------------------------------------
  ttk::Debug msg;
  msg.setDebugMsgPrefix("ScalarFieldCriticalPoints");

  // ---------------------------------------------------------------------------
  // Initialize ttkScalarFieldCriticalPoints module (adjust parameters)
  // ---------------------------------------------------------------------------
  auto scalarFieldCriticalPoints
    = vtkSmartPointer<ttkScalarFieldCriticalPoints>::New();

  // ---------------------------------------------------------------------------
  // Read input vtkDataObjects (optionally: print available arrays)
  // ---------------------------------------------------------------------------
  vtkDataArray *defaultArray = nullptr;
  for(size_t i = 0; i < inputFilePaths.size(); i++) {
    // init a reader that can parse any vtkDataObject stored in xml format
    auto reader = vtkSmartPointer<vtkXMLGenericDataObjectReader>::New();
    reader->SetFileName(inputFilePaths[i].data());
    reader->Update();

    // check if input vtkDataObject was successfully read
    auto inputDataObject = reader->GetOutput();
    if(!inputDataObject) {
      msg.printErr("Unable to read input file `" + inputFilePaths[i] + "' :(");
      return 1;
    }

    auto inputAsVtkDataSet = vtkDataSet::SafeDownCast(inputDataObject);

    // if requested print list of arrays, otherwise proceed with execution
    if(listArrays) {
      msg.printMsg(inputFilePaths[i] + ":");
      if(inputAsVtkDataSet) {
        // Point Data
        msg.printMsg("  PointData:");
        auto pointData = inputAsVtkDataSet->GetPointData();
        for(int j = 0; j < pointData->GetNumberOfArrays(); j++)
          msg.printMsg("    - " + std::string(pointData->GetArrayName(j)));

        // Cell Data
        msg.printMsg("  CellData:");
        auto cellData = inputAsVtkDataSet->GetCellData();
        for(int j = 0; j < cellData->GetNumberOfArrays(); j++)
          msg.printMsg("    - " + std::string(cellData->GetArrayName(j)));
      } else {
        msg.printErr("Unable to list arrays on file `" + inputFilePaths[i]
                     + "'");
        return 1;
      }
    } else {
      // feed input object to ttkScalarFieldCriticalPoints filter
      scalarFieldCriticalPoints->SetInputDataObject(i, reader->GetOutput());

      // default arrays
      if(!defaultArray) {
        defaultArray = inputAsVtkDataSet->GetPointData()->GetArray(0);
        if(!defaultArray)
          defaultArray = inputAsVtkDataSet->GetCellData()->GetArray(0);
      }
    }
  }

  // terminate program if it was just asked to list arrays
  if(listArrays) {
    return 0;
  }

  // ---------------------------------------------------------------------------
  // Specify which arrays of the input vtkDataObjects will be processed
  // ---------------------------------------------------------------------------
  if(!inputArrayNames.size()) {
    if(defaultArray)
      inputArrayNames.emplace_back(defaultArray->GetName());
  }
  for(size_t i = 0; i < inputArrayNames.size(); i++)
    scalarFieldCriticalPoints->SetInputArrayToProcess(
      i, 0, 0, 0, inputArrayNames[i].data());

  // ---------------------------------------------------------------------------
  // Execute ttkScalarFieldCriticalPoints filter
  // ---------------------------------------------------------------------------
  scalarFieldCriticalPoints->SetBackEnd(backEnd);
  scalarFieldCriticalPoints->SetTimeLimit(tl);
  scalarFieldCriticalPoints->SetStartingResolutionLevel(startingRL);
  scalarFieldCriticalPoints->SetStoppingResolutionLevel(stoppingRL);
  scalarFieldCriticalPoints->SetForceInputOffsetScalarField(forceOffset);
  scalarFieldCriticalPoints->Update();

  // ---------------------------------------------------------------------------
  // If output prefix is specified then write all output objects to disk
  // ---------------------------------------------------------------------------
  if(!outputPathPrefix.empty()) {
    for(int i = 0; i < scalarFieldCriticalPoints->GetNumberOfOutputPorts();
        i++) {
      auto output = scalarFieldCriticalPoints->GetOutputDataObject(i);
      auto writer = vtkSmartPointer<vtkXMLWriter>::Take(
        vtkXMLDataObjectWriter::NewWriter(output->GetDataObjectType()));

      std::string outputFileName = outputPathPrefix + "_port_"
                                   + std::to_string(i) + "."
                                   + writer->GetDefaultFileExtension();
      msg.printMsg("Writing output file `" + outputFileName + "'...");
      writer->SetInputDataObject(output);
      writer->SetFileName(outputFileName.data());
      writer->Update();
    }
  }

  return 0;
}
