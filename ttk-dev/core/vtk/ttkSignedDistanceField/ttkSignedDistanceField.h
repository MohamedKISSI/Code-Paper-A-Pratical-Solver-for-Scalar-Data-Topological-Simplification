#pragma once

// VTK Module
#include <ttkSignedDistanceFieldModule.h>

// ttk code includes
#include <SignedDistanceField.h>
#include <ttkAlgorithm.h>

class vtkImageData;

class TTKSIGNEDDISTANCEFIELD_EXPORT ttkSignedDistanceField
  : public ttkAlgorithm,
    protected ttk::SignedDistanceField {
public:
  static ttkSignedDistanceField *New();
  vtkTypeMacro(ttkSignedDistanceField, ttkAlgorithm);

  ///@{
  /**
   * Set/Get sampling dimension along each axis. Default will be [10,10,10]
   */
  vtkSetVector3Macro(SamplingDimensions, int);
  vtkGetVector3Macro(SamplingDimensions, int);
  ///@}

  vtkSetMacro(ExpandBox, bool);
  vtkGetMacro(ExpandBox, bool);

  vtkSetMacro(NaiveMethod, bool);
  vtkGetMacro(NaiveMethod, bool);

  vtkSetMacro(NaiveMethodAllEdges, bool);
  vtkGetMacro(NaiveMethodAllEdges, bool);

  vtkSetMacro(FastMarching, bool);
  vtkGetMacro(FastMarching, bool);

  vtkSetMacro(FastMarchingOrder, int);
  vtkGetMacro(FastMarchingOrder, int);

  vtkSetMacro(FastMarchingIterativeBand, bool);
  vtkGetMacro(FastMarchingIterativeBand, bool);

  vtkSetMacro(FastMarchingIterativeBandRatio, double);
  vtkGetMacro(FastMarchingIterativeBandRatio, double);

  /**
   * Get the output data for this algorithm.
   */
  vtkImageData *GetOutput();

protected:
  ttkSignedDistanceField();

  // Usual data generation method
  vtkTypeBool ProcessRequest(vtkInformation *,
                             vtkInformationVector **,
                             vtkInformationVector *) override;
  int RequestData(vtkInformation *request,
                  vtkInformationVector **inputVector,
                  vtkInformationVector *outputVector) override;
  virtual int RequestInformation(vtkInformation *,
                                 vtkInformationVector **,
                                 vtkInformationVector *);
  virtual int RequestUpdateExtent(vtkInformation *,
                                  vtkInformationVector **,
                                  vtkInformationVector *);
  int FillInputPortInformation(int, vtkInformation *) override;
  int FillOutputPortInformation(int, vtkInformation *) override;

  void computeOutputInformation(vtkInformationVector **inputVector);

  int SamplingDimensions[3];
  bool ExpandBox = true;
  bool NaiveMethod = false;
  bool NaiveMethodAllEdges = false;
  bool FastMarching = false;
  int FastMarchingOrder = 1;
  bool FastMarchingIterativeBand = false;
  double FastMarchingIterativeBandRatio = 1.0;

private:
  std::array<int, 6> DataExtent{0, 0, 0, 0, 0, 0};
  std::array<double, 3> Origin{0.0, 0.0, 0.0};
};
