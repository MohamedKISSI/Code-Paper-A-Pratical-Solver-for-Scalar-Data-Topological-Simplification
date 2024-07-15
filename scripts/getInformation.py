import os
import csv

csvFile = '../results/timePerformanceComparison.csv'

with open(csvFile, 'w', newline='') as csvfileDirect:

    with open("../results/logBaselineOptimizationApproach.txt", 'r') as current_file:
        lines = current_file.readlines()[-38:]
        
        iterationBaseline = int(lines[1].strip().split()[7])
        iterationBaseline = iterationBaseline + 1

        lossPairToDeleteBaseline = float(lines[24].strip().split()[9])

        lossPairToChangeBaseline = float(lines[25].strip().split()[9])

        lossBaseline = lossPairToDeleteBaseline + lossPairToChangeBaseline

        executionTimeBaseline = float(lines[27].strip().split()[7])

        numberPairsInputDiagram = int(lines[28].strip().split()[8])

        numberPairsConstraintDiagram = int(lines[29].strip().split()[8])

        totalTimeSpentCalculatingThePersistenceDiagram = float(lines[30].strip().split()[11])

        averageTimeSpentOnPersistenceDiagrams = float(lines[31].strip().split()[11])

        totalTimeSpentCalculatingThePersistenceDiagramClustering = float(lines[32].strip().split()[12])

        averageTimeSpentOnPersistenceDiagramsClustering = float(lines[33].strip().split()[12])

        totalAveragePercentageOfModifiedVertices = float(lines[34].strip().split()[10])

        totalAveragePercentageOfImmobilePersistencePairs = float(lines[35].strip().split()[11])

        stopCondition = float(lines[36].strip().split()[4])

    with open("../results/logOurSolver.txt", 'r') as current_file:
        lines = current_file.readlines()[-33:]
        
        iterationOurSolver = int(lines[1].strip().split()[7])
        iterationOurSolver = iterationOurSolver + 1
        
        lossPairToDeleteOurSolver = float(lines[20].strip().split()[9])

        lossPairToChangeOurSolver = float(lines[21].strip().split()[9])

        lossOurSolver =  lossPairToDeleteOurSolver + lossPairToChangeOurSolver

        executionTimeOurSolver = float(lines[22].strip().split()[7])

        numberPairsInputDiagram = int(lines[23].strip().split()[8])

        numberPairsConstraintDiagram = int(lines[24].strip().split()[8])


    speedUp = executionTimeBaseline / executionTimeOurSolver

    writer = csv.writer(csvfileDirect)
    writer.writerow([ "Number Pairs Input Diagram", "Number Pairs Constraint Diagram", 
                     "Iteration Number Baseline Approach", "Execution Time Baseline Approach", 
                     "Iteration Number Our Solver", "Execution Time Our Solver", "Speedup"])

    writer.writerow([numberPairsInputDiagram, numberPairsConstraintDiagram,
                     iterationBaseline, executionTimeBaseline, 
                     iterationOurSolver, executionTimeOurSolver, speedUp])



csvFile = '../results/optimizationQualityComparison.csv'

with open(csvFile, 'w', newline='') as csvfileDirect:

    with open("../results/logLinfDistanceBaselineApproach.txt", 'r') as current_file:
        lines = current_file.readlines()[-2:]
        
        LinfDistanceBaselineApproach = float(lines[0].strip().split()[2])

    with open("../results/logL2DistanceBaselineApproach.txt", 'r') as current_file:
        lines = current_file.readlines()[-2:]
        
        logL2DistanceBaselineApproach = float(lines[0].strip().split()[2])

    with open("../results/logLinfDistanceOurSolver.txt", 'r') as current_file:
        lines = current_file.readlines()[-2:]
        
        LinfDistanceOurSolver = float(lines[0].strip().split()[2])

    with open("../results/logL2DistanceOurSolver.txt", 'r') as current_file:
        lines = current_file.readlines()[-2:]
        
        logL2DistanceOurSolver = float(lines[0].strip().split()[2])

    writer = csv.writer(csvfileDirect)
    writer.writerow(["Number Pairs Input Diagram", "Number Pairs Constraint Diagram", 
                     "Loss Baseline Approach", "L2 distance Between The Input And Output Baseline Approach" , "Maximum Pointwise Value Difference Between The Input And Output Baseline Approach",
                     "Loss Our Solver", "L2 Distance Between The Input And Output Of Our Solver", "Maximum Pointwise Value Difference Between The Input And Output Of Our Solver"])

    writer.writerow([numberPairsInputDiagram, numberPairsConstraintDiagram,
                     lossBaseline, logL2DistanceBaselineApproach, LinfDistanceBaselineApproach, 
                     lossOurSolver, logL2DistanceOurSolver, LinfDistanceOurSolver ])