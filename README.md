[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Appointment Scheduling with Delay-Tolerance Heterogeneity

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
["Appointment Scheduling with Delay-Tolerance Heterogeneity"](https://doi.org/10.1287/ijoc.2023.0025) by S. Wang, J. Li, M. Ang, T.S. Ng. 


## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2023.0025

https://doi.org/10.1287/ijoc.2023.0025.cd

Below is the BibTex for citing this snapshot of the respoitory.

```
@article{CacheTest,
  author =        {S. Wang, J. Li, M. Ang, T.S. Ng},
  publisher =     {INFORMS Journal on Computing},
  title =         {Appointment Scheduling with Delay-Tolerance Heterogeneity},
  year =          {2023},
  doi =           {10.1287/ijoc.2023.0025.cd},
  url =           {https://github.com/INFORMSJoC/2023.0025},
}  
```

## Description

This repository includes the original data, source code and computational results for the experiments presented in the paper.


## Data files

The folder data includes all the samples used in our experiments.

1. The file Simulation Samples.xlsx contains 3000 randomly generated service time samples from the generating distribution as stated in Section 6.1.

2. The file Statistics without outlier.xlsx includes the real case samples and statistics used in our case study.

3. The file Service Time Matrix Based on Visit Type1.xlsx contains 2000 randomly selected service time samples from the real case samples via categorizing patients as FV and RV.

4. The file Service Time Matrix Samples.xlsx contains 2000 randomly selected service time samples from the real case samples by categorizing patients with consulting time.

5. The files Simulation samples-1000-10-X.xlsx contains the generated service time samples that are used in Section 6.2, where X represents the number of user types and could take the values 2, 3, 4 and 5.

## Code files

1. In the folder Section 6.1, the code file simulation_A_B_tolerance_C_D.py is for solving the ED, TAD and DUM models under different numbers of users (A,B) and different delay-tolerance thresholds (C,D), where (A,B) could take the values (3,7) (5,5) and (7,3), and (C,D) could take the values (1,1) (1,1.5) and (1.5,1). Note that the ED, TAD and DUM models are encoded in the respective code files inside the model file.

2. The code file experiment_revised.py is for evaluating the out-of-sample performance among these three models under different numbers of user types, which has been used in Section 6.2 of our paper.

3. In the folder Section 6.4, the code file experiment_FV_RV_X.py is for solving the ED, TAD and DUM models under different cases of delay-tolerance thresholds, where X represents the percentile of the historical waiting times and takes the values 0.2, 0.3 and 0.4. Similarly, the code file experiment_categorize_consulting_X.py is for solving the ED, TAD and DUM models under different cases of delay-tolerance thresholds which are categorized by consulting time, where X represents the percentile of the historical waiting times and takes the values 0.2, 0.3 and 0.4.  Note that the ED, TAD and DUM models are encoded in the respective code files inside the model file. These code files are used in Section 6.4 of our paper.

4. In the folder EC E, the code file simulation_idle_A-B.py is for solving the TAD model under different idle time tolerance thresholds, where (A,B) represents the number of users and could take the values (3,7) (5,5) and (7,3). These code files are used in E-Companion EC E of our paper.



## Results

The folder results includes all the results presented in our experiments.

1. In the folder Impact of delay-tolerance, the file Solutions XX with tolerance(A-B).xlsx includes the optimal appointment schedules and sequence solutions as shown in Table 2 for model XX under delay-tolerance threshold (A-B), where XX could be ED, TAD and DUM, and (A,B) could take the values (1,0.5) (1,1) (1,1.5) and (1,2). Moreover, the file Summary.xlsx contains the out-of-sample performance comparisons for the three models that have been presented in Table 3 and Table 4. Finally, the file time tolerance(A-B).txt inside folder Simulation(C,D) records the computational time for these models under delay-tolerance threshold (A-B) and with number of users (C,D), where (A,B) could take the values (1,1) (1,1.5) and (1.5,1), and (C,D) could take the values (3,7) (5,5) and (7,3).
   
2. In the folder Impact of user heterogeneity, the file Summary.xlsx contains the out-of-sample performance comparisons for the three models under different number of user types that have been presented in Table 6. In addition, the file time.txt inside folder Simulations Summary Version-10-A of folder output-KB records the computational times for ED, TAD and DUM models, where A represents the number of user types and takes the values 2, 3, 4 and 5, and B represents the sample size and takes the values 100, 500 and 1000.
   
3. In the folder Impact of service-time distribution ambiguity, the file Summary.xlsx contains the out-of-sample performance comparisons under different delay-tolerance thresholds that have been presented in Table 8.
   
4. In the folder A case using real patient data, the files Summary-A.xlsx and Summary-B.xlsx contain the out-of-sample performance comparisons for the three models in the case study, where the patients are categorized as FV and RV in Section 6.4.1, and with consulting time in Section 6.4.2, respectively. Moreover, the figure Waiting_time_updated.png illustrates the distributions of the waiting times (minutes) and consulting times (minutes) of the outpatient data. The figure Figure-3-updated-2018-05-24-onerow.png shows the out-of-sample delay over tolerance for patients. 
   
5. In the folder Additional numerical experiments, the file Solutions TAD with tolerance(1-1).xlsx inside folder tolerance(1-1)-X includes the optimal appointment schedules and sequence solutions under idle time tolerance threshold X, where X could take the values 1.5, 2, 2.5, 3 and 3.5, which have been summarized in Table EC.1. Moreover, the file Summary(A-B).xlsx contains the out-of-sample performance comparisons for the three models under different idle time tolerances, where (A,B) represents the number of users and takes the values (3,7) (5,5) and (7,3). In addition, the file time tolerance(1-1) inside folder Simulation(A,B) of folder Computational time records the computational times for these models under different numbers of users (A,B). Finally, the file Summary samples-A-B.xlsx includes the relative out-of-sample performance with different training sample sizes.

## Replicating

To replicate any of the results presented above, put the data folder and the respective src folder under the same folder and run the respective code file.
