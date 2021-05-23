# strategic_rep
This is the code for the paper Strategic Classification in the Dark.
There are three main experiment in the paper and you can run them:
1) full information experiment. That mean constants know the classifier. In this experiment you can see the
different between Hardt model and the linear svm model.
**Running the experiment:**
--full_info_exp

2) Movement in the dark experiment. That mean constants doesn't know the classifier. In this experiment you 
can see how the number of sample constant learn from influence Hardt model and svm model results.
**Note: for running this experiment you should run first experiment number 1**
at the end of this experiment you can find the output files at:
 result/dark_exp/cost_factor=_{what u defined}_epsilon={epsilon}
 In this folder you can find
**Running the experiment:**
--dark_exp

3) Run synthetic experiment in one dimension. Sample points from gaussian distribution.
 **Running the experiment:**
 --synthetic_exp

#### **flags for the two first experiment:**
 -c this is the cost_factor default value is 5
 -e this is epsilon default value is 0.2
 -s spare cost used only id dark experiment default value is 0
 -th if set hardt model will train again)
 -ts svm loan return train again
 -cv only if train svm loan is set
 
 example: strategic_main_run.py --full_info_exp -c 5 -e 0.2
 
 For the third experiment there are no flags.