## 4: User does not use safeXClassifier, or provide dataset object
### but does provide description of pre-processing, 
### and provides output probabilities for the train and test set they have used (and true classes?) 
- We cannot assume that the TRE has the capability to get the right bits of pre-processing code from their source code.  
- Do we insist on this (would be needed for ‘outside world’)? what if this is commercially sensitive?  
- TRE can in theory run LIRA and worst-case but not attribute inference attacks. 
  - There is a risk that they have misidentified the train/test splits to give us ones which make the classifier look less disclosive 
  - But this probably falls outside our remit? 
- Recommend reject???   
-We could automate generalisation (as lower bound) and  worst case attacks if they give output probabilities 
   – so we need to specify format 
- TRE would need actual copies of processed data to run LIRA 

**THIS would be the version that let people use R **