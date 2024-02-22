# Overview
 The EMO-SUPERB repository is for the anonymous submission to ACL 2024.

 # Installation
 1. The EMO-SUPERB is developed based on [s3prl](https://github.com/s3prl/s3prl#installation) toolkit, please install it first.
    * Please follow the [instructions](https://s3prl.github.io/s3prl/tutorial/installation.html#editable-installation) to do an editable installation.
      ```
      git clone https://github.com/s3prl/s3prl.git
      cd s3prl
      pip install -e .
      ```
2. Move the ```emo-superb``` folder into the path ```s3prl/s3prl/downstream``` and rename the folder as the **"emotion_dev"**
3. Move the ```data folder``` into the path ```s3prl/s3prl/``` 
   * Download WAV files into the folder for each database (e.g., ```data/IEMOCAP/Audios```)by submitting the EULA form for the six databases.
   * [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm)
   * [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
   * [IMPROV](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html)
   * [PODCAST](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)
   * [NNIME](https://biic.ee.nthu.edu.tw/open_resource_detail.php?id=61)
   * [BIIC-PODCAST](https://biic.ee.nthu.edu.tw/open_resource_detail.php?id=63)
# Train and Evaluation
## Trained Models
* All files can be downloaded by the [link](https://drive.google.com/file/d/15qjtVo46N944R5jRlFvKkIXBerwpjn3O/view?usp=sharing).
* Unzip the .zip file and move the folder into the path (s3prl/s3prl/result/)

## Training Models 
### Use the command line. We take the SAIL-IEMOCAP corpus as an example.
```
for upstream in fbank; do 
 for test_fold in fold1 fold2 fold3 fold4 fold5; do
  for corpus in IEMOCAP; do
  # The default config is "downstream/emotion/config.yaml"
  python3 run_downstream.py -n ${upstream}_${corpus}_$test_fold -m train -u ${upstream} -d emotion_dev -c downstream/emotion/config_${corpus}.yaml -o "config.downstream_expert.datarc.test_fold='$test_fold'"
  python3 run_downstream.py -m evaluate -e result/downstream/${upstream}_${corpus}_$test_fold/dev-best.ckpt
  done;
 done;
done
```

### Run All Experiments
```
bash run_all_dataset_and_fold.sh
```

# Relabel by ChatGPT-4
The folder, named **```chatGPT```**, contains the promot (```Prompt.txt```) for ChatGPT and the input and output files.
* The input files (```input_dev.csv``` and ```input_train.csv```) include the file names, distributional primary emotions labels, and typed descriptions from annotators.
* The output files (```output_dev.csv``` and ```output_train.csv```) consist of the file names, adjusted distributional primary emotions labels, and reasons from ChatGPT.
We encourage everyone to contribute their prompt and results.



