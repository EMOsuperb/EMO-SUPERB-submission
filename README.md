# Overview
 The EMO-SUPERB repository is for the anonymous submission to ACL 2024.

 # Installation
 1. The EMO-SUPERB is developed based on [s3prl](https://github.com/s3prl/s3prl#installation) toolkit, please install it first.
    * Please follow the [instrution](https://s3prl.github.io/s3prl/tutorial/installation.html#editable-installation) to do editable installation
      ```
      git clone https://github.com/s3prl/s3prl.git
      cd s3prl
      pip install -e .
      ```
2. Move the ```emo-superb``` folder into the path ```s3prl/s3prl/downstream```
3. Move the ```data folder``` into the path ```s3prl/s3prl/``` 
   * Download wav files into the folder for each database (e.g., ```data/IEMOCAP/Audios```)by submiting the EULA form for the six databases.
   * [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm)
   * [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
   * [IMPROV](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html)
   * [PODCAST](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)
   * [NNIME](https://biic.ee.nthu.edu.tw/open_resource_detail.php?id=61)
   * [BIIC_PODCAST](https://biic.ee.nthu.edu.tw/open_resource_detail.php?id=63)
# Train and Evaluation

## Train (```.sh```)
