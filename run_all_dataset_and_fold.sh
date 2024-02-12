for upstream in fbank; do 
for test_fold in fold6 fold5 fold4 fold3 fold2 fold1; do 
# IMPROVP IMPROVS
if [[ ${test_fold} == "fold6" ]]; then
    for corpus in IMPROVP IMPROVS; do
    python3 run_downstream.py -n ${upstream}_${corpus}_$test_fold -m train -u ${upstream} -d emotion_dev -c downstream/emotion_dev/config_${corpus}.yaml -o "config.downstream_expert.datarc.test_fold='$test_fold'"
    python3 run_downstream.py -m evaluate -e result/downstream/${upstream}_${corpus}_$test_fold/dev-best.ckpt
    done;
# All databases
elif  [[ ${test_fold} == "fold1" ]]; then
    for corpus in IMPROVP CREMAD IEMOCAP NNIME IMPROVS PODCASTP PODCASTS BPODCASTP BPODCASTS; do
    # The default config is "downstream/emotion/config.yaml"
    python3 run_downstream.py -n ${upstream}_${corpus}_$test_fold -m train -u ${upstream} -d emotion_dev -c downstream/emotion_dev/config_${corpus}.yaml -o "config.downstream_expert.datarc.test_fold='$test_fold'"
    python3 run_downstream.py -m evaluate -e result/downstream/${upstream}_${corpus}_$test_fold/dev-best.ckpt
    done;
else
    for corpus in IMPROVP CREMAD IEMOCAP NNIME IMPROVS; do
    # The default config is "downstream/emotion/config.yaml"
    python3 run_downstream.py -n ${upstream}_${corpus}_$test_fold -m train -u ${upstream} -d emotion_dev -c downstream/emotion_dev/config_${corpus}.yaml -o "config.downstream_expert.datarc.test_fold='$test_fold'"
    python3 run_downstream.py -m evaluate -e result/downstream/${upstream}_${corpus}_$test_fold/dev-best.ckpt
    done;
fi
done;
done