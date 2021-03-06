#!/bin/bash
# usage: ./embedding_extraction_UtteranceLevel.sh ~/kaldi s010Nl03f0010.wav tmp/

' Date created: Jul 7 2018

This script performs diarization using x-vectors as feature representations (https://david-ryan-snyder.github.io/2017/10/04/model_sre16_v2.html). Initially, this will be a wrapper for the default pipeline as provided in kaldi/egs/callhome_diairzation/v2/run.sh (xvectors -> plda -> AHC). 

The PLDA model used in this script is NOT provided as part of the x-vector model and must be created following prepare.sh

Later, viterbi re-alignment might be explored '

currDir=$PWD

# Inputs:
wavList=$PWD/wavList							# File with list of audio files to be diarized
echo "$wavList"
inputVADDir=$PWD/vad/kaldiVAD		# Directory with VAD files; labels@frame-level every line
echo "$inputVADDir"
# Other params
kaldiDir=$1
ivectorbin_path=$kaldiDir/src/ivectorbin
wavFile=$2
#rttmFile=$3
OutDir=$3
voxcelebDir=$kaldiDir/egs/voxceleb/v2/
callhomeDir=$kaldiDir/egs/callhome_diarization/v2/
dataDir=$currDir/tmpkaldidir
workDir=$currDir/data$1/tmpkaldidir_cmn
nnet_dir=$PWD/exp/xvector_nnet_1a/

rm -rf $dataDir; mkdir $dataDir
rm -rf $currDir/diarDir; mkdir $currDir/diarDir

cd $callhomeDir
. cmd.sh
. path.sh
cd $currDir

# Create kaldi directory
#python $currDir/vad_file.py --wavFile $wavFile --rttmFile $rttmFile
echo $2 |xargs readlink -f >$PWD/wavList
echo "Check1"
echo $wavList
paste -d ' ' <(rev $wavList | cut -f 1 -d '/' | rev | sed "s/\.wav$/-rec/g") <(cat $wavList | xargs readlink -f) > $dataDir/wav.scp
paste -d ' ' <(cut -f 1 -d ' ' $dataDir/wav.scp | sed "s/-rec$//g") <(cut -f 1 -d ' ' $dataDir/wav.scp | sed "s/-rec$//g") > $dataDir/utt2spk
cp $dataDir/utt2spk $dataDir/spk2utt
numUtts=`wc -l $dataDir/utt2spk | cut -f 1 -d ' '`
paste -d ' ' <(cut -f 1 -d ' ' $dataDir/utt2spk) <(cut -f 1 -d ' ' $dataDir/wav.scp) <(yes "0" | head -n $numUtts) <(cat $wavList | xargs soxi -D) > $dataDir/segments

# paste -d ' ' <(rev $wavList | cut -f 1 -d '/' | rev | sed "s/\.wav$//g") <(yes "4" | head -n $numUtts) > $dataDir/reco2num_spk

:<<'END1'
# Convert the supplied VAD into kaldi format and prepare the feats for x-vectors
while read -r line; do
	uttID=`echo $line | cut -f 1 -d ' '`
	inVadFile=$inputVADDir/$uttID.csv
	[ ! -f $inVadFile ] && { echo "Input vad file does not exist"; exit 0; }
	paste -d ' ' <(echo $uttID) <(cut -f 2 -d ',' $inVadFile | tr "\n" " " | sed "s/^/ [ /g" | sed "s/$/ ]/g") >> $dataDir/vad.txt
done < $dataDir/utt2spk
copy-vector ark,t:$dataDir/vad.txt ark,scp:$dataDir/vad.ark,$dataDir/vad.scp
END1



[ "$numUtts" -gt 4 ] && nj=4 || nj=1


cd $voxcelebDir
train_cmd=run.pl
# Feature processing pipeline
utils/fix_data_dir.sh $dataDir
bash steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" --mfcc-config conf/mfcc.conf --write-utt2num-frames true $dataDir/ > /dev/null

## VAD from Voxcelb recipe ##

#bash sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
 #     $dataDir/  exp/make_vad $dataDir/vad
#cp $dataDir/vad/vad_tmpkaldidir.1.scp $dataDir/vad.scp
#cp $dataDir/vad/vad_tmpkaldidir.1.ark $dataDir/vad.ark

## Inside code of VoxCelb recipe ##

#vad_config=conf/vad.conf
#compute-vad --config=$vad_config scp:$dataDir/feats.scp \
 # ark,scp:$dataDir/vad.ark,$dataDir/vad.scp || exit 1
#~/kaldi/src/bin/copy-vector ark:$dataDir/vad.ark ark,t:$dataDir/vad.txt 


## Direct VAD

$ivectorbin_path/compute-vad scp:$dataDir/feats.scp ark,t:$dataDir/vad.txt
~/kaldi/src/bin/copy-vector ark,t:$dataDir/vad.txt ark,scp:$dataDir/vad.ark,$dataDir/vad.scp


#utils/fix_data_dir.sh $dataDir

sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj $nj \
    $nnet_dir $dataDir  \
    $dataDir/xvectors/ > /dev/null

#    $nnet_dir/xvectors_train
outname=`echo $2|cut -d'.' -f1|rev|cut -d'/' -f1|rev`  #echo $2|cut -d'.' -f1`
~/kaldi/src/bin/copy-vector ark:$dataDir/xvectors/xvector.1.ark ark,t:$OutDir/$outname.txt
sed -i 's/.*\[\([^]]*\)\].*/\1/g' $OutDir/$outname.txt

:<<'END'
cd $callhomeDir
. path.sh

diarization/vad_to_segments.sh --nj $nj --cmd "$train_cmd" --min-duration 0.5 --segmentation-opts '-silence-proportion 0.011' $dataDir $dataDir/segmented > /dev/null

local/nnet3/xvector/prepare_feats.sh --nj $nj --cmd "$train_cmd" $dataDir/segmented $dataDir/segmented_cmn $dataDir/segmented_cmn/data > /dev/null
cp $dataDir/segmented/segments $dataDir/segmented_cmn

diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" --nj $nj --window 1.5 --period 0.25 --apply-cmn false --min-segment 0.5 $nnet_dir $dataDir/segmented_cmn $dataDir/xvectors/ > /dev/null
END


