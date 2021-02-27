# script for wtc3 folder stucture
#test_dir="../../librispeech/s5/data/train_clean_360"
wav_dir="SPIRE_EMA/DataBase/"$1"/Neutral/WavClean"   
output_dir="/SPIRE_EMA/Xvector_Kaldi/$1";
echo "making directory " $output_dir
mkdir $output_dir

for wavfile in `find $wav_dir -name "*.wav"`; do
	name1=`echo $wavfile`;
	echo "computing X-vector for" $name1
	./embedding_extraction_UtteranceLevel.sh ~/kaldi $name1 $output_dir
done

