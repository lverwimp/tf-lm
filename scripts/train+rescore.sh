#! /bin/bash
# train language model + use it to rescore n-best hypotheses

config=$1 # config file for initial model training
nbest=$2 # n-best hypotheses

if [ $# -ne 2 ]; then
	echo "Usage: <config_file> <n-best_hypotheses_file>"
	exit 1
fi

###### Train language model ######
prefix_lm=`grep "name" $config | sed -e "s/^name[ \t]*//"`
lm="${prefix_lm}.final"


if [ -s $lm ]; then
	echo "Language model $lm already exists. Skipping training."
else
	echo "Train language model: $lm"
	python word_lm.py --config $config
	#python word_lm_v2.py --config $config
fi 

##### Rescore N-best list ######
nbest_full_path=`realpath $nbest`
lm_basename=`basename $lm`
nbest_basename=`basename $nbest`
new_config="${config}.nbest_${nbest_basename}"
result="../results_rescoring/${lm_basename}_${nbest_basename}"

if [ -s $result ]; then
	echo "Results for rescoring $nbest with $lm already exist. Skipping rescoring."
else
	echo "Rescoring $nbest with $lm..."

	if [ -s $new_config ]; then 
		echo "Config file for rescoring already exists: $new_config."
	else
		echo "Make config file for n-best rescoring..."
		cp $config $new_config
		echo "nbest	$nbest_full_path" >> $new_config
		echo "lm	$lm"  >> $new_config
		echo "result	$result" >> $new_config
		sed "s/\.log/_nbest\.log/" $new_config > ${new_config}.tmp
		mv ${new_config}.tmp $new_config
	fi
	
	echo "Calculate probabilities for hypotheses..."
	python word_lm_rescore_nbest.py --config $new_config
fi









